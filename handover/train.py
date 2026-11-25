# -*- coding: utf-8 -*-
"""Two-stage training pipeline for TRA-HGDRL, MA-HGDQN, Cell-RL, and all 5 ablation variants."""

import argparse
import copy
import json
import os
import time
import numpy as np
import torch

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv
from handover.graph import build_cell_graph_obs_batch
from handover.reward import compute_step_rewards
from handover.replay_buffer import SMDPTracker, StratifiedSMDPReplayBuffer
from handover.hgat_qrdqn import DRLHandoverAgent


ALGO_SPECS: dict[str, dict] = {
    "tra_hgdrl": dict(display="TRA-HGDRL", use_hgat=True, distributional=True, use_cvar=True, traffic_aware=True, force_only=False),
    "ma_hgdqn": dict(display="MA-HGDQN", use_hgat=True, distributional=False, use_cvar=False, traffic_aware=True, force_only=False),
    "cell_rl": dict(display="Cell-RL", use_hgat=False, distributional=False, use_cvar=False, traffic_aware=True, force_only=False),
    "wo_traffic": dict(display="w/o Traffic", use_hgat=True, distributional=True, use_cvar=False, traffic_aware=False, force_only=False),
    "wo_hgat": dict(display="w/o HGAT", use_hgat=False, distributional=True, use_cvar=True, traffic_aware=True, force_only=False),
    "wo_distrl": dict(display="w/o DistRL", use_hgat=True, distributional=False, use_cvar=False, traffic_aware=True, force_only=False),
    "wo_risk": dict(display="w/o Risk", use_hgat=True, distributional=True, use_cvar=False, traffic_aware=True, force_only=False),
    "wo_trigger": dict(display="w/o Trigger", use_hgat=True, distributional=True, use_cvar=True, traffic_aware=True, force_only=True),
}


def build_agent(algo_key: str, cfg: HandoverConfig, twin: bool = False, device: str = "cpu") -> tuple[DRLHandoverAgent, dict]:
    if algo_key not in ALGO_SPECS:
        raise ValueError(f"Unknown algo '{algo_key}'. Available: {list(ALGO_SPECS)}")
    spec = ALGO_SPECS[algo_key]
    agent = DRLHandoverAgent(
        cfg=cfg,
        name=spec["display"],
        use_hgat=spec["use_hgat"],
        distributional=spec["distributional"],
        use_cvar=spec["use_cvar"],
        traffic_aware=spec["traffic_aware"],
        twin=twin,
        device=device,
    )
    return agent, spec


def evaluate_agent(
    agent: DRLHandoverAgent,
    cfg: HandoverConfig,
    spec: dict,
    seeds: tuple[int, ...] = (9001, 9002),
    duration_s: int = 300,
) -> tuple[float, dict[str, float]]:
    """Run deterministic validation episodes and return (mean_reward_sum, averaged_metrics)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    env = DynamicLEOHandoverEnv(
        cfg,
        service_mode="surrogate",
        force_only_trigger=spec["force_only"],
        traffic_aware_trigger=spec["traffic_aware"],
    )
    tot_rews = []
    metric_list = []
    for sd in seeds:
        out = env.reset(seed=sd, duration_s=duration_s)
        ep_r = 0.0
        while not out.done:
            prev_out = out
            prop, fb = agent.select_actions(out, env)
            out = env.step(prop, fallback_rankings=fb)
            r_tot, _, _ = compute_step_rewards(prev_out, out, env)
            ep_r += float(r_tot.mean())
        tot_rews.append(ep_r)
        metric_list.append(env.get_episode_summary())
    agent.epsilon = old_eps
    avg_m = {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}
    return float(np.mean(tot_rews)), avg_m


def train_algo(
    algo_key: str,
    cfg: HandoverConfig,
    episodes: int = 40,
    train_duration_s: int = 240,
    val_duration_s: int = 300,
    updates_per_ep: int = 80,
    save_dir: str = "handover/checkpoints",
    seed: int = 42,
    twin: bool = False,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train a DRL handover algorithm with two-stage risk-neutral -> adaptive CVaR schedule."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    agent, spec = build_agent(algo_key, cfg, twin=twin, device=device)
    env = DynamicLEOHandoverEnv(
        cfg,
        service_mode="surrogate",
        force_only_trigger=spec["force_only"],
        traffic_aware_trigger=spec["traffic_aware"],
    )
    tracker = SMDPTracker(cfg.M, gamma=cfg.gamma, max_step=cfg.smdp_max_step)
    replay = StratifiedSMDPReplayBuffer(capacity=80000, ho_sample_frac=0.75)

    # Two-stage CVaR schedule: first 40% episodes risk-neutral warmup (blend=0), next 30% linear anneal -> 1.0
    warmup_eps = max(1, int(round(episodes * 0.40)))
    anneal_eps = max(1, int(round(episodes * 0.30)))

    best_val_ret = -1e18
    best_state = copy.deepcopy(agent.net.state_dict())
    history = []
    t0 = time.perf_counter()

    # Diverse training load & user-count sampling for robust generalization
    train_rates = [20.0, 25.0, 28.0, 32.0, 38.0]
    train_users = [180, 200, 240, 280]

    for ep in range(1, episodes + 1):
        # Epsilon exponential decay: 0.60 -> 0.04
        frac = (ep - 1) / max(episodes - 1, 1)
        agent.epsilon = float(max(0.04, 0.60 * (0.06 ** frac)))

        # Two-stage CVaR annealing
        if agent.use_cvar:
            if ep <= warmup_eps:
                agent.cvar_blend = 0.0
            else:
                agent.cvar_blend = float(min(1.0, (ep - warmup_eps) / float(anneal_eps)))
        else:
            agent.cvar_blend = 0.0

        ep_rate = float(rng.choice(train_rates))
        ep_users = int(rng.choice(train_users))
        out = env.reset(seed=seed * 100 + ep, duration_s=train_duration_s, num_users=ep_users, input_rate_mbps=ep_rate)
        tracker.reset()

        ep_rew_sum = 0.0
        while not out.done:
            prev_out = out
            prop, fb, decisions = agent.select_actions_with_obs(out, env, rng=rng)
            for m, (obs_m, act_m) in decisions.items():
                tracker.record_decision(m, obs_m, act_m)

            out = env.step(prop, fallback_rankings=fb)
            r_tot, _, _ = compute_step_rewards(prev_out, out, env)
            ep_rew_sum += float(r_tot.mean())
            tracker.accumulate_step_rewards(r_tot)

            # Finalize SMDP transitions for cells that triggered again, hit max_step, or finished episode
            ready_cells = tracker.pop_ready_cells(out.trigger.i_trig, out.done)
            if ready_cells:
                next_obs_list = build_cell_graph_obs_batch(ready_cells, out, env, traffic_aware=spec["traffic_aware"])
                for m, next_obs_m in zip(ready_cells, next_obs_list):
                    tr = tracker.finalize(m, next_obs_m, out.done)
                    if tr is not None:
                        replay.push(tr)

        # Off-policy gradient updates
        losses = []
        if len(replay) >= cfg.batch_size:
            for _ in range(updates_per_ep):
                batch_np = replay.sample(cfg.batch_size, rng)
                losses.append(agent.update(batch_np))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        ep_sum = env.get_episode_summary()
        rec = {
            "episode": ep,
            "train_ret": ep_rew_sum,
            "loss": mean_loss,
            "epsilon": agent.epsilon,
            "cvar_blend": agent.cvar_blend,
            "buffer_size": len(replay),
            "ho": ep_sum["total_handovers"],
            "outage": ep_sum["outage_rate"],
            "rt_drop": ep_sum["rt_drop_rate"],
            "tput": ep_sum["effective_tput_mbps"],
            "load_var": ep_sum["load_variance"],
        }

        val_interval = episodes if episodes <= 5 else max(1, episodes // 5)
        if ep % val_interval == 0 or ep == episodes:
            val_ret, val_m = evaluate_agent(agent, cfg, spec, seeds=(9001, 9002), duration_s=val_duration_s)
            rec["val_ret"] = val_ret
            rec["val_metrics"] = val_m
            # Save best after warmup (or anytime if not CVaR)
            if (not agent.use_cvar or ep >= warmup_eps) and val_ret > best_val_ret:
                best_val_ret = val_ret
                best_state = copy.deepcopy(agent.net.state_dict())
            if verbose:
                print(
                    f"[{spec['display']:11s}] ep {ep:3d}/{episodes} | loss={mean_loss:6.3f} | "
                    f"cvar={agent.cvar_blend:.2f} | val_ret={val_ret:6.1f} | "
                    f"HO={val_m['total_handovers']:4.0f} PP={val_m['ping_pong_rate']*100:4.1f}% "
                    f"Out={val_m['outage_rate']*100:4.2f}% Drop={val_m['rt_drop_rate']*100:4.1f}% "
                    f"Tput={val_m['effective_tput_mbps']:5.2f} Var={val_m['load_variance']:.3f} | "
                    f"{time.perf_counter()-t0:.1f}s"
                )
        history.append(rec)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{algo_key}.pt")
    agent.net.load_state_dict(best_state)
    agent.cvar_blend = 1.0 if agent.use_cvar else 0.0
    agent.save(ckpt_path, extra={"algo_key": algo_key, "best_val_ret": best_val_ret, "history": history})
    if verbose:
        print(f"Saved best checkpoint -> {ckpt_path} (best_val_ret={best_val_ret:.2f})")
    return {"algo_key": algo_key, "ckpt_path": ckpt_path, "best_val_ret": best_val_ret, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train Chapter 4 TRA-HGDRL and DRL baselines/ablations")
    parser.add_argument("--algos", nargs="+", default=["tra_hgdrl"], choices=list(ALGO_SPECS.keys()))
    parser.add_argument("--preset", type=str, default="tight_45n")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--train-duration", type=int, default=240)
    parser.add_argument("--val-duration", type=int, default=300)
    parser.add_argument("--updates-per-ep", type=int, default=80)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--twin", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="handover/checkpoints")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    cfg = HandoverConfig(
        constellation_name=args.preset,
        num_users=args.num_users,
        input_rate_mbps=args.rate,
    )
    summaries = {}
    for i, algo in enumerate(args.algos):
        summaries[algo] = train_algo(
            algo_key=algo,
            cfg=cfg,
            episodes=args.episodes,
            train_duration_s=args.train_duration,
            val_duration_s=args.val_duration,
            updates_per_ep=args.updates_per_ep,
            save_dir=args.save_dir,
            seed=args.seed + i,
            twin=args.twin,
        )

    out_json = os.path.join(args.save_dir, "train_summary.json")
    with open(out_json, "w") as f:
        json.dump({k: {"best_val_ret": v["best_val_ret"], "ckpt_path": v["ckpt_path"]} for k, v in summaries.items()}, f, indent=2)


if __name__ == "__main__":
    main()
