# -*- coding: utf-8 -*-
"""Evaluate trained .pt checkpoints and heuristic baselines across 10-50 Mbps."""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from env import LEOBHEnv
from heuristics import PeriodicBH, QLFBeamHopping, USWGPBeamHopping
from schedule_search import best_global_swap
from scipy.stats import t as student_t
from models import MAPPOActor, PPOBHModel, QMIXAgent, MADDPGActor


def equal_power_np(b_mask: np.ndarray) -> np.ndarray:
    c = b_mask.sum(axis=1, keepdims=True)
    return np.where(c > 0, b_mask / np.maximum(c, 1.0), 0.0)


class MAPPOPolicyWrapper:
    def __init__(self, ckpt_path: str, bc_only: bool = False):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cross_satellite = ckpt.get("cross_satellite", False)
        self.power_conditioned = ckpt.get("power_conditioned", False)
        self.coop_power_gnn = ckpt.get("coop_power_gnn", False)
        if self.coop_power_gnn:
            self.cross_satellite = True
        state = ckpt.get("best_actor_state_dict") or ckpt["actor_state_dict"]
        if bc_only:
            state = ckpt["bc_actor_state_dict"]
        self.actor = MAPPOActor(
            node_dim=state["node_proj.weight"].shape[1],
            power_conditioned=self.power_conditioned,
            coop_power_gnn=self.coop_power_gnn,
        )
        self.actor.load_state_dict(state)
        self.actor.eval()
        self.h = None

    def reset(self):
        self.h = torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)

    @torch.no_grad()
    def act(self, obs: dict):
        e_mat = torch.from_numpy(obs["E_mat"]) if (self.coop_power_gnn and "E_mat" in obs) else None
        b, p, _, _, _, self.h = self.actor.act(
            torch.from_numpy(obs["F_rich"][..., :self.actor.node_proj.in_features].copy()),
            torch.from_numpy(obs["I_mat"]),
            torch.from_numpy(obs["Omega"]),
            torch.from_numpy(obs["C"]),
            torch.from_numpy(obs["real_mask"]),
            torch.from_numpy(obs["action_mask"]),
            self.h,
            deterministic=True,
            E_mat=e_mat,
        )
        return b.numpy(), p.numpy()


class PPOPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = PPOBHModel()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        b, p, _, _, _, _ = self.model.act(
            torch.from_numpy(obs["s"]).unsqueeze(0),
            torch.from_numpy(obs["F"]).unsqueeze(0),
            torch.from_numpy(obs["action_mask"]),
            deterministic=True,
        )
        return b.numpy(), p.numpy()


class QMIXPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.agent = QMIXAgent()
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.agent.eval()
        self.h = None

    def reset(self):
        self.h = torch.zeros(1, cfg.N, 128)

    @torch.no_grad()
    def act(self, obs: dict):
        q_vals, self.h = self.agent(torch.from_numpy(obs["F"]), self.h)
        q_np = q_vals.numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                chosen = avail[np.argsort(-q_np[n, avail])[:k]]
                b[n, chosen] = 1.0
        return b, equal_power_np(b)


class MADDPGPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.actor = MADDPGActor()
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        logits = self.actor(torch.from_numpy(obs["F"])).numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                chosen = avail[np.argsort(-logits[n, avail])[:k]]
                b[n, chosen] = 1.0
        return b, equal_power_np(b)


def benchmark_mappo_latency(policy: MAPPOPolicyWrapper, env: LEOBHEnv):
    torch.set_num_threads(1)
    obs = env.reset(input_rate_mbps=40.0, num_users=cfg.EVAL_USERS, seed=999)
    policy.reset()
    for _ in range(30):
        policy.act(obs)
    t0 = time.perf_counter()
    for _ in range(300):
        policy.act(obs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / 300.0


def paired_differences(results):
    metrics = ("throughput_mbps", "queue_age_slots", "delivered_rt_age_slots", "rt_expired_pct")
    differences = {}
    for name, entry in results.items():
        refs = {"MAPPO-BH", "USWGP-BH"}
        if name.endswith("+Swap"):
            refs.update((name.removesuffix("+Swap"), "USWGP-BH+Swap"))
            if name == "MAPPO-BH+Swap":
                refs.add("MAPPO-BC+Swap")
        if f"{name}-BC" in results:
            refs.add(f"{name}-BC")
        if name == "MAPPO-BH" and "MAPPO-BC" in results:
            refs.add("MAPPO-BC")
        for ref in sorted((refs & results.keys()) - {name}):
            paired = []
            for rate, cases, base_cases in zip(entry["rates"], entry["cases"], results[ref]["cases"]):
                by_seed = {case["seed"]: case for case in base_cases}
                stats = {"rate_mbps": rate}
                for metric in metrics:
                    delta = np.array([case[metric] - by_seed[case["seed"]][metric] for case in cases])
                    half = float(student_t.ppf(0.975, delta.size - 1) * delta.std(ddof=1) / np.sqrt(delta.size))
                    mean = float(delta.mean())
                    stats[metric] = {"mean": mean, "ci95": [mean - half, mean + half]}
                paired.append(stats)
            differences[f"{name} vs {ref}"] = paired
    return differences


def plot_results(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    styles = {
        "P-BH": ("--", "x"),
        "QLF-BH": (":", "v"),
        "USWGP-BH": ("-.", "*"),
        "PPO-BH": ("--", "o"),
        "QMIX-BH": ("-", "D"),
        "MADDPG-BH": (":", "^"),
        "MAPPO-BH": ("-", "s"),
    }
    for name, res in results.items():
        if name.endswith("-BC"):
            continue
        ls, mk = styles.get(name, ("--", "d"))
        axes[0].errorbar(res["rates"], res["tput_mean"], yerr=res["tput_ci"], linestyle=ls, marker=mk, capsize=3, label=name)
        axes[1].errorbar(res["rates"], res["delay_mean"], yerr=res["delay_ci"], linestyle=ls, marker=mk, capsize=3, label=name)
    axes[0].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[0].set_ylabel("Throughput per user (Mbps)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[1].set_ylabel("Average RT queueing delay (time slots)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bh_comparison.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--test-cases", type=int, default=20)
    parser.add_argument("--slots", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--rates", nargs="+", type=float, default=cfg.EVAL_RATES_MBPS)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--mappo-variant", action="append", nargs=2, default=[], metavar=("LABEL", "CHECKPOINT"))
    parser.add_argument("--include-bc", action="store_true")
    parser.add_argument("--search-initializers", "--search-seeds", nargs="+", default=[], metavar="POLICY",
                        help="Compare the same global swap search initialized by each listed policy")
    parser.add_argument("--search-steps", type=int, default=1)
    parser.add_argument("--policies", nargs="+", default=[], metavar="POLICY",
                        help="Evaluate only the listed policies")
    parser.add_argument("--orbit", type=str, default="planar", choices=["planar", "tle"],
                        help="Orbital geometry mode: planar (4x4 grid) or tle (real Starlink SGP4 + WGS-84 ECEF)")
    args = parser.parse_args()
    if args.test_cases < 2 or args.slots < 1 or args.search_steps < 1:
        parser.error("Require at least two test cases and positive slot and search counts")

    os.makedirs(args.out_dir, exist_ok=True)
    policies = {
        "P-BH": PeriodicBH(),
        "QLF-BH": QLFBeamHopping(),
        "USWGP-BH": USWGPBeamHopping(),
        "PPO-BH": PPOPolicyWrapper(os.path.join(args.ckpt_dir, "ppo_bh.pt")),
        "QMIX-BH": QMIXPolicyWrapper(os.path.join(args.ckpt_dir, "qmix_bh.pt")),
        "MADDPG-BH": MADDPGPolicyWrapper(os.path.join(args.ckpt_dir, "maddpg_bh.pt")),
        "MAPPO-BH": MAPPOPolicyWrapper(os.path.join(args.ckpt_dir, "mappo_bh.pt")),
    }

    if args.include_bc:
        policies["MAPPO-BC"] = MAPPOPolicyWrapper(os.path.join(args.ckpt_dir, "mappo_bh.pt"), bc_only=True)
    for name, path in args.mappo_variant:
        if name in policies:
            parser.error(f"Duplicate policy label: {name}")
        policies[name] = MAPPOPolicyWrapper(path)
        if args.include_bc:
            policies[f"{name}-BC"] = MAPPOPolicyWrapper(path, bc_only=True)
    for name in args.search_initializers:
        if name not in policies:
            parser.error(f"Unknown search initializer: {name}")
        search_name = f"{name}+Swap"
        if search_name in policies:
            parser.error(f"Duplicate policy label: {search_name}")
        policies[search_name] = policies[name]

    cross_satellite = any(getattr(policy, "cross_satellite", False) for policy in policies.values())
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=0, cross_satellite=cross_satellite, orbit_mode=args.orbit)
    lat_ms = benchmark_mappo_latency(policies["MAPPO-BH"], env)
    print(f"[Latency] MAPPO-BH single-thread 16-sat inference: {lat_ms:.3f} ms/slot (300-slot avg)")
    if args.policies:
        unknown = set(args.policies) - policies.keys()
        if unknown:
            parser.error(f"Unknown policies: {', '.join(sorted(unknown))}")
        policies = {name: policies[name] for name in args.policies}

    results = {name: {"rates": args.rates, "tput_mean": [], "tput_ci": [],
                      "delay_mean": [], "delay_ci": [], "cases": []} for name in policies}

    for rate in args.rates:
        print(f"\n=== Evaluating @ {rate:g} Mbps ({args.test_cases} test cases x {args.slots} slots) ===")
        for name, pol in policies.items():
            cases = []
            for tc in range(args.test_cases):
                seed = args.seed_start + tc
                obs = env.reset(input_rate_mbps=rate, num_users=cfg.EVAL_USERS, seed=seed)
                pol.reset()
                tot_bits = d_sum = delivered_rt = delivered_age = expired_rt = 0.0
                decision_ns = []
                warmup = 20
                for step_i in range(args.slots + warmup):
                    begin = time.perf_counter_ns()
                    b, p = pol.act(obs)
                    if name.endswith("+Swap"):
                        for _ in range(args.search_steps):
                            b, p = best_global_swap(env, b, p)
                    elapsed = time.perf_counter_ns() - begin
                    obs, _, info = env.step(b, p)
                    if step_i >= warmup:
                        decision_ns.append(elapsed)
                        tot_bits += info["zeta_t"]
                        d_sum += info["gamma_t"]
                        delivered_rt += info["served_rt"]
                        delivered_age += info["served_rt_age"]
                        expired_rt += info["expired_rt"]
                cases.append({
                    "seed": seed,
                    "throughput_mbps": tot_bits / (args.slots * cfg.T_SLOT * env.U * 1e6),
                    "queue_age_slots": d_sum / args.slots,
                    "delivered_rt_age_slots": delivered_age / max(delivered_rt, 1e-9),
                    "rt_expired_pct": 100 * expired_rt / max(expired_rt + delivered_rt, 1e-9),
                    "decision_ms": float(np.mean(decision_ns) / 1e6),
                })
            stats = results[name]
            stats["cases"].append(cases)
            for prefix, metric in [("tput", "throughput_mbps"), ("delay", "queue_age_slots")]:
                sample = np.array([case[metric] for case in cases])
                stats[f"{prefix}_mean"].append(float(sample.mean()))
                stats[f"{prefix}_ci"].append(float(1.96 * sample.std(ddof=1) / np.sqrt(sample.size)))
            print(f"  {name:22s} {stats['tput_mean'][-1]:6.2f} Mbps  {stats['delay_mean'][-1]:5.2f} slots"
                  f"  expired {np.mean([c['rt_expired_pct'] for c in cases]):5.2f}%"
                  f"  decision {np.mean([c['decision_ms'] for c in cases]):6.3f} ms", flush=True)

    settings = {
        "association": env.assoc_method,
        "orbit_mode": args.orbit,
        "user_count": env.U,
        "satellite_spacing_km": cfg.SAT_SPACING_M / 1000,
        "cases": args.test_cases,
        "seed_start": args.seed_start,
        "slots": args.slots,
        "warmup_slots": warmup,
        "search_steps": args.search_steps if args.search_initializers else 0,
    }
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump({"settings": settings, "mappo_single_thread_ms": lat_ms, "results": results,
                   "paired_differences": paired_differences(results)}, f, indent=2)

    plot_results(results, args.out_dir)
    print(f"\nSaved results to {args.out_dir}/eval_results.json and {args.out_dir}/bh_comparison.png")


if __name__ == "__main__":
    main()
