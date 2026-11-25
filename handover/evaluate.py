# -*- coding: utf-8 -*-
"""Unified evaluation harness for Chapter 4: traditional baselines, DRL models, ablations, and frozen MAPPO closed-loop."""

import argparse
import json
import os
import time
import numpy as np
import torch

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv
from handover.baselines import MaxRVTPolicy, ISAHOPolicy, SwapHOPolicy
from handover.train import ALGO_SPECS, build_agent


def run_policy_episodes(
    policy,
    cfg: HandoverConfig,
    seeds: list[int],
    duration_s: int,
    service_mode: str = "surrogate",
    force_only_trigger: bool = False,
    traffic_aware_trigger: bool = True,
    micro_slots_override: int | None = None,
) -> dict:
    """Evaluate a handover policy over matched episode seeds and return mean + 95% CI metrics."""
    per_seed = []
    decision_ms_list = []

    for sd in seeds:
        env = DynamicLEOHandoverEnv(
            cfg,
            service_mode=service_mode,
            force_only_trigger=force_only_trigger,
            traffic_aware_trigger=traffic_aware_trigger,
        )
        policy.reset()
        out = env.reset(seed=sd, duration_s=duration_s)
        t_dec = 0.0
        n_dec_steps = 0
        while not out.done:
            t0 = time.perf_counter()
            prop, fb = policy.select_actions(out, env)
            t_dec += (time.perf_counter() - t0) * 1000.0
            n_dec_steps += 1
            out = env.step(
                prop,
                fallback_rankings=fb,
                micro_slots_override=micro_slots_override,
                bypass_trigger_gate=policy.bypass_trigger_gate,
            )
        m = env.get_episode_summary()
        m["decision_ms_per_step"] = t_dec / max(n_dec_steps, 1)
        per_seed.append(m)
        decision_ms_list.append(m["decision_ms_per_step"])

    keys = list(per_seed[0].keys())
    summary = {"mean": {}, "ci95": {}, "seeds": seeds, "cases": per_seed}
    n = len(per_seed)
    for k in keys:
        vals = np.array([c[k] for c in per_seed], dtype=np.float64)
        summary["mean"][k] = float(vals.mean())
        summary["ci95"][k] = float(1.96 * vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chapter 4 handover algorithms")
    parser.add_argument("--preset", type=str, default="tight_45n")
    parser.add_argument("--service-mode", type=str, default="surrogate", choices=["surrogate", "mappo_local", "mappo_coop", "uswgp_micro"])
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--shadow-scale", type=float, default=1.0)
    parser.add_argument("--tau-ho", type=float, default=30.0)
    parser.add_argument("--hotspot-drift", action="store_true")
    parser.add_argument("--micro-slots", type=int, default=0, help="Override micro slots/sec in Mode B (0=full 500)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--ckpt-dir", type=str, default="handover/checkpoints")
    parser.add_argument("--drl-algos", nargs="*", default=["cell_rl", "ma_hgdqn", "tra_hgdrl"])
    parser.add_argument("--out-json", type=str, default="handover/results/eval_results.json")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    cfg = HandoverConfig(
        constellation_name=args.preset,
        episode_duration_s=args.duration,
        num_users=args.num_users,
        input_rate_mbps=args.rate,
        shadow_sigma_scale=args.shadow_scale,
        tau_ho_ms=args.tau_ho,
        enable_hotspot_drift=args.hotspot_drift,
    )
    micro_override = args.micro_slots if args.micro_slots > 0 else None

    entries: list[tuple[str, object, bool, bool]] = [
        ("Max-RVT", MaxRVTPolicy(), False, True),
        ("ISA-HO", ISAHOPolicy(reassoc_period_s=15), False, True),
        ("Swap-HO", SwapHOPolicy(), False, True),
    ]
    for key in args.drl_algos:
        ckpt_path = os.path.join(args.ckpt_dir, f"{key}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Skip] checkpoint not found: {ckpt_path}")
            continue
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        agent, spec = build_agent(key, cfg, twin=bool(ckpt_meta.get("twin", False)))
        agent.load(ckpt_path)
        agent.epsilon = 0.0
        entries.append((spec["display"], agent, spec["force_only"], spec["traffic_aware"]))

    print(
        f"\n=== Chapter 4 Evaluation | preset={args.preset} mode={args.service_mode} "
        f"T={args.duration}s U={args.num_users} rate={args.rate}Mbps drift={args.hotspot_drift} ==="
    )
    all_res = {}
    for label, pol, force_only, traffic_aware in entries:
        res = run_policy_episodes(
            pol,
            cfg,
            seeds=args.seeds,
            duration_s=args.duration,
            service_mode=args.service_mode,
            force_only_trigger=force_only,
            traffic_aware_trigger=traffic_aware,
            micro_slots_override=micro_override,
        )
        all_res[label] = res
        m = res["mean"]
        print(
            f"  {label:11s} | HO={m['total_handovers']:6.1f} | PP={m['ping_pong_rate']*100:5.2f}% | "
            f"Out={m['outage_rate']*100:5.2f}% | RT_drop={m['rt_drop_rate']*100:5.2f}% | "
            f"RT_del={m['rt_delivered_delay_ms']:4.1f}ms | Tput={m['effective_tput_mbps']:5.2f}Mbps | "
            f"LoadVar={m['load_variance']:.3f} | Jain={m['jain_fairness']:.3f} | "
            f"Dec={m['decision_ms_per_step']:.2f}ms"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "preset": args.preset,
                    "service_mode": args.service_mode,
                    "duration": args.duration,
                    "num_users": args.num_users,
                    "rate": args.rate,
                    "shadow_scale": args.shadow_scale,
                    "tau_ho": args.tau_ho,
                    "hotspot_drift": args.hotspot_drift,
                    "seeds": args.seeds,
                },
                "results": all_res,
            },
            f,
            indent=2,
        )
    print(f"\nSaved evaluation report -> {args.out_json}")


if __name__ == "__main__":
    main()
