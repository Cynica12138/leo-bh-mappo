# -*- coding: utf-8 -*-
"""Batch paper evaluation runner for Chapter 4 TRA-HGDRL.

Runs all 6 paper experiment suites under matched seeds and unified physical/admission rules:
  1. main          : 600 s primary benchmark (5 seeds, tight_45n, 6 algorithms)
  2. ablation      : 600 s ablation study (5 seeds, standard + hotspot-drift stress, TRA-HGDRL + 5 ablations)
  3. load_sweep    : 600 s offered-rate sweep [20, 28, 36, 44] Mbps + hotspot-drift (3 seeds, 6 algorithms)
  4. robustness    : 600 s shadowing scale [0.67, 1.0, 1.5, 2.0] & tau_HO [0, 15, 30, 50] ms (3 seeds, 6 algorithms)
  5. constellation : 600 s constellation density generalization (tight_45n, standard_47n, dense_35n; 3 seeds)
  6. closed_loop   : Mode B cross-timescale closed-loop coupled with frozen Chapter 3 MAPPO-APG-Local (3 seeds)
"""

import argparse
import json
import os
import time
import torch

from handover.config import HandoverConfig
from handover.baselines import (
    MaxRVTPolicy,
    ISAHOPolicy,
    SwapHOPolicy,
    RAHOPolicy,
    MCMFHOPolicy,
    RHSTGPolicy,
)
from handover.evaluate import run_policy_episodes
from handover.mappo_ho import MAPPOHOAgent
from handover.train import build_agent


ALL_MAIN_ORDER = [
    "Max-RVT",
    "ISA-HO",
    "Swap-HO",
    "RA-HO",
    "MCMF-HO",
    "RH-STG",
    "MAPPO-HO",
    "Hyst-MADQN",
    "Cell-RL",
    "MA-HGDQN",
    "TRA-HGDRL",
]
MAIN_DRL_KEYS = ["mappo_ho", "hyst_madqn", "cell_rl", "ma_hgdqn", "tra_hgdrl"]
ABLATION_KEYS = ["tra_hgdrl", "wo_traffic", "wo_hgat", "wo_distrl", "wo_risk", "wo_trigger"]


def _load_drl_entries(ckpt_dir: str, cfg: HandoverConfig, keys: list[str]) -> list[tuple[str, object, bool, bool]]:
    entries = []
    for key in keys:
        ckpt_path = os.path.join(ckpt_dir, f"{key}.pt")
        if not os.path.exists(ckpt_path) and key == "wo_distrl":
            # w/o DistRL has identical architecture/spec to MA-HGDQN if separate file wasn't trained
            alt = os.path.join(ckpt_dir, "ma_hgdqn.pt")
            if os.path.exists(alt):
                ckpt_path = alt
        if not os.path.exists(ckpt_path):
            print(f"[Warn] Missing checkpoint for {key}: {ckpt_path} (skipping)", flush=True)
            continue
        if key == "mappo_ho":
            agent = MAPPOHOAgent(cfg=cfg)
            agent.load(ckpt_path)
            entries.append(("MAPPO-HO", agent, False, True))
        else:
            meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            agent, spec = build_agent(key, cfg, twin=bool(meta.get("twin", True)))
            agent.load(ckpt_path)
            agent.epsilon = 0.0
            entries.append((spec["display"], agent, spec["force_only"], spec["traffic_aware"]))
    return entries


def _build_main_entries(ckpt_dir: str, cfg: HandoverConfig) -> list[tuple[str, object, bool, bool]]:
    trad = [
        ("Max-RVT", MaxRVTPolicy(), False, True),
        ("ISA-HO", ISAHOPolicy(reassoc_period_s=15), False, True),
        ("Swap-HO", SwapHOPolicy(), False, True),
        ("RA-HO", RAHOPolicy(), False, True),
        ("MCMF-HO", MCMFHOPolicy(), False, True),
        ("RH-STG", RHSTGPolicy(), False, True),
    ]
    return trad + _load_drl_entries(ckpt_dir, cfg, MAIN_DRL_KEYS)


def _reorder_results(res_dict: dict, canonical_order: list[str] | None = None) -> dict:
    if not canonical_order:
        return res_dict
    ordered = {}
    for k in canonical_order:
        if k in res_dict:
            ordered[k] = res_dict[k]
    for k, v in res_dict.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _eval_entries(
    entries: list[tuple[str, object, bool, bool]],
    cfg: HandoverConfig,
    seeds: list[int],
    duration_s: int,
    service_mode: str = "surrogate",
    micro_slots_override: int | None = None,
    title: str = "",
    existing: dict | None = None,
    overwrite: bool = False,
    canonical_order: list[str] | None = None,
) -> dict:
    out = dict(existing) if (existing is not None and not overwrite) else {}
    pending = [e for e in entries if e[0] not in out]
    if not pending:
        if title:
            print(f"[Up-to-date] {title}", flush=True)
        return _reorder_results(out, canonical_order)
    if title:
        print(f"\n=== {title} (evaluating {len(pending)}/{len(entries)} policies) ===", flush=True)
    for label, pol, force_only, traffic_aware in pending:
        t0 = time.perf_counter()
        res = run_policy_episodes(
            pol,
            cfg,
            seeds=seeds,
            duration_s=duration_s,
            service_mode=service_mode,
            force_only_trigger=force_only,
            traffic_aware_trigger=traffic_aware,
            micro_slots_override=micro_slots_override,
        )
        out[label] = res
        m = res["mean"]
        print(
            f"  {label:11s} | HO={m['total_handovers']:6.1f} | PP={m['ping_pong_rate']*100:5.2f}% | "
            f"Out={m['outage_rate']*100:5.2f}% | Drop={m['rt_drop_rate']*100:5.2f}% | "
            f"Del={m['rt_delivered_delay_ms']:4.1f}ms | Tput={m['effective_tput_mbps']:5.2f} | "
            f"Var={m['load_variance']:.3f} | Jain={m['jain_fairness']:.3f} | "
            f"Dec={m['decision_ms_per_step']:.2f}ms ({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )
    return _reorder_results(out, canonical_order)


def _load_existing(path: str, overwrite: bool) -> dict | None:
    if os.path.exists(path) and not overwrite:
        with open(path) as f:
            return json.load(f)
    return None


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {path}", flush=True)


def run_suite_main(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite1_main_600s.json")
    prev = _load_existing(path, overwrite)
    cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=600, num_users=200, input_rate_mbps=28.0)
    seeds = [1001, 1002, 1003, 1004, 1005]
    entries = _build_main_entries(ckpt_dir, cfg)
    res = _eval_entries(
        entries,
        cfg,
        seeds,
        600,
        title="Suite 1: Main 600s Benchmark (5 seeds, tight_45n, U=200, 28 Mbps)",
        existing=prev.get("results") if prev else None,
        overwrite=overwrite,
        canonical_order=ALL_MAIN_ORDER,
    )
    _save_json(path, {"suite": "main", "seeds": seeds, "duration": 600, "preset": "tight_45n", "num_users": 200, "rate": 28.0, "results": res})


def run_suite_ablation(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite2_ablation_600s.json")
    prev = _load_existing(path, overwrite)
    seeds = [1001, 1002, 1003, 1004, 1005]
    cfg_std = HandoverConfig(constellation_name="tight_45n", episode_duration_s=600, num_users=200, input_rate_mbps=28.0)
    entries_std = _load_drl_entries(ckpt_dir, cfg_std, ABLATION_KEYS)
    res_std = _eval_entries(
        entries_std,
        cfg_std,
        seeds,
        600,
        title="Suite 2a: Ablation Study - Standard (5 seeds, 600s)",
        existing=prev.get("standard") if prev else None,
        overwrite=overwrite,
    )

    cfg_str = HandoverConfig(
        constellation_name="tight_45n",
        episode_duration_s=600,
        num_users=220,
        input_rate_mbps=32.0,
        shadow_sigma_scale=1.5,
        enable_hotspot_drift=True,
    )
    entries_str = _load_drl_entries(ckpt_dir, cfg_str, ABLATION_KEYS)
    res_str = _eval_entries(
        entries_str,
        cfg_str,
        seeds,
        600,
        title="Suite 2b: Ablation Study - Hotspot Drift + 1.5x Shadowing (5 seeds, 600s)",
        existing=prev.get("stress") if prev else None,
        overwrite=overwrite,
    )
    _save_json(path, {"suite": "ablation", "seeds": seeds, "standard": res_std, "stress": res_str})


def run_suite_load_sweep(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite3_load_drift_600s.json")
    prev = _load_existing(path, overwrite)
    seeds = [1001, 1002, 1003]
    rates = [20.0, 28.0, 36.0, 44.0]
    by_rate = {}
    prev_rates = prev.get("by_rate", {}) if prev else {}
    for r in rates:
        r_key = str(int(r))
        cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=600, num_users=200, input_rate_mbps=r)
        entries = _build_main_entries(ckpt_dir, cfg)
        by_rate[r_key] = _eval_entries(
            entries,
            cfg,
            seeds,
            600,
            title=f"Suite 3: Load Sweep rate={r:.0f} Mbps (3 seeds, 600s)",
            existing=prev_rates.get(r_key),
            overwrite=overwrite,
            canonical_order=ALL_MAIN_ORDER,
        )

    cfg_drift = HandoverConfig(
        constellation_name="tight_45n",
        episode_duration_s=600,
        num_users=220,
        input_rate_mbps=32.0,
        enable_hotspot_drift=True,
    )
    entries_d = _build_main_entries(ckpt_dir, cfg_drift)
    res_drift = _eval_entries(
        entries_d,
        cfg_drift,
        seeds,
        600,
        title="Suite 3b: Hotspot Drift (U=220, 32 Mbps, 3 seeds, 600s)",
        existing=prev.get("hotspot_drift") if prev else None,
        overwrite=overwrite,
        canonical_order=ALL_MAIN_ORDER,
    )
    _save_json(path, {"suite": "load_sweep", "seeds": seeds, "rates": rates, "by_rate": by_rate, "hotspot_drift": res_drift})


def run_suite_robustness(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite4_robustness_600s.json")
    prev = _load_existing(path, overwrite)
    seeds = [1001, 1002, 1003]
    shadow_scales = [0.67, 1.0, 1.5, 2.0]
    by_shadow = {}
    prev_sh = prev.get("by_shadow", {}) if prev else {}
    for sc in shadow_scales:
        sc_key = f"{sc:.2f}"
        cfg = HandoverConfig(
            constellation_name="tight_45n",
            episode_duration_s=600,
            num_users=200,
            input_rate_mbps=28.0,
            shadow_sigma_scale=sc,
        )
        entries = _build_main_entries(ckpt_dir, cfg)
        by_shadow[sc_key] = _eval_entries(
            entries,
            cfg,
            seeds,
            600,
            title=f"Suite 4a: Shadowing scale={sc:.2f}x (3 seeds, 600s)",
            existing=prev_sh.get(sc_key),
            overwrite=overwrite,
            canonical_order=ALL_MAIN_ORDER,
        )

    tau_hos = [0.0, 15.0, 30.0, 50.0]
    by_tau = {}
    prev_tau = prev.get("by_tau", {}) if prev else {}
    for tau in tau_hos:
        t_key = str(int(tau))
        cfg = HandoverConfig(
            constellation_name="tight_45n",
            episode_duration_s=600,
            num_users=200,
            input_rate_mbps=28.0,
            tau_ho_ms=tau,
        )
        entries = _build_main_entries(ckpt_dir, cfg)
        by_tau[t_key] = _eval_entries(
            entries,
            cfg,
            seeds,
            600,
            title=f"Suite 4b: Interruption tau_HO={tau:.0f} ms (3 seeds, 600s)",
            existing=prev_tau.get(t_key),
            overwrite=overwrite,
            canonical_order=ALL_MAIN_ORDER,
        )

    _save_json(
        path,
        {
            "suite": "robustness",
            "seeds": seeds,
            "shadow_scales": shadow_scales,
            "by_shadow": by_shadow,
            "tau_hos": tau_hos,
            "by_tau": by_tau,
        },
    )


def run_suite_constellation(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite5_constellation_600s.json")
    prev = _load_existing(path, overwrite)
    seeds = [1001, 1002, 1003]
    presets = ["tight_45n", "standard_47n", "dense_35n"]
    by_preset = {}
    prev_pr = prev.get("by_preset", {}) if prev else {}
    for pr in presets:
        cfg = HandoverConfig(constellation_name=pr, episode_duration_s=600, num_users=200, input_rate_mbps=28.0)
        entries = _build_main_entries(ckpt_dir, cfg)
        by_preset[pr] = _eval_entries(
            entries,
            cfg,
            seeds,
            600,
            title=f"Suite 5: Constellation Generalization preset={pr} (3 seeds, 600s)",
            existing=prev_pr.get(pr),
            overwrite=overwrite,
            canonical_order=ALL_MAIN_ORDER,
        )
    _save_json(path, {"suite": "constellation", "seeds": seeds, "presets": presets, "by_preset": by_preset})


def run_suite_closed_loop(ckpt_dir: str, out_dir: str, overwrite: bool = False) -> None:
    path = os.path.join(out_dir, "suite6_mappo_closed_loop.json")
    prev = _load_existing(path, overwrite)
    seeds = [1001, 1002, 1003]
    duration_s = 30
    micro_slots = 200
    cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=duration_s, num_users=200, input_rate_mbps=28.0)
    entries = _build_main_entries(ckpt_dir, cfg)
    res = _eval_entries(
        entries,
        cfg,
        seeds,
        duration_s,
        service_mode="mappo_local",
        micro_slots_override=micro_slots,
        title=f"Suite 6: Frozen Chapter 3 MAPPO-APG-Local 2ms Closed-Loop (3 seeds, {duration_s}s, {micro_slots} slots/s)",
        existing=prev.get("results") if prev else None,
        overwrite=overwrite,
        canonical_order=ALL_MAIN_ORDER,
    )
    _save_json(
        path,
        {
            "suite": "closed_loop",
            "service_mode": "mappo_local",
            "seeds": seeds,
            "duration": duration_s,
            "micro_slots_per_s": micro_slots,
            "results": res,
        },
    )


SUITE_MAP = {
    "main": run_suite_main,
    "ablation": run_suite_ablation,
    "load_sweep": run_suite_load_sweep,
    "robustness": run_suite_robustness,
    "constellation": run_suite_constellation,
    "closed_loop": run_suite_closed_loop,
}


def main():
    parser = argparse.ArgumentParser(description="Run Chapter 4 paper evaluation suites")
    parser.add_argument("--suites", nargs="+", default=["all"], choices=["all"] + list(SUITE_MAP.keys()))
    parser.add_argument("--ckpt-dir", type=str, default="handover/checkpoints_paper")
    parser.add_argument("--out-dir", type=str, default="handover/results/paper")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    os.makedirs(args.out_dir, exist_ok=True)
    selected = list(SUITE_MAP.keys()) if "all" in args.suites else args.suites
    for s in selected:
        SUITE_MAP[s](args.ckpt_dir, args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
