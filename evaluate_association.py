# -*- coding: utf-8 -*-
"""Simulate and compare cell-satellite association schemes (NLB, MEA, MMTG, ISA)
under the same trained MAPPO-BH scheduling policy across 10-40 Mbps.
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from association import solve_isa, solve_nlb, solve_mea, solve_mmtg
from env import LEOBHEnv, bessel_beam_gain
from evaluate import MAPPOPolicyWrapper

ASSOC_RATES_MBPS = [10, 15, 20, 25, 30, 35, 40]


def make_clustered_hotspots(cell_pos: np.ndarray, rng: np.random.Generator, n_hotspots: int = 3, sigma_m: float = 95e3):
    """Spatially clustered regional hotspot distribution across the 144 cells."""
    lo, hi = cell_pos[:, 0].min() * 0.65, cell_pos[:, 0].max() * 0.65
    centers = rng.uniform(lo, hi, size=(n_hotspots, 2))
    w = np.full(len(cell_pos), 0.06, dtype=np.float64)
    for c in centers:
        d2 = np.sum((cell_pos[:, :2] - c[None, :]) ** 2, axis=-1)
        w += np.exp(-d2 / (2.0 * sigma_m ** 2))
    return w / w.sum()


class AssociationEvalEnv(LEOBHEnv):
    """LEO environment with realistic overlapping satellite visibility cone (<= J associated cells)."""

    def __init__(self, assoc_method: str = "isa", alpha: float = cfg.ALPHA, seed: int = 0):
        super().__init__(assoc_method=assoc_method, alpha=alpha, seed=seed)
        # True overlapping visibility: each satellite can see up to 28 cells within its elevation cone,
        # while association capacity constraint C3 strictly enforces <= J=14 associated cells per satellite.
        diff = self.cell_pos[None, :, :] - self.sat_pos[:, None, :]
        horiz_dist = np.linalg.norm(diff[:, :, :2], axis=-1)
        self.elev_rad = np.arctan2(cfg.H, np.maximum(horiz_dist, 1.0))
        min_elev_rad = math.radians(cfg.THETA_MIN_DEG)

        self.V_full = np.zeros((cfg.N, cfg.M), dtype=np.int32)
        for n in range(cfg.N):
            eligible = np.where(self.elev_rad[n] >= min_elev_rad)[0]
            order = eligible[np.argsort(horiz_dist[n, eligible])]
            self.V_full[n, order[:28]] = 1
        for m in range(cfg.M):
            if self.V_full[:, m].sum() == 0:
                self.V_full[int(np.argmin(horiz_dist[:, m])), m] = 1
        self.horiz_dist = horiz_dist

    def reset(self, input_rate_mbps: float = 40.0, num_users: int = 240, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.input_rate_mbps = float(input_rate_mbps)
        self.U = int(num_users)

        # Identical user & hotspot realization across all association methods for the same seed
        w = make_clustered_hotspots(self.cell_pos, self.rng, n_hotspots=3, sigma_m=95e3)
        self.user_cell = self.rng.choice(cfg.M, size=self.U, p=w).astype(np.int32)
        ang = self.rng.uniform(0.0, 2.0 * math.pi, size=self.U)
        rad = cfg.CELL_RADIUS_M * np.sqrt(self.rng.uniform(0.0, 1.0, size=self.U))
        self.user_pos = np.zeros((self.U, 3), dtype=np.float64)
        self.user_pos[:, :2] = self.cell_pos[self.user_cell, :2] + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1)

        # Pre-draw shadowing and traffic RNG streams cleanly so association RNG doesn't alter physics!
        sh_db = self.rng.normal(0.0, cfg.SHADOW_STD_DB, size=(cfg.N, self.U))
        assoc_rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        traffic_seed = int(self.rng.integers(0, 2**31 - 1))

        cell_centroids = self.cell_pos[:, :2].copy()
        self.cell_users = [[] for _ in range(cfg.M)]
        for u in range(self.U):
            self.cell_users[self.user_cell[u]].append(u)
        self.cell_users_arr = [np.array(lst, dtype=np.int32) for lst in self.cell_users]
        for m in range(cfg.M):
            if len(self.cell_users_arr[m]) > 0:
                cell_centroids[m] = self.user_pos[self.cell_users_arr[m], :2].mean(axis=0)

        self.cell_demands = np.array([len(lst) * self.input_rate_mbps for lst in self.cell_users], dtype=np.float64)

        # Solve cell-satellite association under identical V_full and C3 <= J=14
        if self.assoc_method == "isa":
            self.Z = solve_isa(self.cell_demands, self.V_full, cfg.J, rng=assoc_rng)
        elif self.assoc_method == "nlb":
            self.Z = solve_nlb(self.V_full, cfg.J, rng=assoc_rng)
        elif self.assoc_method == "mea":
            self.Z = solve_mea(self.elev_rad, self.V_full, cfg.J)
        elif self.assoc_method == "mmtg":
            self.Z = solve_mmtg(self.cell_demands, self.V_full, self.elev_rad, cfg.J)
        else:
            raise ValueError(f"Unknown assoc_method: {self.assoc_method}")

        self.cell_sat = np.argmax(self.Z, axis=0).astype(np.int32)

        # Build each satellite's J=14 local candidate graph: all associated cells first, padded with nearest visible cells
        self.cand_cells = np.full((cfg.N, cfg.J), -1, dtype=np.int32)
        self.real_mask = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        self.action_mask = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            assoc_cells = np.where(self.Z[n] == 1)[0]
            other_vis = np.where((self.V_full[n] == 1) & (self.Z[n] == 0))[0]
            other_vis = other_vis[np.argsort(self.horiz_dist[n, other_vis])]
            cands = np.concatenate([assoc_cells, other_vis])[: cfg.J]
            self.cand_cells[n, : len(cands)] = cands
            self.real_mask[n, : len(cands)] = 1.0
            self.action_mask[n, : len(assoc_cells)] = 1.0

        # Update coverage-overlap and common-visibility priors from V_full
        vis_count = self.V_full.sum(axis=0).astype(np.float32)
        self.v_cov = np.clip((vis_count - 1.0) / (cfg.N - 1.0), 0.0, 1.0)
        self.v_local = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        self.C_prior = np.zeros((cfg.N, cfg.J, cfg.J), dtype=np.float32)
        self.Omega_prior.fill(0.0)
        for n in range(cfg.N):
            other = np.delete(self.V_full, n, axis=0)
            for i in range(cfg.J):
                mi = self.cand_cells[n, i]
                if mi < 0:
                    continue
                self.v_local[n, i] = (self.V_full[:, mi].sum() - self.V_full[n, mi]) / (cfg.N - 1.0)
                for j in range(cfg.J):
                    mj = self.cand_cells[n, j]
                    if mj < 0:
                        continue
                    d_ij = np.linalg.norm(cell_centroids[mi] - cell_centroids[mj])
                    self.Omega_prior[n, i, j] = 1.0 / (1.0 + math.exp((d_ij - cfg.D_TH_M) / cfg.ETA_M))
                    self.C_prior[n, i, j] = np.mean(other[:, mi] * other[:, mj])

        # Exact 3GPP TR 38.811 channel gains
        vec_su = self.user_pos[None] - self.sat_pos[:, None]
        dist_su = np.linalg.norm(vec_su, axis=-1)
        unit_su = vec_su / np.maximum(dist_su[:, :, None], 1.0)
        L_sh = 10.0 ** (sh_db / 10.0)
        fspl = (4.0 * math.pi * dist_su / cfg.LAMBDA) ** 2
        base_g = (cfg.G_MAX_T_LIN * cfg.G_R_LIN) / (fspl * L_sh)
        self.H_sq = np.zeros((cfg.N, cfg.J, self.U), dtype=np.float64)
        for n in range(cfg.N):
            for j in range(cfg.J):
                m = self.cand_cells[n, j]
                if m < 0:
                    continue
                usb = self.cell_pos[m] - self.sat_pos[n]
                usb /= np.linalg.norm(usb)
                oa = np.arccos(np.clip(unit_su[n] @ usb, -1.0, 1.0))
                self.H_sq[n, j] = base_g[n] * bessel_beam_gain(oa)

        # Reset queues with identical traffic RNG stream across all 4 association schemes!
        self.rng = np.random.default_rng(traffic_seed)
        self.rt_buckets = np.zeros((self.U, cfg.T_TH_R), dtype=np.float64)
        self.bulk_queue = np.zeros(self.U, dtype=np.float64)
        self.hist_tput = np.full(self.U, 1e5, dtype=np.float64)
        self.hist_buf = np.zeros((cfg.PF_WINDOW, self.U), dtype=np.float64)
        self.slot_idx = 0

        self.p_10 = 1.0 - cfg.MMPP_STAY_ON
        self.p_01 = self.p_10 * cfg.MMPP_P_ON / (1.0 - cfg.MMPP_P_ON)
        low_m = max((1.0 - cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT) / (1.0 - cfg.MMPP_P_ON), 0.12)
        mean_m = cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT + (1.0 - cfg.MMPP_P_ON) * low_m
        self.burst_mult = cfg.MMPP_BURST_MULT / mean_m
        self.low_mult = low_m / mean_m
        self.cell_on = (self.rng.random(cfg.M) < cfg.MMPP_P_ON)

        tot_lam = (self.input_rate_mbps * 1e6 * cfg.T_SLOT) / cfg.L_P
        self.base_lam_rt = tot_lam * cfg.R_RT_FRAC
        self.base_lam_bulk = tot_lam * cfg.R_BULK_FRAC
        mult0 = np.where(self.cell_on[self.user_cell], self.burst_mult, self.low_mult)
        self.rt_buckets[:, 0] = self.rng.poisson(self.base_lam_rt * mult0).astype(np.float64)
        self.bulk_queue[:] = self.rng.poisson(self.base_lam_bulk, size=self.U).astype(np.float64)

        return self._get_obs()


def run_isa_sensitivity(env: AssociationEvalEnv, num_cases: int = 40):
    """Evaluate ISA parameter sensitivity (rho and L_max) on clustered hotspot demands."""
    demands_list = []
    for tc in range(num_cases):
        rng = np.random.default_rng(7000 + tc)
        w = make_clustered_hotspots(env.cell_pos, rng, n_hotspots=3, sigma_m=95e3)
        uc = rng.choice(cfg.M, size=240, p=w)
        dem = np.bincount(uc, minlength=cfg.M).astype(np.float64) * 40.0
        demands_list.append(dem)

    rho_res = {}
    for rho in [0.90, 0.93, 0.95, 0.97, 0.98]:
        vlist, times = [], []
        for idx, dem in enumerate(demands_list):
            t0 = time.perf_counter()
            Z = solve_isa(dem, env.V_full, cfg.J, rho=rho, L_max=100, rng=np.random.default_rng(100 + idx))
            times.append((time.perf_counter() - t0) * 1e3)
            vlist.append(float(np.var(Z @ dem)))
        rho_res[str(rho)] = {"var_mean": float(np.mean(vlist)), "time_ms": float(np.mean(times))}

    lmax_res = {}
    for lmax in [50, 75, 100, 150, 200]:
        vlist, times = [], []
        for idx, dem in enumerate(demands_list):
            t0 = time.perf_counter()
            Z = solve_isa(dem, env.V_full, cfg.J, rho=0.95, L_max=lmax, rng=np.random.default_rng(100 + idx))
            times.append((time.perf_counter() - t0) * 1e3)
            vlist.append(float(np.var(Z @ dem)))
        lmax_res[str(lmax)] = {"var_mean": float(np.mean(vlist)), "time_ms": float(np.mean(times))}
    return {"rho_sweep": rho_res, "lmax_sweep": lmax_res}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/mappo_bh.pt")
    parser.add_argument("--test-cases", type=int, default=25)
    parser.add_argument("--slots", type=int, default=120)
    parser.add_argument("--num-users", type=int, default=240)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out_dir, exist_ok=True)
    pol = MAPPOPolicyWrapper(args.ckpt)

    methods = [("NLB", "nlb"), ("MEA", "mea"), ("MMTG", "mmtg"), ("ISA (Proposed)", "isa")]
    results = {
        label: {"rates": ASSOC_RATES_MBPS, "tput_mean": [], "tput_ci": [], "delay_mean": [], "delay_ci": [], "load_var_40": 0.0}
        for label, _ in methods
    }

    for rate in ASSOC_RATES_MBPS:
        print(f"\n=== Association Eval @ {rate} Mbps ({args.test_cases} cases x {args.slots} slots, U={args.num_users}) ===")
        for label, key in methods:
            env = AssociationEvalEnv(assoc_method=key, alpha=cfg.ALPHA, seed=0)
            tputs, delays, vlist = [], [], []
            for tc in range(args.test_cases):
                obs = env.reset(input_rate_mbps=rate, num_users=args.num_users, seed=4000 + tc)
                vlist.append(float(np.var(env.Z @ env.cell_demands)))
                pol.reset()
                tot_bits, d_sum = 0.0, 0.0
                warmup = 20
                for step_i in range(args.slots + warmup):
                    b, p = pol.act(obs)
                    obs, _, info = env.step(b, p)
                    if step_i >= warmup:
                        tot_bits += info["zeta_t"]
                        d_sum += info["gamma_t"]
                tputs.append((tot_bits / (args.slots * cfg.T_SLOT)) / (env.U * 1e6))
                delays.append(d_sum / args.slots)
            tm, tc_ci = float(np.mean(tputs)), float(1.96 * np.std(tputs, ddof=1) / np.sqrt(len(tputs)))
            dm, dc_ci = float(np.mean(delays)), float(1.96 * np.std(delays, ddof=1) / np.sqrt(len(delays)))
            results[label]["tput_mean"].append(tm)
            results[label]["tput_ci"].append(tc_ci)
            results[label]["delay_mean"].append(dm)
            results[label]["delay_ci"].append(dc_ci)
            if rate == 40:
                results[label]["load_var_40"] = float(np.mean(vlist))
            print(f"  {label:15s} | LoadVar: {np.mean(vlist):9.1f} | Tput: {tm:6.2f} +/- {tc_ci:4.2f} Mbps | RT Delay: {dm:5.2f} +/- {dc_ci:4.2f} slots")

    print("\n=== Running ISA parameter sensitivity sweep ===")
    sens = run_isa_sensitivity(AssociationEvalEnv("isa"), num_cases=40)
    for k, v in sens["rho_sweep"].items():
        print(f"  rho={k} (L_max=100): mean_var={v['var_mean']:9.1f}, runtime={v['time_ms']:.2f} ms")
    for k, v in sens["lmax_sweep"].items():
        print(f"  L_max={k} (rho=0.95): mean_var={v['var_mean']:9.1f}, runtime={v['time_ms']:.2f} ms")

    out_json = os.path.join(args.out_dir, "association_results.json")
    with open(out_json, "w") as f:
        json.dump({"results": results, "isa_sensitivity": sens}, f, indent=2)

    # Plot 3-panel association comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=180)
    styles = {
        "NLB": ("--", "x", "#1f77b4"),
        "MEA": (":", "v", "#ff7f0e"),
        "MMTG": ("-.", "*", "#2ca02c"),
        "ISA (Proposed)": ("-", "^", "#d62728"),
    }
    for label, res in results.items():
        ls, mk, col = styles[label]
        axes[0].errorbar(res["rates"], res["tput_mean"], yerr=res["tput_ci"], linestyle=ls, marker=mk, color=col, capsize=3, label=label)
        axes[1].errorbar(res["rates"], res["delay_mean"], yerr=res["delay_ci"], linestyle=ls, marker=mk, color=col, capsize=3, label=label)
    axes[0].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[0].set_ylabel("Throughput per user (Mbps)")
    axes[0].set_title("(a) Per-User Throughput")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[1].set_ylabel("Average RT queueing delay (time slots)")
    axes[1].set_title("(b) Average RT Queueing Delay")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)

    labels = [m[0] for m in methods]
    lvars = [results[l]["load_var_40"] for l in labels]
    cols = [styles[l][2] for l in labels]
    bars = axes[2].bar(labels, lvars, color=cols, alpha=0.85, width=0.55)
    axes[2].set_ylabel("Satellite Load Variance (Mbps$^2$)")
    axes[2].set_title("(c) Load Variance @ 40 Mbps")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[2].tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, lvars):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "association_comparison.png")
    plt.savefig(out_png)
    print(f"\nSaved {out_json} and {out_png}")


if __name__ == "__main__":
    main()
