# -*- coding: utf-8 -*-
"""Vectorized LEO multi-satellite direct-to-cell beam hopping simulator."""

import math
import numpy as np
from scipy.special import j1
import config as cfg
from coordination import NeighborInterference
from association import solve_isa, solve_nlb, solve_mea, solve_mmtg


def bessel_beam_gain(off_axis_rad: np.ndarray) -> np.ndarray:
    """3GPP TR 38.811 circular-aperture multibeam antenna gain G^T(theta)."""
    u = cfg.KA_APERTURE * np.sin(np.clip(np.abs(off_axis_rad), 0.0, math.pi / 2.0))
    gain = np.ones_like(u, dtype=np.float64)
    nz = u > 1e-8
    gain[nz] = (2.0 * j1(u[nz]) / u[nz]) ** 2
    return gain


class LEOBHEnv:
    """16-satellite, 144-cell LEO direct-to-cell beam hopping environment."""

    def __init__(self, assoc_method: str = "isa", alpha: float = cfg.ALPHA, seed: int = 0, cross_satellite: bool = False):
        self.assoc_method = assoc_method.lower()
        self.alpha = alpha
        self.cross_satellite = cross_satellite
        self.rng = np.random.default_rng(seed)

        # Build 16 satellite positions (4x4 grid) and 144 ground cells (16 local 3x3 multibeam clusters)
        self.cell_pos = np.zeros((cfg.M, 3), dtype=np.float64)
        self.sat_pos = np.zeros((cfg.N, 3), dtype=np.float64)
        sx = (np.arange(4) - 1.5) * cfg.SAT_SPACING_M
        sy = (np.arange(4) - 1.5) * cfg.SAT_SPACING_M
        idx_m = 0
        for ix in range(4):
            for iy in range(4):
                n = ix * 4 + iy
                self.sat_pos[n] = [sx[ix], sy[iy], cfg.H]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        self.cell_pos[idx_m] = [
                            sx[ix] + dx * cfg.CELL_SPACING_M,
                            sy[iy] + dy * cfg.CELL_SPACING_M,
                            0.0,
                        ]
                        idx_m += 1

        # Precompute geometry & visibility matrix V (N, M) with <= J=14 candidate cells per satellite
        diff = self.cell_pos[None, :, :] - self.sat_pos[:, None, :]  # (N, M, 3)
        horiz_dist = np.linalg.norm(diff[:, :, :2], axis=-1)  # (N, M)
        self.elev_rad = np.arctan2(cfg.H, np.maximum(horiz_dist, 1.0))  # (N, M)
        min_elev_rad = math.radians(cfg.THETA_MIN_DEG)

        # Visibility: each satellite sees up to J=14 closest cells above minimum elevation
        self.V = np.zeros((cfg.N, cfg.M), dtype=np.int32)
        self.cand_cells = np.full((cfg.N, cfg.J), -1, dtype=np.int32)
        for n in range(cfg.N):
            eligible = np.where(self.elev_rad[n] >= min_elev_rad)[0]
            order = eligible[np.argsort(horiz_dist[n, eligible])]
            chosen = order[: cfg.J]
            self.V[n, chosen] = 1
            self.cand_cells[n, : len(chosen)] = chosen

        # Guarantee every cell is visible to at least one satellite
        for m in range(cfg.M):
            if self.V[:, m].sum() == 0:
                nearest_n = int(np.argmin(horiz_dist[:, m]))
                self.V[nearest_n, m] = 1

        # Rebuild candidate cell index map chi_n(j) cleanly
        self.cand_cells = np.full((cfg.N, cfg.J), -1, dtype=np.int32)
        self.real_mask = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            cands = np.where(self.V[n] == 1)[0][: cfg.J]
            self.cand_cells[n, : len(cands)] = cands
            self.real_mask[n, : len(cands)] = 1.0

        # Coverage-overlap degree v_cov^m (M,) and local v_j^n (N, J)
        vis_count = self.V.sum(axis=0).astype(np.float32)
        self.v_cov = np.clip((vis_count - 1.0) / (cfg.N - 1.0), 0.0, 1.0)
        self.v_local = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            for j in range(cfg.J):
                m = self.cand_cells[n, j]
                if m >= 0:
                    self.v_local[n, j] = (self.V[:, m].sum() - self.V[n, m]) / (cfg.N - 1.0)

        # Common-visibility prior C_n (N, J, J)
        self.C_prior = np.zeros((cfg.N, cfg.J, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            other = np.delete(self.V, n, axis=0)  # (N-1, M)
            for i in range(cfg.J):
                mi = self.cand_cells[n, i]
                if mi < 0:
                    continue
                for j in range(cfg.J):
                    mj = self.cand_cells[n, j]
                    if mj < 0:
                        continue
                    self.C_prior[n, i, j] = np.mean(other[:, mi] * other[:, mj])

        # Precompute cell-to-cell beam gain table: for satellite n pointing at cell m_b, gain toward cell m_u
        # (Using user-level exact positions at reset for realistic intra/inter-beam interference)
        self.Z = None
        self.action_mask = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        self.Omega_prior = np.zeros((cfg.N, cfg.J, cfg.J), dtype=np.float32)
        self.I_mat = np.zeros((cfg.N, cfg.J, cfg.J), dtype=np.float32)

    def reset(
        self,
        input_rate_mbps: float | None = None,
        num_users: int | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if input_rate_mbps is None:
            self.input_rate_mbps = float(self.rng.choice(cfg.EVAL_RATES_MBPS))
        else:
            self.input_rate_mbps = float(input_rate_mbps)

        if num_users is None:
            self.U = int(self.rng.integers(cfg.U_RANGE[0], cfg.U_RANGE[1] + 1))
        else:
            self.U = int(num_users)

        # Realistic spatial user distribution across 144 cells with hotspots
        hotspot_weights = self.rng.gamma(shape=1.2, scale=1.0, size=cfg.M)
        hotspot_weights /= hotspot_weights.sum()
        self.user_cell = self.rng.choice(cfg.M, size=self.U, p=hotspot_weights).astype(np.int32)

        # User coordinates randomly inside the cell's 3 dB beam footprint radius
        ang = self.rng.uniform(0.0, 2.0 * math.pi, size=self.U)
        rad = cfg.CELL_RADIUS_M * np.sqrt(self.rng.uniform(0.0, 1.0, size=self.U))
        offsets = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1)
        self.user_pos = np.zeros((self.U, 3), dtype=np.float64)
        self.user_pos[:, :2] = self.cell_pos[self.user_cell, :2] + offsets

        # User centroids per cell for distance prior omega_{ij}^n
        cell_centroids = self.cell_pos[:, :2].copy()
        self.cell_users = [[] for _ in range(cfg.M)]
        for u in range(self.U):
            self.cell_users[self.user_cell[u]].append(u)
        self.cell_users_arr = [np.array(lst, dtype=np.int32) for lst in self.cell_users]
        for m in range(cfg.M):
            if len(self.cell_users_arr[m]) > 0:
                cell_centroids[m] = self.user_pos[self.cell_users_arr[m], :2].mean(axis=0)

        # Distance prior Omega_n (N, J, J)
        self.Omega_prior = np.zeros_like(self.Omega_prior)
        for n in range(cfg.N):
            for i in range(cfg.J):
                mi = self.cand_cells[n, i]
                if mi < 0:
                    continue
                for j in range(cfg.J):
                    mj = self.cand_cells[n, j]
                    if mj < 0:
                        continue
                    d_ij = np.linalg.norm(cell_centroids[mi] - cell_centroids[mj])
                    self.Omega_prior[n, i, j] = 1.0 / (1.0 + math.exp((d_ij - cfg.D_TH_M) / cfg.ETA_M))

        # Cell aggregate statistical traffic demand T_m (Mbps)
        cell_user_counts = np.array([len(lst) for lst in self.cell_users], dtype=np.float64)
        self.cell_demands = cell_user_counts * self.input_rate_mbps

        # Solve long-timescale cell-satellite association
        if self.assoc_method == "isa":
            self.Z = solve_isa(self.cell_demands, self.V, cfg.J, rng=self.rng)
        elif self.assoc_method == "nlb":
            self.Z = solve_nlb(self.V, cfg.J, rng=self.rng)
        elif self.assoc_method == "mea":
            self.Z = solve_mea(self.elev_rad, self.V, cfg.J)
        elif self.assoc_method == "mmtg":
            self.Z = solve_mmtg(self.cell_demands, self.V, self.elev_rad, cfg.J)
        else:
            raise ValueError(f"Unknown association method: {self.assoc_method}")

        self.cell_sat = np.argmax(self.Z, axis=0).astype(np.int32)  # (M,)
        self.action_mask = np.zeros_like(self.action_mask)
        for n in range(cfg.N):
            for j in range(cfg.J):
                m = self.cand_cells[n, j]
                if m >= 0 and self.Z[n, m] == 1:
                    self.action_mask[n, j] = 1.0

        # Precompute channel gain |h^{n, j, u}|^2 for every satellite n, candidate beam j, user u
        # Shape: (N, J, U) — only 16 * 14 * 500 = 112k floats (~0.9 MB)!
        vec_su = self.user_pos[None, :, :] - self.sat_pos[:, None, :]  # (N, U, 3)
        dist_su = np.linalg.norm(vec_su, axis=-1)  # (N, U)
        unit_su = vec_su / np.maximum(dist_su[:, :, None], 1.0)  # (N, U, 3)

        # Log-normal shadow fading L_sh (N, U)
        sh_db = self.rng.normal(0.0, cfg.SHADOW_STD_DB, size=(cfg.N, self.U))
        L_sh = 10.0 ** (sh_db / 10.0)
        fspl = (4.0 * math.pi * dist_su / cfg.LAMBDA) ** 2  # (N, U)
        base_path_gain = (cfg.G_MAX_T_LIN * cfg.G_R_LIN) / (fspl * L_sh)  # (N, U)

        self.H_sq = np.zeros((cfg.N, cfg.J, self.U), dtype=np.float64)
        for n in range(cfg.N):
            for j in range(cfg.J):
                m = self.cand_cells[n, j]
                if m < 0:
                    continue
                vec_sb = self.cell_pos[m] - self.sat_pos[n]  # (3,)
                unit_sb = vec_sb / np.linalg.norm(vec_sb)
                cos_ang = np.clip(unit_su[n] @ unit_sb, -1.0, 1.0)
                off_axis = np.arccos(cos_ang)  # (U,)
                self.H_sq[n, j, :] = base_path_gain[n] * bessel_beam_gain(off_axis)

        self.H_sq_base = self.H_sq.copy()
        self.cell_fade_db = np.zeros(cfg.M, dtype=np.float64)
        if cfg.TIME_VARYING_FADING and cfg.FADING_STD_DB > 0.0:
            self.cell_fade_db = self.rng.normal(0.0, cfg.FADING_STD_DB, size=cfg.M)
            fade_lin = 10.0 ** (self.cell_fade_db[self.user_cell] / 10.0)
            self.H_sq = self.H_sq_base * fade_lin[None, None, :]

        # Per-satellite associated user lists for local credit assignment
        self.sat_users_arr = []
        for n in range(cfg.N):
            lst = []
            for m in range(cfg.M):
                if self.cell_sat[m] == n and len(self.cell_users_arr[m]) > 0:
                    lst.extend(self.cell_users_arr[m].tolist())
            self.sat_users_arr.append(np.array(lst, dtype=np.int32))

        # Exact normalized intra-satellite beam interference coupling matrix I_mat (N, J, J)
        self.I_mat = np.zeros_like(self.I_mat)
        for n in range(cfg.N):
            for j in range(cfg.J):
                mj = self.cand_cells[n, j]
                if mj < 0:
                    continue
                us = self.cell_users_arr[mj]
                if len(us) == 0:
                    continue
                sig_j = max(float(self.H_sq[n, j, us].mean()), 1e-30)
                for i in range(cfg.J):
                    if i == j or self.cand_cells[n, i] < 0:
                        continue
                    self.I_mat[n, i, j] = float(np.clip(self.H_sq[n, i, us].mean() / sig_j, 0.0, 2.0))

        # Queue state:
        # rt_buckets[u, age-1] for age in 1..T_TH_R (float packets)
        self.rt_buckets = np.zeros((self.U, cfg.T_TH_R), dtype=np.float64)
        self.bulk_queue = np.zeros(self.U, dtype=np.float64)
        # Historical 50-slot throughput EMA/ring buffer (bps)
        self.hist_tput = np.full(self.U, 1e5, dtype=np.float64)
        self.hist_buf = np.zeros((cfg.PF_WINDOW, self.U), dtype=np.float64)
        self.slot_idx = 0

        # Two-state Markov-Modulated Poisson Process (MMPP) for bursty RT traffic
        # Preserves long-term mean traffic split: 20% RT, 80% BULK
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

        # Seed initial slot arrivals
        mult0 = np.where(self.cell_on[self.user_cell], self.burst_mult, self.low_mult)
        self.rt_buckets[:, 0] = self.rng.poisson(self.base_lam_rt * mult0).astype(np.float64)
        self.bulk_queue[:] = self.rng.poisson(self.base_lam_bulk, size=self.U).astype(np.float64)

        if self.cross_satellite:
            self.neighbors = NeighborInterference(self)
        return self._get_obs()

    def _get_obs(self):
        """Return global critic state s_t (5M,), local node features F_t (N, J, 5), masks, priors."""
        ages = np.arange(1, cfg.T_TH_R + 1, dtype=np.float64)
        rt_q = self.rt_buckets.sum(axis=1)  # (U,)
        tot_q = rt_q + self.bulk_queue  # (U,)

        # Mean RT age per user
        rt_age_sum = self.rt_buckets @ ages
        rt_avg_delay = np.where(rt_q > 0, rt_age_sum / np.maximum(rt_q, 1e-9), 0.0)

        # Head-of-line (oldest non-empty RT bucket) waiting time per user
        has_pkt = self.rt_buckets > 0
        hol = np.where(has_pkt.any(axis=1), cfg.T_TH_R - np.argmax(has_pkt[:, ::-1], axis=1), 0.0)

        # Aggregate per-cell normalized features (M, 5)
        cell_feat = np.zeros((cfg.M, 5), dtype=np.float32)
        for m in range(cfg.M):
            us = self.cell_users_arr[m]
            if len(us) > 0:
                cell_feat[m, 0] = min(float(tot_q[us].sum()) / 2000.0, 5.0)       # normalized queue length
                cell_feat[m, 1] = float(rt_avg_delay[us].mean()) / cfg.T_TH_R    # normalized avg RT delay
                cell_feat[m, 2] = float(hol[us].max()) / cfg.T_TH_R              # normalized HOL waiting time
                cell_feat[m, 3] = min(float(self.hist_tput[us].mean()) / 50e6, 2.0)  # normalized hist throughput
            cell_feat[m, 4] = self.v_cov[m]

        s_t = cell_feat.ravel().copy()  # (5*M,)
        F_t = np.zeros((cfg.N, cfg.J, 5), dtype=np.float32)
        F_rich = np.zeros((cfg.N, cfg.J, cfg.RICH_NODE_DIM), dtype=np.float32)
        eff_t_noise = cfg.T_NOISE * (10.0 ** (cfg.UE_LOSS_NF_DB / 10.0))
        noise_full_bw = cfg.K_BOLTZMANN * eff_t_noise * cfg.B_TOT
        p_nom = cfg.P_TOT_W / cfg.K_MAX
        for n in range(cfg.N):
            for j in range(cfg.J):
                m = self.cand_cells[n, j]
                if m >= 0:
                    F_t[n, j] = cell_feat[m]
                    F_t[n, j, 4] = self.v_local[n, j]
                    us = self.cell_users_arr[m]
                    if len(us) > 0:
                        F_rich[n, j, 0] = min(float(rt_q[us].sum()) / 800.0, 5.0)
                        F_rich[n, j, 1] = min(float(self.bulk_queue[us].sum()) / 2000.0, 5.0)
                        F_rich[n, j, 2] = cell_feat[m, 1]
                        F_rich[n, j, 3] = cell_feat[m, 2]
                        snr_lin = (p_nom * self.H_sq[n, j, us]) / noise_full_bw
                        snr_db = float(10.0 * np.log10(max(float(snr_lin.mean()), 1e-6)))
                        F_rich[n, j, 4] = float(np.clip((snr_db - 20.0) / 15.0, -2.5, 2.5))
                        F_rich[n, j, 5] = cell_feat[m, 3]
                        F_rich[n, j, 7] = min(float(len(us)) / 5.0, 2.0)
                    F_rich[n, j, 6] = self.v_local[n, j]

        if self.cross_satellite:
            F_rich = self.neighbors.features(self, F_rich)

        obs = {
            "s": s_t,
            "F": F_t,
            "F_rich": F_rich,
            "I_mat": self.I_mat,
            "Omega": self.Omega_prior,
            "C": self.C_prior,
            "real_mask": self.real_mask,
            "action_mask": self.action_mask,
        }
        if self.cross_satellite:
            obs["E_mat"] = self.neighbors.E_mat
        return obs

    def step(self, beam_sel: np.ndarray, power_ratios: np.ndarray):
        """Execute one time slot.

        Args:
            beam_sel: binary (N, J) array selecting K <= K_MAX associated candidate cells per satellite.
            power_ratios: float (N, J) array summing to 1 over selected cells (0 otherwise).
        """
        # Start-of-slot RT queueing delay Gamma_t
        ages = np.arange(1, cfg.T_TH_R + 1, dtype=np.float64)
        rt_q_start = self.rt_buckets.sum(axis=1)
        tau_u_start = np.where(rt_q_start > 0, (self.rt_buckets @ ages) / np.maximum(rt_q_start, 1e-9), 0.0)
        gamma_t = float(tau_u_start.mean())

        # Enforce masks and power budget
        b = (beam_sel > 0.5).astype(np.float64) * self.action_mask
        p = np.maximum(power_ratios, 0.0) * b
        p_sum = p.sum(axis=1, keepdims=True)
        p = np.where(p_sum > 1e-12, p / np.maximum(p_sum, 1e-12), 0.0)
        P_beam = p * cfg.P_TOT_W  # (N, J) watts

        # Total received power from ALL active beams across all satellites for every user u:
        # P_beam (N, J) contracted with H_sq (N, J, U) -> total_rx (U,)
        total_rx = np.einsum("nj,nju->u", P_beam, self.H_sq)  # (U,)
        eff_t_noise = cfg.T_NOISE * (10.0 ** (cfg.UE_LOSS_NF_DB / 10.0))
        noise_full_bw = cfg.K_BOLTZMANN * eff_t_noise * cfg.B_TOT

        # Compute achieved rate R_t^u (bps) for users in illuminated cells via PF RB scheduling
        user_rate = np.zeros(self.U, dtype=np.float64)
        rt_q = self.rt_buckets.sum(axis=1)
        has_backlog = (rt_q + self.bulk_queue) > 1e-9

        active_n, active_j = np.where(b > 0.5)
        for n, j in zip(active_n, active_j):
            m = self.cand_cells[n, j]
            us = self.cell_users_arr[m]
            if len(us) == 0:
                continue
            active_us = us[has_backlog[us]]
            if len(active_us) == 0:
                continue

            sig = P_beam[n, j] * self.H_sq[n, j, active_us]  # (A,)
            interf = np.maximum(total_rx[active_us] - sig, 0.0)
            # Notice B_u(t) cancels out in SINR = (P*B_u/B_tot * |h|^2) / (k*T*B_u + I*B_u/B_tot)!
            sinr = sig / (noise_full_bw + interf)
            se = np.log2(1.0 + sinr)  # bits/s/Hz

            if len(active_us) == 1:
                user_rate[active_us[0]] = cfg.B_TOT * se[0]
            else:
                # PF scheduling over N_RB = 80 RBs
                hist = np.maximum(self.hist_tput[active_us], 1e4)
                pf_metric = se / hist
                # Allocate RBs proportionally to PF priority (fast accurate approximation to sequential PF)
                # Assign RBs iteratively in batches of 4 for speed while preserving PF fairness
                rb_counts = np.zeros(len(active_us), dtype=np.float64)
                est_hist = hist.copy()
                rb_bw = cfg.B_TOT / cfg.N_RB
                step_rbs = 4
                for _ in range(cfg.N_RB // step_rbs):
                    winner = int(np.argmax(se / est_hist))
                    rb_counts[winner] += step_rbs
                    est_hist[winner] += (rb_bw * step_rbs * se[winner]) / cfg.PF_WINDOW
                user_rate[active_us] = (rb_counts * rb_bw) * se

        # Serve queues: RT first (FIFO = oldest buckets first!), then BULK
        cap_pkts = (user_rate * cfg.T_SLOT) / cfg.L_P  # (U,)
        rem_cap = cap_pkts.copy()

        # Serve RT packets from oldest (index T_TH_R-1) down to newest (index 0)
        served_rt = np.zeros(self.U, dtype=np.float64)
        served_rt_age = 0.0
        for age_idx in range(cfg.T_TH_R - 1, -1, -1):
            avail = self.rt_buckets[:, age_idx]
            take = np.minimum(avail, rem_cap)
            self.rt_buckets[:, age_idx] -= take
            rem_cap -= take
            served_rt += take
            served_rt_age += float(take.sum()) * (age_idx + 1)

        # Serve BULK packets with remaining capacity
        served_bulk = np.minimum(self.bulk_queue, rem_cap)
        self.bulk_queue -= served_bulk
        served_total = served_rt + served_bulk  # (U,)

        # Delivered data volume zeta_t (bits)
        zeta_t = float(served_total.sum() * cfg.L_P)

        # Update 50-slot historical throughput window
        delivered_bps = (served_total * cfg.L_P) / cfg.T_SLOT
        self.hist_buf[self.slot_idx % cfg.PF_WINDOW] = delivered_bps
        self.slot_idx += 1
        win = min(self.slot_idx, cfg.PF_WINDOW)
        self.hist_tput = self.hist_buf[:win].mean(axis=0)

        # Discard expired RT packets (those still at age T_TH_R = index T_TH_R-1)
        expired_rt = float(self.rt_buckets[:, cfg.T_TH_R - 1].sum())

        # Age surviving RT packets by 1 slot and insert new MMPP bursty RT & BULK arrivals
        self.rt_buckets[:, 1:] = self.rt_buckets[:, :-1]
        mult = np.where(self.cell_on[self.user_cell], self.burst_mult, self.low_mult)
        new_rt = self.rng.poisson(self.base_lam_rt * mult).astype(np.float64)
        new_bulk = self.rng.poisson(self.base_lam_bulk * np.where(self.cell_on[self.user_cell], 0.6, 1.08)).astype(np.float64)
        self.rt_buckets[:, 0] = new_rt
        self.bulk_queue += new_bulk
        # Advance cell-level Markov burst states
        u_flip = self.rng.random(cfg.M)
        self.cell_on = np.where(self.cell_on, u_flip > self.p_10, u_flip < self.p_01)

        # Shared team reward R_t and per-satellite local reward vector for credit assignment
        reward = self.alpha * (zeta_t / cfg.ZETA_NORM) - (1.0 - self.alpha) * (gamma_t / cfg.GAMMA_NORM)
        delivered_bits_u = served_total * cfg.L_P
        r_local = np.zeros(cfg.N, dtype=np.float32)
        sat_zeta_norm = cfg.ZETA_NORM / cfg.N
        for n in range(cfg.N):
            us = self.sat_users_arr[n]
            if len(us) == 0:
                continue
            sat_bits = float(delivered_bits_u[us].sum())
            sat_delay = float(tau_u_start[us].mean())
            r_local[n] = self.alpha * (sat_bits / sat_zeta_norm) - (1.0 - self.alpha) * (sat_delay / cfg.GAMMA_NORM)

        # Advance AR(1) time-varying fading if enabled
        if cfg.TIME_VARYING_FADING and cfg.FADING_STD_DB > 0.0:
            innov_std = cfg.FADING_STD_DB * math.sqrt(max(1.0 - cfg.FADING_CORR ** 2, 1e-6))
            self.cell_fade_db = cfg.FADING_CORR * self.cell_fade_db + self.rng.normal(0.0, innov_std, size=cfg.M)
            fade_lin = 10.0 ** (self.cell_fade_db[self.user_cell] / 10.0)
            self.H_sq = self.H_sq_base * fade_lin[None, None, :]

        obs = self._get_obs()
        info = {
            "zeta_t": zeta_t,
            "gamma_t": gamma_t,
            "expired_rt": expired_rt,
            "served_rt": float(served_rt.sum()),
            "served_rt_age": served_rt_age,
            "arrived_rt": float(new_rt.sum()),
            "per_user_mbps": (zeta_t / cfg.T_SLOT) / (self.U * 1e6),
            "r_local": r_local,
        }
        return obs, float(reward), info
