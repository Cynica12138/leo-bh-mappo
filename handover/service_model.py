# -*- coding: utf-8 -*-
"""Two-mode beam-hopping service execution:
  - Mode A (SurrogateServiceModel): Fast vectorized 20 ms sub-frame beam-sharing & 4-bucket 80 ms RT aging
    calibrated against Chapter 3 paper-simulation 2 ms PF scheduling (<1 ms per 1 s macro step).
  - Mode B (MicroSlotServiceModel): Exact 500 x 2 ms micro-slot execution with 3GPP Bessel H_sq,
    intra+inter-satellite CCI SINR, PF RB scheduling, 40-bucket RT aging, and frozen MAPPO-BH / USWGP-BH.
"""

from dataclasses import dataclass
import math
import numpy as np

import config as base_cfg
from handover.config import HandoverConfig
from handover.geometry import GroundCellLayout, TopologySnapshot
from handover.channel import DynamicChannelModel
from handover.traffic import TrafficSnapshot
from handover.mappo_adapter import DynamicMAPPOAdapter, DynamicUSWGPAdapter


@dataclass
class MacroServiceMetrics:
    """Aggregated 1 s macro-step service statistics (per cell and system-wide)."""

    cell_delivered_bits: np.ndarray     # (M,) total delivered bits in this 1 s
    cell_served_rt_pkts: np.ndarray     # (M,) RT packets delivered
    cell_expired_rt_pkts: np.ndarray    # (M,) RT packets expired (>80 ms TTL)
    cell_arrived_rt_pkts: np.ndarray    # (M,) RT packets arrived
    cell_rt_drop_rate: np.ndarray       # (M,) expired / max(served + expired, 1)
    cell_rt_deliv_delay_ms: np.ndarray  # (M,) mean delivered RT packet age (ms), 0 if none served
    cell_rt_urg: np.ndarray             # (M,) normalized HOL urgency in [0, 1]
    cell_total_queue_bits: np.ndarray   # (M,) end-of-step total queue bits (RT + BULK)
    cell_measured_interf_w: np.ndarray  # (M,) estimated/measured CCI power (W) for nominal EMA update
    system_per_user_mbps: float
    system_rt_drop_rate: float
    system_rt_deliv_delay_ms: float


class SurrogateServiceModel:
    """Mode A: Fast vectorized 20 ms sub-frame (50/sec) beam-hopping & 80 ms RT-TTL surrogate.

    Preserves exact physical mechanisms:
      - 4 RT age buckets of 20 ms each (TTL = 80 ms)
      - Handover interruption tau_HO pauses service while arrivals & RT aging continue
      - Never resets queues or HOL on handover
      - MMPP bursty RT arrivals (scaled to 20 ms step)
      - Calibrated beam-sharing (K_MAX=4) & cluster-level co-channel interference factor
    """

    SUBFRAME_MS = 20.0
    NUM_SUBFRAMES = 50   # 50 * 20 ms = 1000 ms
    NUM_BUCKETS = 4      # 4 * 20 ms = 80 ms TTL

    def __init__(
        self,
        cfg: HandoverConfig,
        layout: GroundCellLayout,
        cci_sig_ratio: float = 0.24,
        beam_eff_scale: float = 0.78,
        max_bulk_pkts: float = 40000.0,
    ):
        self.cfg = cfg
        self.layout = layout
        # Calibrated CCI-to-signal leakage ratio & PF/protocol efficiency scale (calibrated against Mode B MAPPO-2ms)
        self.cci_sig_ratio = float(cci_sig_ratio)
        self.beam_eff_scale = float(beam_eff_scale)
        self.max_bulk_pkts = float(max_bulk_pkts)

        # Precompute cell-to-cell proximity kernel for realistic cluster CCI scaling
        enu = layout.cell_pos_enu
        dmat = np.linalg.norm(enu[:, None, :] - enu[None, :, :], axis=-1)
        # Adjacent cells within ~35 km (25 km cluster neighbors) dominate CCI
        self._adj_kernel = np.exp(-((dmat / 28.0e3) ** 2))
        np.fill_diagonal(self._adj_kernel, 0.0)

        # Persistent cell-level queue states (packets, L_p = 1000 bits)
        self.rt_buckets = np.zeros((cfg.M, self.NUM_BUCKETS), dtype=np.float64)
        self.bulk_queue = np.zeros(cfg.M, dtype=np.float64)
        self.cell_on = np.zeros(cfg.M, dtype=bool)
        self.ho_remaining_ms = np.zeros(cfg.M, dtype=np.float64)

        # MMPP transition probabilities scaled from 2 ms -> 20 ms (10 micro slots)
        self.p_10 = 1.0 - (cfg.MMPP_STAY_ON ** 10)
        self.p_01 = self.p_10 * cfg.MMPP_P_ON / (1.0 - cfg.MMPP_P_ON)
        low_m = max((1.0 - cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT) / (1.0 - cfg.MMPP_P_ON), 0.12)
        mean_m = cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT + (1.0 - cfg.MMPP_P_ON) * low_m
        self.burst_mult = cfg.MMPP_BURST_MULT / mean_m
        self.low_mult = low_m / mean_m

    def reset(self, rng: np.random.Generator, traffic: TrafficSnapshot) -> None:
        self.rt_buckets.fill(0.0)
        self.bulk_queue.fill(0.0)
        self.ho_remaining_ms.fill(0.0)
        self.cell_on = rng.random(self.cfg.M) < self.cfg.MMPP_P_ON

    def trigger_handover_pause(self, cell_id: int, tau_ho_ms: float) -> None:
        """Pause service for cell_id during handover interruption WITHOUT resetting queues or HOL."""
        self.ho_remaining_ms[cell_id] = max(self.ho_remaining_ms[cell_id], float(tau_ho_ms))

    def get_queue_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (total_queue_bits (M,), rt_hol_urgency (M,) in [0, 1])."""
        rt_pkts = self.rt_buckets.sum(axis=1)
        tot_bits = (rt_pkts + self.bulk_queue) * self.cfg.L_P
        has_pkt = self.rt_buckets > 1e-6
        # Oldest non-empty bucket index in 0..3 -> age in {20, 40, 60, 80} ms
        rev_idx = np.argmax(has_pkt[:, ::-1], axis=1)
        hol_buckets = np.where(has_pkt.any(axis=1), self.NUM_BUCKETS - rev_idx, 0.0)
        rt_urg = hol_buckets / float(self.NUM_BUCKETS)
        return tot_bits, rt_urg

    def step_macro(
        self,
        Z: np.ndarray,
        cell_path_gain: np.ndarray,
        cell_rt_bps: np.ndarray,
        cell_bulk_bps: np.ndarray,
        num_users: int,
        rng: np.random.Generator,
    ) -> MacroServiceMetrics:
        """Execute one 1 s macro step via 50 vectorized 20 ms sub-frames."""
        n_act, m_cnt = Z.shape
        sub_s = self.SUBFRAME_MS * 1e-3  # 0.02 s
        lp = self.cfg.L_P
        k_max = float(self.cfg.K_MAX)
        p_nom = self.cfg.P_TOT_W / k_max
        noise_w = self.cfg.noise_full_bw_w

        # Map each cell to its serving local satellite (-1 if outage)
        has_sat = Z.sum(axis=0) > 0
        cell_sat = np.where(has_sat, np.argmax(Z, axis=0), -1).astype(np.int32)

        # Compute calibrated per-cell peak single-beam rate (bps) under cluster CCI
        sig_w = np.zeros(m_cnt, dtype=np.float64)
        valid_cells = np.flatnonzero(has_sat)
        if len(valid_cells) > 0:
            sig_w[valid_cells] = p_nom * cell_path_gain[cell_sat[valid_cells], valid_cells]

        # Active beam illumination duty proxy per cell for CCI estimation
        sat_counts = Z.sum(axis=1).astype(np.float64)  # (N_act,)
        cell_duty = np.zeros(m_cnt, dtype=np.float64)
        if len(valid_cells) > 0:
            cell_duty[valid_cells] = np.minimum(1.0, k_max / np.maximum(sat_counts[cell_sat[valid_cells]], 1.0))

        # Nearby active cells on OTHER satellites cause stronger uncoordinated CCI than same-sat neighbors
        same_sat = (cell_sat[:, None] == cell_sat[None, :]) & (cell_sat[:, None] >= 0)
        eff_kernel = np.where(same_sat, 0.35 * self._adj_kernel, 1.35 * self._adj_kernel)
        nbr_cci_load = eff_kernel @ cell_duty
        interf_w = np.where(has_sat, self.cci_sig_ratio * sig_w * np.maximum(nbr_cci_load, 0.45), 0.0)
        sinr = np.where(has_sat, sig_w / np.maximum(noise_w + interf_w, 1e-30), 0.0)
        beam_rate_bps = np.where(has_sat, self.beam_eff_scale * self.cfg.B_TOT * np.log2(1.0 + sinr), 0.0)
        beam_cap_pkts_sub = (beam_rate_bps * sub_s) / lp  # max pkts served if illuminated full 20 ms

        lam_rt_sub = (cell_rt_bps * sub_s) / lp
        lam_bulk_sub = (cell_bulk_bps * sub_s) / lp

        tot_deliv_pkts = np.zeros(m_cnt, dtype=np.float64)
        tot_served_rt = np.zeros(m_cnt, dtype=np.float64)
        tot_served_rt_age_ms = np.zeros(m_cnt, dtype=np.float64)
        tot_expired_rt = np.zeros(m_cnt, dtype=np.float64)
        tot_arrived_rt = np.zeros(m_cnt, dtype=np.float64)

        bucket_mid_ms = np.array([10.0, 30.0, 50.0, 70.0], dtype=np.float64)
        # Safe satellite index for vectorized bincount (map -1 -> n_act dummy bin)
        sat_bin = np.where(has_sat, cell_sat, n_act)
        n_bins = n_act + 1

        for _ in range(self.NUM_SUBFRAMES):
            # 1. Handover interruption pause fraction in this 20 ms sub-frame
            paused_ms = np.minimum(self.ho_remaining_ms, self.SUBFRAME_MS)
            self.ho_remaining_ms = np.maximum(self.ho_remaining_ms - self.SUBFRAME_MS, 0.0)
            avail_frac = np.clip(1.0 - paused_ms / self.SUBFRAME_MS, 0.0, 1.0)

            # 2. Vectorized urgency-weighted K_MAX beam sharing per satellite (2-pass water-filling via bincount)
            rt_q = self.rt_buckets.sum(axis=1)
            tot_q = rt_q + self.bulk_queue
            active_cell = has_sat & (tot_q > 1e-6) & (avail_frac > 1e-6)

            has_rt = self.rt_buckets > 1e-6
            hol_b = np.where(has_rt.any(axis=1), self.NUM_BUCKETS - np.argmax(has_rt[:, ::-1], axis=1), 0.0)
            priority = np.where(active_cell, 0.4 * tot_q + 1.8 * rt_q + 45.0 * hol_b + 1.0, 0.0)

            sat_w_sum = np.bincount(sat_bin, weights=priority, minlength=n_bins)
            raw_alloc = np.where(active_cell, k_max * priority / np.maximum(sat_w_sum[sat_bin], 1e-12), 0.0)
            alloc_frac = np.minimum(raw_alloc, avail_frac)
            # Second pass: redistribute leftover beam budget from cells capped at avail_frac
            sat_used = np.bincount(sat_bin, weights=alloc_frac, minlength=n_bins)
            rem_b = np.maximum(k_max - sat_used, 0.0)
            uncapped = active_cell & (alloc_frac < avail_frac - 1e-9)
            w_uncap = np.where(uncapped, priority, 0.0)
            sat_w_uncap = np.bincount(sat_bin, weights=w_uncap, minlength=n_bins)
            add_alloc = np.where(uncapped, rem_b[sat_bin] * w_uncap / np.maximum(sat_w_uncap[sat_bin], 1e-12), 0.0)
            alloc_frac = np.minimum(alloc_frac + add_alloc, avail_frac)

            cap_pkts = beam_cap_pkts_sub * alloc_frac
            rem_cap = cap_pkts.copy()

            # 3. Serve RT FIFO (oldest bucket 3 -> newest bucket 0), then BULK
            served_rt = np.zeros(m_cnt, dtype=np.float64)
            for b_idx in range(self.NUM_BUCKETS - 1, -1, -1):
                avail = self.rt_buckets[:, b_idx]
                take = np.minimum(avail, rem_cap)
                self.rt_buckets[:, b_idx] -= take
                rem_cap -= take
                served_rt += take
                tot_served_rt_age_ms += take * bucket_mid_ms[b_idx]

            served_bulk = np.minimum(self.bulk_queue, rem_cap)
            self.bulk_queue -= served_bulk
            tot_served_rt += served_rt
            tot_deliv_pkts += served_rt + served_bulk

            # 4. Expire remaining RT packets in oldest bucket (age >= 80 ms) and age survivors
            expired = self.rt_buckets[:, self.NUM_BUCKETS - 1].copy()
            tot_expired_rt += expired
            self.rt_buckets[:, 1:] = self.rt_buckets[:, :-1]

            # 5. MMPP bursty RT & BULK arrivals (with finite gateway BULK buffer cap)
            mult = np.where(self.cell_on, self.burst_mult, self.low_mult)
            new_rt = rng.poisson(np.maximum(lam_rt_sub * mult, 0.0)).astype(np.float64)
            new_bulk = rng.poisson(np.maximum(lam_bulk_sub * np.where(self.cell_on, 0.6, 1.08), 0.0)).astype(np.float64)
            self.rt_buckets[:, 0] = new_rt
            self.bulk_queue = np.minimum(self.bulk_queue + new_bulk, self.max_bulk_pkts)
            tot_arrived_rt += new_rt

            u_flip = rng.random(m_cnt)
            self.cell_on = np.where(self.cell_on, u_flip > self.p_10, u_flip < self.p_01)

        # Aggregate 1 s statistics
        cell_deliv_bits = tot_deliv_pkts * lp
        rt_exit = tot_served_rt + tot_expired_rt
        cell_rt_drop = np.where(rt_exit > 1e-9, tot_expired_rt / np.maximum(rt_exit, 1e-9), 0.0)
        cell_rt_delay_ms = np.where(tot_served_rt > 1e-9, tot_served_rt_age_ms / np.maximum(tot_served_rt, 1e-9), 0.0)
        tot_q_bits, rt_urg = self.get_queue_state()

        sys_rt_exit = float(rt_exit.sum())
        sys_rt_drop = float(tot_expired_rt.sum()) / sys_rt_exit if sys_rt_exit > 1e-9 else 0.0
        sys_rt_served = float(tot_served_rt.sum())
        sys_rt_delay = float(tot_served_rt_age_ms.sum()) / sys_rt_served if sys_rt_served > 1e-9 else 0.0
        sys_tput_mbps = float(cell_deliv_bits.sum()) / (max(num_users, 1) * 1e6)

        return MacroServiceMetrics(
            cell_delivered_bits=cell_deliv_bits,
            cell_served_rt_pkts=tot_served_rt,
            cell_expired_rt_pkts=tot_expired_rt,
            cell_arrived_rt_pkts=tot_arrived_rt,
            cell_rt_drop_rate=cell_rt_drop,
            cell_rt_deliv_delay_ms=cell_rt_delay_ms,
            cell_rt_urg=rt_urg,
            cell_total_queue_bits=tot_q_bits,
            cell_measured_interf_w=np.where(has_sat, interf_w, -1.0),
            system_per_user_mbps=sys_tput_mbps,
            system_rt_drop_rate=sys_rt_drop,
            system_rt_deliv_delay_ms=sys_rt_delay,
        )


class MicroSlotServiceModel:
    """Mode B: Exact 500 x 2 ms micro-slot execution using Chapter 3 paper-simulation physics & schedulers.

    Supports frozen MAPPO-APG-Local, frozen MAPPO-CoopGNN, and USWGP-BH with dynamic N_act(t),
    real intra+inter-satellite CCI SINR, PF RB scheduling, 40-bucket RT aging, and tau_HO pause.
    """

    def __init__(
        self,
        cfg: HandoverConfig,
        layout: GroundCellLayout,
        channel: DynamicChannelModel,
        scheduler_mode: str = "mappo_local",
        ckpt_path: str | None = None,
    ):
        self.cfg = cfg
        self.layout = layout
        self.channel = channel
        self.scheduler_mode = scheduler_mode

        if scheduler_mode == "mappo_local":
            self.policy = DynamicMAPPOAdapter(ckpt_path or cfg.mappo_local_ckpt)
        elif scheduler_mode == "mappo_coop":
            self.policy = DynamicMAPPOAdapter(ckpt_path or cfg.mappo_coop_ckpt)
        elif scheduler_mode == "uswgp_micro":
            self.policy = DynamicUSWGPAdapter()
        else:
            raise ValueError(f"Unknown MicroSlot scheduler_mode: {scheduler_mode}")

        # Persistent user-level queues (never reset on handover!)
        self.U = 0
        self.rt_buckets: np.ndarray | None = None
        self.bulk_queue: np.ndarray | None = None
        self.hist_tput: np.ndarray | None = None
        self.hist_buf: np.ndarray | None = None
        self.slot_idx = 0
        self.cell_on = np.zeros(cfg.M, dtype=bool)
        self.ho_remaining_slots = np.zeros(cfg.M, dtype=np.int32)

        self.p_10 = 1.0 - cfg.MMPP_STAY_ON
        self.p_01 = self.p_10 * cfg.MMPP_P_ON / (1.0 - cfg.MMPP_P_ON)
        low_m = max((1.0 - cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT) / (1.0 - cfg.MMPP_P_ON), 0.12)
        mean_m = cfg.MMPP_P_ON * cfg.MMPP_BURST_MULT + (1.0 - cfg.MMPP_P_ON) * low_m
        self.burst_mult = cfg.MMPP_BURST_MULT / mean_m
        self.low_mult = low_m / mean_m

    def reset(self, rng: np.random.Generator, traffic: TrafficSnapshot) -> None:
        self.U = traffic.U
        self.rt_buckets = np.zeros((self.U, self.cfg.T_TH_R), dtype=np.float64)
        self.bulk_queue = np.zeros(self.U, dtype=np.float64)
        self.hist_tput = np.full(self.U, 1e5, dtype=np.float64)
        self.hist_buf = np.zeros((self.cfg.PF_WINDOW, self.U), dtype=np.float64)
        self.slot_idx = 0
        self.cell_on = rng.random(self.cfg.M) < self.cfg.MMPP_P_ON
        self.ho_remaining_slots.fill(0)
        self.policy.reset()

    def trigger_handover_pause(self, cell_id: int, tau_ho_ms: float) -> None:
        pause_slots = max(0, int(round((tau_ho_ms * 1e-3) / self.cfg.T_SLOT)))
        self.ho_remaining_slots[cell_id] = max(int(self.ho_remaining_slots[cell_id]), pause_slots)

    def get_queue_state(self, traffic: TrafficSnapshot) -> tuple[np.ndarray, np.ndarray]:
        rt_q = self.rt_buckets.sum(axis=1)
        tot_q = rt_q + self.bulk_queue
        has_pkt = self.rt_buckets > 1e-6
        hol_u = np.where(has_pkt.any(axis=1), self.cfg.T_TH_R - np.argmax(has_pkt[:, ::-1], axis=1), 0.0)
        cell_bits = np.zeros(self.cfg.M, dtype=np.float64)
        cell_urg = np.zeros(self.cfg.M, dtype=np.float64)
        for m in range(self.cfg.M):
            us = traffic.cell_users_arr[m]
            if len(us) > 0:
                cell_bits[m] = float(tot_q[us].sum()) * self.cfg.L_P
                cell_urg[m] = float(hol_u[us].max()) / float(self.cfg.T_TH_R)
        return cell_bits, cell_urg

    def _build_local_graph_inputs(
        self,
        snap: TopologySnapshot,
        Z: np.ndarray,
        traffic: TrafficSnapshot,
        shadow_db: np.ndarray,
    ):
        """Build per-active-satellite candidate slots (N_act, J), priors Omega/C, H_sq, I_mat, and optional E_mat."""
        n_act, m_cnt = Z.shape
        j_dim = self.cfg.J
        cand_cells = np.full((n_act, j_dim), -1, dtype=np.int32)
        real_mask = np.zeros((n_act, j_dim), dtype=np.float32)
        action_mask = np.zeros((n_act, j_dim), dtype=np.float32)

        for n in range(n_act):
            assoc_m = np.flatnonzero(Z[n] == 1)
            vis_other = np.flatnonzero((snap.V_geo[n] == 1) & (Z[n] == 0))
            if len(vis_other) > 0:
                vis_other = vis_other[np.argsort(snap.slant_dist_m[n, vis_other])]
            cands = np.concatenate([assoc_m, vis_other])[:j_dim]
            cand_cells[n, : len(cands)] = cands
            real_mask[n, : len(cands)] = 1.0
            for j, m in enumerate(cands):
                if Z[n, m] == 1:
                    action_mask[n, j] = 1.0

        # Coverage overlap & common-visibility prior C_n
        vis_count = snap.V_geo.sum(axis=0).astype(np.float32)
        denom_n = max(n_act - 1, 1)
        v_cov = np.clip((vis_count - 1.0) / denom_n, 0.0, 1.0)
        v_local = np.zeros((n_act, j_dim), dtype=np.float32)
        C_prior = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)
        Omega_prior = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)

        # Cell centroids in ENU for distance prior Omega
        centroids = self.layout.cell_pos_enu.copy()
        for m in range(m_cnt):
            us = traffic.cell_users_arr[m]
            if len(us) > 0:
                centroids[m] = traffic.user_enu[us].mean(axis=0)

        for n in range(n_act):
            other_v = np.delete(snap.V_geo, n, axis=0) if n_act > 1 else np.zeros((1, m_cnt), dtype=np.int32)
            for i in range(j_dim):
                mi = int(cand_cells[n, i])
                if mi < 0:
                    continue
                v_local[n, i] = float(snap.V_geo[:, mi].sum() - snap.V_geo[n, mi]) / denom_n
                for j in range(j_dim):
                    mj = int(cand_cells[n, j])
                    if mj < 0:
                        continue
                    d_ij = float(np.linalg.norm(centroids[mi] - centroids[mj]))
                    Omega_prior[n, i, j] = 1.0 / (1.0 + math.exp((d_ij - self.cfg.D_TH_M) / self.cfg.ETA_M))
                    C_prior[n, i, j] = float(np.mean(other_v[:, mi] * other_v[:, mj]))

        H_sq, I_mat = self.channel.build_user_channel_tensors(
            snap, shadow_db, cand_cells, traffic.user_ecef, traffic.user_cell, traffic.cell_users_arr
        )

        # Optional cross-satellite structures for MAPPO-CoopGNN
        E_mat = None
        cross_ctx = None
        if isinstance(self.policy, DynamicMAPPOAdapter) and self.policy.cross_satellite and n_act > 0:
            members = np.eye(m_cnt, dtype=np.float64)[traffic.user_cell]
            members = members / np.maximum(members.sum(axis=0), 1.0)
            gains = H_sq.reshape(n_act * j_dim, traffic.U) @ members  # (N_act*J, M)
            cand_clip = cand_cells.clip(min=0)
            valid = cand_cells >= 0
            serving_beams = np.flatnonzero(action_mask.ravel())
            served_cells = cand_clip.ravel()[serving_beams]
            home_gains = np.zeros(m_cnt, dtype=np.float64)
            home_gains[served_cells] = gains[serving_beams, served_cells]

            has_sat = Z.sum(axis=0) > 0
            cell_sat_local = np.where(has_sat, np.argmax(Z, axis=0), -1)
            foreign = np.arange(n_act).repeat(j_dim)[:, None] != cell_sat_local[None, :]

            m_tot = n_act * j_dim
            E_mat = np.zeros((m_tot, m_tot), dtype=np.float32)
            if serving_beams.size > 0:
                denom = np.maximum(home_gains[served_cells], 1e-30)
                E_mat[:, serving_beams] = np.clip(gains[:, served_cells] / denom[None, :], 0.0, 5.0).astype(np.float32)
            cross_ctx = (gains, home_gains, foreign, cand_clip, valid, serving_beams, served_cells)

        return cand_cells, real_mask, action_mask, v_cov, v_local, Omega_prior, C_prior, H_sq, I_mat, E_mat, cross_ctx

    def step_macro(
        self,
        snap: TopologySnapshot,
        Z: np.ndarray,
        shadow_db: np.ndarray,
        traffic: TrafficSnapshot,
        cell_rate_mult: np.ndarray,
        rng: np.random.Generator,
        num_slots: int | None = None,
    ) -> MacroServiceMetrics:
        """Run 500 (or num_slots) exact 2 ms micro slots using frozen MAPPO-BH or USWGP-BH."""
        slots = int(self.cfg.slots_per_macro if num_slots is None else num_slots)
        n_act, m_cnt = Z.shape
        j_dim = self.cfg.J
        noise_w = self.cfg.noise_full_bw_w
        p_nom = self.cfg.P_TOT_W / self.cfg.K_MAX
        ages = np.arange(1, self.cfg.T_TH_R + 1, dtype=np.float64)

        (
            cand_cells,
            real_mask,
            action_mask,
            v_cov,
            v_local,
            Omega_prior,
            C_prior,
            H_sq,
            I_mat,
            E_mat,
            cross_ctx,
        ) = self._build_local_graph_inputs(snap, Z, traffic, shadow_db)

        # Precompute per-user arrival lambdas for this 1 s macro step
        u_mult = cell_rate_mult[traffic.user_cell]
        tot_lam_u = (traffic.base_user_rate_bps * u_mult * self.cfg.T_SLOT) / self.cfg.L_P
        lam_rt_u = tot_lam_u * traffic.user_rt_frac
        lam_bulk_u = tot_lam_u * (1.0 - traffic.user_rt_frac)

        # Precompute static cell SNR feature for F_rich col 4
        snr_feat = np.zeros((n_act, j_dim), dtype=np.float32)
        n_users_feat = np.zeros((n_act, j_dim), dtype=np.float32)
        for n in range(n_act):
            for j in range(j_dim):
                m = int(cand_cells[n, j])
                if m >= 0:
                    us = traffic.cell_users_arr[m]
                    if len(us) > 0:
                        snr_lin = (p_nom * H_sq[n, j, us]) / noise_w
                        snr_db = float(10.0 * np.log10(max(float(snr_lin.mean()), 1e-6)))
                        snr_feat[n, j] = float(np.clip((snr_db - 20.0) / 15.0, -2.5, 2.5))
                        n_users_feat[n, j] = min(float(len(us)) / 5.0, 2.0)

        cell_deliv_pkts = np.zeros(m_cnt, dtype=np.float64)
        cell_served_rt = np.zeros(m_cnt, dtype=np.float64)
        cell_served_rt_age_slots = np.zeros(m_cnt, dtype=np.float64)
        cell_expired_rt = np.zeros(m_cnt, dtype=np.float64)
        cell_arrived_rt = np.zeros(m_cnt, dtype=np.float64)
        cell_interf_sum = np.zeros(m_cnt, dtype=np.float64)
        cell_interf_cnt = np.zeros(m_cnt, dtype=np.int32)

        sat_ids_list = snap.active_sat_ids.tolist()

        for _ in range(slots):
            # Mask out cells currently in handover interruption pause
            paused_cells = self.ho_remaining_slots > 0
            eff_action_mask = action_mask.copy()
            if np.any(paused_cells):
                for n in range(n_act):
                    for j in range(j_dim):
                        m = int(cand_cells[n, j])
                        if m >= 0 and paused_cells[m]:
                            eff_action_mask[n, j] = 0.0
                self.ho_remaining_slots = np.maximum(self.ho_remaining_slots - 1, 0)

            # Build per-cell features F_basic (N_act, J, 5) and F_rich (N_act, J, 8/11)
            rt_q = self.rt_buckets.sum(axis=1)
            tot_q = rt_q + self.bulk_queue
            rt_avg_delay = np.where(rt_q > 0, (self.rt_buckets @ ages) / np.maximum(rt_q, 1e-9), 0.0)
            has_pkt = self.rt_buckets > 0
            hol_u = np.where(has_pkt.any(axis=1), self.cfg.T_TH_R - np.argmax(has_pkt[:, ::-1], axis=1), 0.0)

            cell_f5 = np.zeros((m_cnt, 5), dtype=np.float32)
            cell_rt_sum = np.zeros(m_cnt, dtype=np.float32)
            cell_bk_sum = np.zeros(m_cnt, dtype=np.float32)
            for m in range(m_cnt):
                us = traffic.cell_users_arr[m]
                if len(us) > 0:
                    rt_s = float(rt_q[us].sum())
                    bk_s = float(self.bulk_queue[us].sum())
                    cell_rt_sum[m] = min(rt_s / 800.0, 5.0)
                    cell_bk_sum[m] = min(bk_s / 2000.0, 5.0)
                    cell_f5[m, 0] = min((rt_s + bk_s) / 2000.0, 5.0)
                    cell_f5[m, 1] = float(rt_avg_delay[us].mean()) / self.cfg.T_TH_R
                    cell_f5[m, 2] = float(hol_u[us].max()) / self.cfg.T_TH_R
                    cell_f5[m, 3] = min(float(self.hist_tput[us].mean()) / 50e6, 2.0)
                cell_f5[m, 4] = v_cov[m]

            F_basic = np.zeros((n_act, j_dim, 5), dtype=np.float32)
            F_rich = np.zeros((n_act, j_dim, 8), dtype=np.float32)
            for n in range(n_act):
                for j in range(j_dim):
                    m = int(cand_cells[n, j])
                    if m >= 0:
                        F_basic[n, j] = cell_f5[m]
                        F_basic[n, j, 4] = v_local[n, j]
                        F_rich[n, j, 0] = cell_rt_sum[m]
                        F_rich[n, j, 1] = cell_bk_sum[m]
                        F_rich[n, j, 2] = cell_f5[m, 1]
                        F_rich[n, j, 3] = cell_f5[m, 2]
                        F_rich[n, j, 4] = snr_feat[n, j]
                        F_rich[n, j, 5] = cell_f5[m, 3]
                        F_rich[n, j, 6] = v_local[n, j]
                        F_rich[n, j, 7] = n_users_feat[n, j]

            if cross_ctx is not None:
                gains, home_gains, foreign, cand_clip, valid, serving_beams, served_cells = cross_ctx
                demand = (F_rich[..., 0] + F_rich[..., 1]) * eff_action_mask
                occupied = demand > 1e-6
                priority = np.where(occupied, np.sqrt(demand), 0.0)
                act_cnt = np.minimum(occupied.sum(axis=1, keepdims=True), self.cfg.K_MAX)
                prob = np.minimum(act_cnt * priority / np.maximum(priority.sum(axis=1, keepdims=True), 1e-9), 1.0)
                exp_p = self.cfg.P_TOT_W * prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-9)
                cross_gains = gains * foreign
                incoming = exp_p.ravel() @ cross_gains
                own_rx = exp_p.ravel() @ (gains - cross_gains)
                sig_c = np.zeros(m_cnt, dtype=np.float64)
                if serving_beams.size > 0:
                    sig_c[served_cells] = exp_p.ravel()[serving_beams] * home_gains[served_cells]
                denom_c = noise_w + np.maximum(own_rx - sig_c, 0.0) + incoming
                direct_ratio = incoming[cand_clip] / np.maximum(
                    p_nom * gains[np.arange(gains.shape[0]), cand_clip.ravel()].reshape(n_act, j_dim),
                    noise_w,
                )
                rt_dl = self.rt_buckets[:, -8:].sum(axis=1)
                near_exp = np.bincount(traffic.user_cell, weights=rt_dl, minlength=m_cnt)
                v_w = np.zeros(m_cnt, dtype=np.float64)
                if serving_beams.size > 0:
                    flat = F_rich.reshape(-1, 8)[serving_beams]
                    v_w[served_cells] = np.minimum(0.35 + flat[:, 0] * 2.0 + flat[:, 1] + flat[:, 3], 3.0)
                leak = p_nom * cross_gains
                harm = (np.log1p(sig_c / denom_c) - np.log1p(sig_c / (denom_c + leak))) @ v_w
                msgs = np.stack([
                    np.log1p(direct_ratio),
                    np.log1p(harm.reshape(n_act, j_dim)),
                    np.log1p(near_exp[cand_clip] / 40.0),
                ], axis=-1) * valid[..., None]
                F_rich = np.concatenate([F_rich, msgs.astype(np.float32)], axis=-1)

            # Policy step
            if isinstance(self.policy, DynamicMAPPOAdapter):
                beam_sel, power_ratios = self.policy.act(
                    sat_ids_list, F_rich, I_mat, Omega_prior, C_prior, real_mask, eff_action_mask,
                    E_mat=E_mat, deterministic=True
                )
            else:
                beam_sel, power_ratios = self.policy.act(sat_ids_list, F_basic, eff_action_mask)

            b = (beam_sel > 0.5).astype(np.float64) * eff_action_mask
            p = np.maximum(power_ratios, 0.0) * b
            p_sum = p.sum(axis=1, keepdims=True)
            p = np.where(p_sum > 1e-12, p / np.maximum(p_sum, 1e-12), 0.0)
            P_beam = p * self.cfg.P_TOT_W

            # Exact multi-satellite CCI SINR & PF RB scheduling
            total_rx = np.einsum("nj,nju->u", P_beam, H_sq) if n_act > 0 else np.zeros(self.U, dtype=np.float64)
            user_rate = np.zeros(self.U, dtype=np.float64)
            has_backlog = (rt_q + self.bulk_queue) > 1e-9

            active_n, active_j = np.where(b > 0.5)
            for n, j in zip(active_n, active_j):
                m = int(cand_cells[n, j])
                us = traffic.cell_users_arr[m]
                if len(us) == 0:
                    continue
                active_us = us[has_backlog[us]]
                if len(active_us) == 0:
                    continue
                sig = P_beam[n, j] * H_sq[n, j, active_us]
                interf = np.maximum(total_rx[active_us] - sig, 0.0)
                cell_interf_sum[m] += float(interf.mean())
                cell_interf_cnt[m] += 1
                sinr = sig / (noise_w + interf)
                se = np.log2(1.0 + sinr)
                if len(active_us) == 1:
                    user_rate[active_us[0]] = self.cfg.B_TOT * se[0]
                else:
                    hist = np.maximum(self.hist_tput[active_us], 1e4)
                    rb_counts = np.zeros(len(active_us), dtype=np.float64)
                    est_hist = hist.copy()
                    rb_bw = self.cfg.B_TOT / self.cfg.N_RB
                    step_rbs = 4
                    for _rb in range(self.cfg.N_RB // step_rbs):
                        winner = int(np.argmax(se / est_hist))
                        rb_counts[winner] += step_rbs
                        est_hist[winner] += (rb_bw * step_rbs * se[winner]) / self.cfg.PF_WINDOW
                    user_rate[active_us] = (rb_counts * rb_bw) * se

            # Serve queues: RT FIFO oldest-first, then BULK
            cap_pkts = (user_rate * self.cfg.T_SLOT) / self.cfg.L_P
            rem_cap = cap_pkts.copy()
            served_rt_u = np.zeros(self.U, dtype=np.float64)
            served_rt_age_u = np.zeros(self.U, dtype=np.float64)
            for age_idx in range(self.cfg.T_TH_R - 1, -1, -1):
                avail = self.rt_buckets[:, age_idx]
                take = np.minimum(avail, rem_cap)
                self.rt_buckets[:, age_idx] -= take
                rem_cap -= take
                served_rt_u += take
                served_rt_age_u += take * (age_idx + 1)

            served_bulk_u = np.minimum(self.bulk_queue, rem_cap)
            self.bulk_queue -= served_bulk_u
            served_tot_u = served_rt_u + served_bulk_u

            # Update PF 50-slot window
            delivered_bps = (served_tot_u * self.cfg.L_P) / self.cfg.T_SLOT
            self.hist_buf[self.slot_idx % self.cfg.PF_WINDOW] = delivered_bps
            self.slot_idx += 1
            win = min(self.slot_idx, self.cfg.PF_WINDOW)
            self.hist_tput = self.hist_buf[:win].mean(axis=0)

            # Expire & age RT buckets, insert MMPP arrivals
            expired_u = self.rt_buckets[:, self.cfg.T_TH_R - 1].copy()
            self.rt_buckets[:, 1:] = self.rt_buckets[:, :-1]
            mult = np.where(self.cell_on[traffic.user_cell], self.burst_mult, self.low_mult)
            new_rt_u = rng.poisson(np.maximum(lam_rt_u * mult, 0.0)).astype(np.float64)
            new_bulk_u = rng.poisson(
                np.maximum(lam_bulk_u * np.where(self.cell_on[traffic.user_cell], 0.6, 1.08), 0.0)
            ).astype(np.float64)
            self.rt_buckets[:, 0] = new_rt_u
            self.bulk_queue += new_bulk_u
            u_flip = rng.random(m_cnt)
            self.cell_on = np.where(self.cell_on, u_flip > self.p_10, u_flip < self.p_01)

            # Accumulate per-cell metrics
            np.add.at(cell_deliv_pkts, traffic.user_cell, served_tot_u)
            np.add.at(cell_served_rt, traffic.user_cell, served_rt_u)
            np.add.at(cell_served_rt_age_slots, traffic.user_cell, served_rt_age_u)
            np.add.at(cell_expired_rt, traffic.user_cell, expired_u)
            np.add.at(cell_arrived_rt, traffic.user_cell, new_rt_u)

        cell_deliv_bits = cell_deliv_pkts * self.cfg.L_P
        rt_exit = cell_served_rt + cell_expired_rt
        cell_rt_drop = np.where(rt_exit > 1e-9, cell_expired_rt / np.maximum(rt_exit, 1e-9), 0.0)
        slot_ms = self.cfg.T_SLOT * 1000.0
        cell_rt_delay_ms = np.where(
            cell_served_rt > 1e-9, (cell_served_rt_age_slots / np.maximum(cell_served_rt, 1e-9)) * slot_ms, 0.0
        )
        tot_q_bits, rt_urg = self.get_queue_state(traffic)
        meas_interf = np.where(cell_interf_cnt > 0, cell_interf_sum / np.maximum(cell_interf_cnt, 1), -1.0)

        duration_s = slots * self.cfg.T_SLOT
        sys_rt_exit = float(rt_exit.sum())
        sys_rt_drop = float(cell_expired_rt.sum()) / sys_rt_exit if sys_rt_exit > 1e-9 else 0.0
        sys_rt_served = float(cell_served_rt.sum())
        sys_rt_delay = (
            (float(cell_served_rt_age_slots.sum()) / sys_rt_served) * slot_ms if sys_rt_served > 1e-9 else 0.0
        )
        sys_tput_mbps = float(cell_deliv_bits.sum()) / (max(duration_s, 1e-9) * max(self.U, 1) * 1e6)

        return MacroServiceMetrics(
            cell_delivered_bits=cell_deliv_bits,
            cell_served_rt_pkts=cell_served_rt,
            cell_expired_rt_pkts=cell_expired_rt,
            cell_arrived_rt_pkts=cell_arrived_rt,
            cell_rt_drop_rate=cell_rt_drop,
            cell_rt_deliv_delay_ms=cell_rt_delay_ms,
            cell_rt_urg=rt_urg,
            cell_total_queue_bits=tot_q_bits,
            cell_measured_interf_w=meas_interf,
            system_per_user_mbps=sys_tput_mbps,
            system_rt_drop_rate=sys_rt_drop,
            system_rt_deliv_delay_ms=sys_rt_delay,
        )
