# -*- coding: utf-8 -*-
"""Traditional, risk-aware, graph-theoretic, and optimization handover baselines:
  - Max-RVT : Longest Remaining Visibility Time geometric baseline
  - ISA-HO  : Chapter 3 periodic ISA load-balancing re-association
  - Swap-HO : Beam-load-aware proactive Shift & Swap (Zhu, Sun & Peng, IEEE TCOM 2025)
  - RA-HO   : Link-layer VaR Risk-Aware Handover (adapted from Almekhlafi & Kurt, arXiv 2026)
  - MCMF-HO : Triggered-subset Min-Cost Max-Flow exact matching (Wiriya et al. GLOBECOM 2025; Yang et al. IoTJ 2026)
  - RH-STG  : Rolling-Horizon Space-Time Graph shortest path (Wu et al. CL 2016; Liu et al. 2024)
"""

from abc import ABC, abstractmethod
import math
import networkx as nx
import numpy as np

from handover.admission import solve_isa_feasible
from handover.dynamic_env import DynamicLEOHandoverEnv, EnvStepOutput


class BaseHandoverPolicy(ABC):
    """Unified interface for all handover algorithms (traditional and DRL)."""

    name: str = "Base"
    bypass_trigger_gate: bool = False

    def reset(self) -> None:
        pass

    @abstractmethod
    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        """Return (proposed_target_local (M,), optional fallback_rankings per cell)."""
        raise NotImplementedError


class MaxRVTPolicy(BaseHandoverPolicy):
    """Longest Remaining Visibility Time (Max-RVT) geometric baseline."""

    name = "Max-RVT"
    bypass_trigger_gate = False

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        snap = out.snap
        trig = out.trigger
        m_cnt = snap.V_geo.shape[1]
        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        for m in range(m_cnt):
            if trig.i_trig[m] == 0:
                continue
            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            if len(vis) == 0:
                continue
            order = vis[np.lexsort((-snap.elev_rad[vis, m], -snap.rvt_s[vis, m]))]
            fallbacks[m] = order.tolist()
            cur_local = sat_id_to_local.get(int(out.cell_sat_id[m]), -1)
            # If triggered proactively or mandatorily, pick highest-RVT visible sat (excluding exiting cur_local if RVT<=T_cov)
            chosen = int(order[0])
            if trig.i_force[m] == 1 and cur_local >= 0 and snap.rvt_s[cur_local, m] <= env.cfg.T_cov_s:
                for c in order:
                    if int(c) != cur_local:
                        chosen = int(c)
                        break
            prop[m] = chosen
        return prop, fallbacks


class ISAHOPolicy(BaseHandoverPolicy):
    """Cross-chapter baseline: periodically re-runs Chapter 3 ISA load-balancing association."""

    name = "ISA-HO"
    bypass_trigger_gate = True

    def __init__(self, reassoc_period_s: int = 15, seed: int = 42):
        self.reassoc_period_s = max(1, int(reassoc_period_s))
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.base_seed)

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        snap = out.snap
        trig = out.trigger
        m_cnt = snap.V_geo.shape[1]
        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        do_global_isa = (out.t > 0 and (out.t % self.reassoc_period_s == 0))
        if do_global_isa:
            tot_bps, _, _ = env.traffic_model.get_offered_rates_bps(out.t)
            # Filter out satellites about to set within T_cov_s when feasible to avoid immediate re-handover
            V_eff = snap.V_geo.copy()
            long_rvt = (snap.V_geo == 1) & (snap.rvt_s > env.cfg.T_cov_s)
            for m in range(m_cnt):
                if long_rvt[:, m].sum() > 0:
                    V_eff[:, m] = long_rvt[:, m].astype(np.int32)
            Z_isa = solve_isa_feasible(
                tot_bps / 1e6,
                V_eff,
                J=env.cfg.J,
                elev_rad=snap.elev_rad,
                L_max=60,
                rng=self.rng,
            )
            for m in range(m_cnt):
                s = np.flatnonzero(Z_isa[:, m] == 1)
                if len(s) == 1:
                    prop[m] = int(s[0])
                vis = np.flatnonzero(snap.V_geo[:, m] == 1)
                if len(vis) > 0:
                    fallbacks[m] = vis[np.argsort(-snap.rvt_s[vis, m])].tolist()
            return prop, fallbacks

        # Between periodic ISA updates, handle mandatory coverage triggers via lowest-load feasible visible sat
        counts = out.Z.sum(axis=1).astype(np.int32).copy()
        rho = out.sat_rho.copy()
        k_max = float(env.cfg.K_MAX)
        for m in range(m_cnt):
            if trig.i_force[m] == 0:
                continue
            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            if len(vis) == 0:
                continue
            scores = rho[vis] + 0.25 * (counts[vis] / float(env.cfg.J)) - 0.15 * (snap.rvt_s[vis, m] / 150.0)
            order = vis[np.argsort(scores)]
            fallbacks[m] = order.tolist()
            best = int(order[0])
            prop[m] = best
            counts[best] += 1
            rho[best] += float(out.d_nm[best, m]) / k_max
        return prop, fallbacks


class SwapHOPolicy(BaseHandoverPolicy):
    """Adapted from Sun et al. (IEEE TCOM 2025): beam-load-aware proactive Shift & Swap handover.

    Coordinates triggered cells in two stages:
      1. Mandatory coverage handovers assigned to best feasible low-load / long-RVT satellite;
      2. Proactive load/link handovers perform greedy Shift or 2-cell Swap whenever it reduces
         satellite squared beam-load (rho^2) by a hysteresis margin while preserving RVT & SINR.
    """

    name = "Swap-HO"
    bypass_trigger_gate = False

    def __init__(self, rel_gain_margin: float = 0.08, min_target_rvt_s: float = 25.0):
        self.rel_gain_margin = float(rel_gain_margin)
        self.min_target_rvt_s = float(min_target_rvt_s)

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        snap = out.snap
        trig = out.trigger
        n_act, m_cnt = snap.V_geo.shape
        k_max = float(env.cfg.K_MAX)
        j_cap = env.cfg.J

        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        cur_local = np.array(
            [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
            dtype=np.int32,
        )

        # Working copy of projected counts and beam loads for coordinated multi-cell decisions
        counts = out.Z.sum(axis=1).astype(np.int32).copy()
        rho = out.sat_rho.copy()
        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        # Stage 1: Mandatory handovers (most constrained / highest RT urgency first)
        mand_cells = np.flatnonzero(trig.i_force > 0)
        vis_deg = snap.V_geo.sum(axis=0)
        mand_cells = mand_cells[np.lexsort((-env.cell_rt_urg[mand_cells], vis_deg[mand_cells]))]
        for m in mand_cells:
            old_n = int(cur_local[m])
            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            if len(vis) == 0:
                continue
            # Filter out dying satellites (RVT <= T_cov_s) whenever healthier visible candidates exist
            healthy = [int(n) for n in vis if int(n) != old_n and snap.rvt_s[n, m] > env.cfg.T_cov_s + 5.0]
            if healthy:
                cands = np.array(healthy, dtype=np.int32)
            else:
                alt = [int(n) for n in vis if int(n) != old_n]
                cands = np.array(alt if len(alt) > 0 else vis.tolist(), dtype=np.int32)

            # Composite utility: projected beam load + steep slot-headroom barrier + RVT + SINR
            proj_rho = rho[cands] + out.d_nm[cands, m] / k_max
            slot_frac = counts[cands] / float(j_cap)
            cap_pen = np.where(counts[cands] >= j_cap, 25.0, 4.5 * (slot_frac ** 3))
            rvt_bonus = np.minimum(snap.rvt_s[cands, m] / 120.0, 1.0)
            sinr_bonus = np.clip((out.links["nominal_sinr_db"][cands, m] - 5.0) / 20.0, -0.5, 1.0)
            cost = 0.85 * proj_rho + cap_pen - 1.20 * rvt_bonus - 0.25 * sinr_bonus
            order = cands[np.argsort(cost)]
            # Always append any remaining visible satellites to fallback so admission never causes artificial outage
            order_list = order.tolist()
            order_set = set(order_list)
            extra = [int(n) for n in vis if int(n) != old_n and int(n) not in order_set]
            extra.sort(key=lambda n: (-float(snap.rvt_s[n, m]), int(counts[n])))
            fallbacks[m] = order_list + extra

            best = int(order[0])
            prop[m] = best
            if old_n >= 0 and counts[old_n] > 0:
                counts[old_n] -= 1
                rho[old_n] = max(0.0, rho[old_n] - float(out.d_nm[old_n, m]) / k_max)
            counts[best] += 1
            rho[best] += float(out.d_nm[best, m]) / k_max

        # Stage 2: Proactive Shift & 2-cell Swap with strict relative gain hysteresis (suppresses ping-pong)
        pro_cells = np.flatnonzero((trig.i_pro > 0) & (trig.i_force == 0))
        pro_cells = pro_cells[np.argsort(-env.cell_rt_urg[pro_cells])]
        sat_pro_set: dict[int, list[int]] = {}
        for m in pro_cells:
            n = int(cur_local[m])
            if n >= 0:
                sat_pro_set.setdefault(n, []).append(int(m))

        swapped = set()
        for m in pro_cells:
            m = int(m)
            if m in swapped:
                continue
            old_n = int(cur_local[m])
            if old_n < 0 or (rho[old_n] <= env.cfg.rho_th and counts[old_n] < j_cap):
                continue
            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            alt = [int(n) for n in vis if int(n) != old_n and snap.rvt_s[n, m] >= self.min_target_rvt_s]
            if not alt:
                continue

            best_rel_gain = self.rel_gain_margin
            best_dst = -1
            dm_old = float(out.d_nm[old_n, m]) / k_max
            for dst in alt:
                if counts[dst] >= j_cap - 1:  # keep 1-slot headroom for mandatory arrivals
                    continue
                dm_new = float(out.d_nm[dst, m]) / k_max
                if rho[dst] + dm_new > env.cfg.rho_max_soft:
                    continue
                # Combined beam-load + barrier slot-occupancy energy (offloads 14/14 full sats onto rising sats)
                b_old = 0.35 * max(0.0, float(counts[old_n]) - (j_cap - 2)) ** 2
                b_dst = 0.35 * max(0.0, float(counts[dst]) - (j_cap - 2)) ** 2
                b_old_n = 0.35 * max(0.0, float(counts[old_n] - 1) - (j_cap - 2)) ** 2
                b_dst_n = 0.35 * max(0.0, float(counts[dst] + 1) - (j_cap - 2)) ** 2
                old_e = rho[old_n] ** 2 + rho[dst] ** 2 + 1.2 * ((counts[old_n] / j_cap) ** 2 + (counts[dst] / j_cap) ** 2) + b_old + b_dst
                new_e = (
                    max(0.0, rho[old_n] - dm_old) ** 2
                    + (rho[dst] + dm_new) ** 2
                    + 1.2 * (((counts[old_n] - 1) / j_cap) ** 2 + ((counts[dst] + 1) / j_cap) ** 2)
                    + b_old_n
                    + b_dst_n
                )
                rel_gain = (old_e - new_e) / max(old_e, 1e-6)
                if rel_gain > best_rel_gain:
                    best_rel_gain = rel_gain
                    best_dst = dst

            if best_dst >= 0:
                prop[m] = best_dst
                fallbacks[m] = [best_dst]
                counts[old_n] -= 1
                rho[old_n] = max(0.0, rho[old_n] - dm_old)
                counts[best_dst] += 1
                rho[best_dst] += float(out.d_nm[best_dst, m]) / k_max
                swapped.add(m)
                continue

            best_swap = None
            best_swap_rel = self.rel_gain_margin
            for dst in alt:
                for m2 in sat_pro_set.get(dst, []):
                    if m2 in swapped or snap.V_geo[old_n, m2] != 1:
                        continue
                    if snap.rvt_s[old_n, m2] < self.min_target_rvt_s:
                        continue
                    dm1_old = float(out.d_nm[old_n, m]) / k_max
                    dm1_new = float(out.d_nm[dst, m]) / k_max
                    dm2_old = float(out.d_nm[dst, m2]) / k_max
                    dm2_new = float(out.d_nm[old_n, m2]) / k_max
                    r1_new = max(0.0, rho[old_n] - dm1_old + dm2_new)
                    r2_new = max(0.0, rho[dst] - dm2_old + dm1_new)
                    old_e = rho[old_n] ** 2 + rho[dst] ** 2
                    new_e = r1_new ** 2 + r2_new ** 2
                    rel_gain = (old_e - new_e) / max(old_e, 1e-6)
                    if rel_gain > best_swap_rel:
                        best_swap_rel = rel_gain
                        best_swap = (dst, m2, r1_new, r2_new)

            if best_swap is not None:
                dst, m2, r1_new, r2_new = best_swap
                prop[m] = dst
                prop[m2] = old_n
                fallbacks[m] = [dst]
                fallbacks[m2] = [old_n]
                rho[old_n] = r1_new
                rho[dst] = r2_new
                swapped.add(m)
                swapped.add(m2)

        return prop, fallbacks


class RAHOPolicy(BaseHandoverPolicy):
    """Link-layer Risk-Aware Handover (adapted from Almekhlafi & Kurt, arXiv 2026).

    Computes the elevation-dependent AR(1) predictive (1 - epsilon) Value-at-Risk (VaR)
    conservative link rate R_hat^VaR_{n,m}(t) for each visible satellite using the known
    3GPP TR 38.811 shadow fading standard deviation sigma_sh(theta), filters risk-safe
    candidates satisfying R_hat^VaR >= R_m^min, and ranks them by link-risk utility and RVT.
    """

    name = "RA-HO"
    bypass_trigger_gate = False

    def __init__(
        self,
        epsilon: float = 0.05,
        risk_horizon_s: int = 5,
        proactive_gain_margin: float = 0.15,
        min_target_rvt_s: float = 25.0,
    ):
        self.epsilon = float(epsilon)
        # Normal upper quantile z_{1-eps} (1.64485 for eps=0.05, 1.28155 for eps=0.10)
        if abs(self.epsilon - 0.05) < 1e-9:
            self.z_eps = 1.6448536
        elif abs(self.epsilon - 0.10) < 1e-9:
            self.z_eps = 1.2815516
        elif abs(self.epsilon - 0.01) < 1e-9:
            self.z_eps = 2.3263479
        else:
            from scipy.stats import norm
            self.z_eps = float(norm.ppf(1.0 - self.epsilon))
        self.risk_horizon_s = max(1, int(risk_horizon_s))
        self.proactive_gain_margin = float(proactive_gain_margin)
        self.min_target_rvt_s = float(min_target_rvt_s)

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        cfg = env.cfg
        snap = out.snap
        trig = out.trigger
        n_act, m_cnt = snap.V_geo.shape
        j_cap = cfg.J

        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        trig_cells = np.flatnonzero(trig.i_trig > 0)
        if len(trig_cells) == 0 or n_act == 0:
            return prop, fallbacks

        # Compute (1 - epsilon) conservative link rate R_hat^VaR (N_act, M) without peeking at future noise
        rho_h = cfg.shadow_rho ** self.risk_horizon_s
        sig_db = env.channel.shadowing.sigma_db(snap.elev_rad)
        pred_std_db = math.sqrt(max(1.0 - rho_h * rho_h, 0.0)) * sig_db
        shadow_var_db = rho_h * out.links["shadow_db"] + self.z_eps * pred_std_db

        dist = np.maximum(snap.slant_dist_m.astype(np.float64), 1.0)
        fspl = (4.0 * math.pi * dist / cfg.LAMBDA) ** 2
        g_var = (cfg.G_MAX_T_LIN * cfg.G_R_LIN) / (fspl * (10.0 ** (shadow_var_db / 10.0)))
        p_nom = cfg.P_TOT_W / cfg.K_MAX
        sig_var = p_nom * g_var
        denom_var = cfg.noise_full_bw_w + np.maximum(
            env.channel.ema_cell_interf_w[None, :], 0.5 * cfg.nominal_interf_ratio * sig_var
        )
        rate_var_bps = cfg.B_TOT * np.log2(1.0 + sig_var / np.maximum(denom_var, 1e-30))

        tot_bps, _, _ = env.traffic_model.get_offered_rates_bps(out.t)
        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        cur_local = np.array(
            [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
            dtype=np.int32,
        )
        counts = out.Z.sum(axis=1).astype(np.int32).copy()

        # Process mandatory cells first, then proactive cells by RT urgency
        order_cells = trig_cells[np.lexsort((-env.cell_rt_urg[trig_cells], -trig.i_force[trig_cells]))]
        recent_mask = (out.t - env._last_ho_time) <= cfg.ping_pong_window_s

        for raw_m in order_cells:
            m = int(raw_m)
            old_n = int(cur_local[m])
            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            if len(vis) == 0:
                continue

            req_m = float(tot_bps[m])
            rvt_v = snap.rvt_s[vis, m]
            rate_v = rate_var_bps[vis, m]
            safe_mask = rate_v >= req_m
            pp_pen = np.zeros(len(vis), dtype=np.float64)
            if recent_mask[m]:
                pp_pen = np.where(snap.active_sat_ids[vis] == env._last_ho_from_sat[m], 0.45, 0.0)

            # Pure link-risk utility: VaR safety filter first, then RVT + conservative rate + slot feasibility
            slot_pen = np.where(counts[vis] >= j_cap, 3.0, 0.45 * ((counts[vis] / float(j_cap)) ** 2))
            score = (
                0.50 * safe_mask.astype(np.float64)
                + 0.60 * np.minimum(rvt_v / 120.0, 1.0)
                + 0.35 * np.clip(rate_v / 250e6, 0.0, 1.5)
                - slot_pen
                - pp_pen
            )
            ranked = vis[np.argsort(-score)]

            if trig.i_force[m] == 1:
                healthy_free = [
                    int(n) for n in ranked
                    if int(n) != old_n and snap.rvt_s[n, m] > cfg.T_cov_s + 5.0 and counts[n] < j_cap
                ]
                healthy_any = [int(n) for n in ranked if int(n) != old_n and snap.rvt_s[n, m] > cfg.T_cov_s + 5.0]
                if healthy_free:
                    best = healthy_free[0]
                    fb_order = healthy_free + [int(n) for n in ranked if int(n) not in set(healthy_free)]
                elif healthy_any:
                    best = healthy_any[0]
                    fb_order = healthy_any + [int(n) for n in ranked if int(n) not in set(healthy_any)]
                else:
                    alt = [int(n) for n in ranked if int(n) != old_n]
                    best = alt[0] if alt else int(ranked[0])
                    fb_order = alt + ([old_n] if old_n >= 0 and old_n in vis else [])
                prop[m] = best
                fallbacks[m] = fb_order
                if old_n >= 0 and counts[old_n] > 0:
                    counts[old_n] -= 1
                counts[best] += 1
            else:
                fallbacks[m] = ranked.tolist()
                # Proactive trigger: if current link fails VaR threshold or serving sat is congested/full,
                # move to best risk-safe long-RVT satellite with headroom; otherwise require VaR rate gain margin.
                cur_ok = (
                    old_n >= 0
                    and snap.V_geo[old_n, m] == 1
                    and snap.rvt_s[old_n, m] >= self.min_target_rvt_s
                )
                cur_rate_var = float(rate_var_bps[old_n, m]) if cur_ok else 0.0
                cur_risk_violated = (not cur_ok) or (cur_rate_var < req_m)
                cur_slot_full = (old_n >= 0) and (counts[old_n] >= j_cap)
                best_cand = -1
                for n in ranked:
                    n_i = int(n)
                    if n_i == old_n:
                        continue
                    if snap.rvt_s[n_i, m] < self.min_target_rvt_s or counts[n_i] >= j_cap - 1:
                        continue
                    if cur_risk_violated or cur_slot_full:
                        if rate_var_bps[n_i, m] >= req_m or rate_var_bps[n_i, m] > cur_rate_var:
                            best_cand = n_i
                            break
                    elif rate_var_bps[n_i, m] >= cur_rate_var * (1.0 + self.proactive_gain_margin):
                        best_cand = n_i
                        break
                if best_cand >= 0:
                    prop[m] = best_cand
                    if old_n >= 0 and counts[old_n] > 0:
                        counts[old_n] -= 1
                    counts[best_cand] += 1

        return prop, fallbacks


class MCMFHOPolicy(BaseHandoverPolicy):
    """Triggered-subset Minimum-Cost Maximum-Flow exact matching baseline.

    Adapted from Wiriya et al. (IEEE GLOBECOM 2025) & Yang et al. (IEEE IoTJ 2026):
      - Fixes untriggered cells' occupied slots k_n^fix and beam loads rho_n^fix;
      - Jointly assigns all triggered cells M_trig(t) via exact integer min-cost max-flow;
      - Approximates convex quadratic load balancing using (J - k_n^fix) parallel unit-capacity
        arcs from each active satellite n -> T with strictly increasing marginal cost.
    """

    name = "MCMF-HO"
    bypass_trigger_gate = False

    def __init__(
        self,
        lambda_d: float = 1.50,
        lambda_r: float = 1.20,
        lambda_h: float = 0.65,
        lambda_pp: float = 1.80,
        lambda_bal: float = 0.22,
        lambda_rho: float = 0.75,
        cost_scale: int = 1000,
        m_big: float = 1000.0,
    ):
        self.lambda_d = float(lambda_d)
        self.lambda_r = float(lambda_r)
        self.lambda_h = float(lambda_h)
        self.lambda_pp = float(lambda_pp)
        self.lambda_bal = float(lambda_bal)
        self.lambda_rho = float(lambda_rho)
        self.cost_scale = int(cost_scale)
        self.m_big = float(m_big)

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        cfg = env.cfg
        snap = out.snap
        trig = out.trigger
        n_act, m_cnt = snap.V_geo.shape
        j_cap = cfg.J

        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        trig_cells = np.flatnonzero(trig.i_trig > 0)
        if len(trig_cells) == 0 or n_act == 0:
            return prop, fallbacks

        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        cur_local = np.array(
            [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
            dtype=np.int32,
        )

        # Fix untriggered cells' occupied slot counts k_n^fix and beam loads rho_n^fix
        untrig_mask = trig.i_trig == 0
        Z_fix = out.Z[:, untrig_mask]
        k_fix = Z_fix.sum(axis=1).astype(np.int32)
        rho_fix = (Z_fix * out.d_nm[:, untrig_mask]).sum(axis=1) / float(cfg.K_MAX)

        G = nx.MultiDiGraph()
        n_trig = len(trig_cells)
        G.add_node("S", demand=-n_trig)
        G.add_node("T", demand=n_trig)

        scale = self.cost_scale
        # Active satellite n -> sink T: (J - k_n^fix) parallel unit-capacity convex-cost arcs
        for n in range(n_act):
            rem = max(0, j_cap - int(k_fix[n]))
            base_rho_cost = self.lambda_rho * float(rho_fix[n])
            for l in range(1, rem + 1):
                arc_cost = self.lambda_bal * float(int(k_fix[n]) + l) + base_rho_cost
                G.add_edge(("sat", n), "T", key=l, capacity=1, weight=int(round(scale * arc_cost)))

        recent_mask = (out.t - env._last_ho_time) <= cfg.ping_pong_window_s
        overflow_w = int(round(scale * self.m_big))
        has_healthy_map: dict[int, bool] = {}

        for raw_m in trig_cells:
            m = int(raw_m)
            c_node = ("cell", m)
            G.add_edge("S", c_node, key=0, capacity=1, weight=0)
            G.add_edge(c_node, "T", key=0, capacity=1, weight=overflow_w)

            vis = np.flatnonzero(snap.V_geo[:, m] == 1)
            if len(vis) == 0:
                has_healthy_map[m] = False
                continue

            old_n = int(cur_local[m])
            has_healthy = any((int(n) != old_n) and (snap.rvt_s[n, m] > cfg.T_cov_s) for n in vis)
            has_healthy_map[m] = has_healthy
            ho_c = float(out.cell_ho_cost[m])

            cand_rank: list[tuple[float, int]] = []
            for raw_n in vis:
                n = int(raw_n)
                is_switch = n != old_n
                rvt_norm = min(float(snap.rvt_s[n, m]) / 120.0, 1.0)
                c_mn = (
                    self.lambda_d * float(out.d_nm[n, m])
                    - self.lambda_r * rvt_norm
                    + (self.lambda_h * ho_c if is_switch else 0.0)
                )
                if is_switch and recent_mask[m] and int(snap.active_sat_ids[n]) == int(env._last_ho_from_sat[m]):
                    c_mn += self.lambda_pp
                if not is_switch:
                    if (not trig.stay_allowed[m]) or (
                        trig.i_force[m] == 1 and snap.rvt_s[n, m] <= cfg.T_cov_s and has_healthy
                    ):
                        c_mn += 80.0
                elif snap.rvt_s[n, m] <= cfg.T_cov_s and has_healthy:
                    c_mn += 25.0

                G.add_edge(c_node, ("sat", n), key=0, capacity=1, weight=int(round(scale * c_mn)))
                est_total = c_mn + self.lambda_bal * float(int(k_fix[n]) + 1) + self.lambda_rho * float(rho_fix[n])
                cand_rank.append((est_total, n))

            cand_rank.sort(key=lambda x: x[0])
            fallbacks[m] = [n for _, n in cand_rank]

        flow_dict = nx.min_cost_flow(G)

        for raw_m in trig_cells:
            m = int(raw_m)
            c_node = ("cell", m)
            chosen_n = -1
            for nbr, key_flows in flow_dict.get(c_node, {}).items():
                if nbr == "T":
                    continue
                if sum(key_flows.values()) > 0:
                    chosen_n = int(nbr[1])
                    break

            old_n = int(cur_local[m])
            if chosen_n < 0:
                prop[m] = -1
            elif (
                chosen_n == old_n
                and bool(trig.stay_allowed[m])
                and not (trig.i_force[m] == 1 and snap.rvt_s[chosen_n, m] <= cfg.T_cov_s and has_healthy_map[m])
            ):
                prop[m] = -1
            else:
                prop[m] = chosen_n

        return prop, fallbacks


class RHSTGPolicy(BaseHandoverPolicy):
    """Rolling-Horizon Space-Time Graph (RH-STG) deterministic shortest-path handover baseline.

    Adapted from Wu et al. (IEEE CL 2016) & Liu et al. (2024):
      - Looks ahead W = 60 s sampled every dt = 2 s (L = 30 DAG layers) using deterministic
        Walker Delta ephemeris (no future AR(1) shadowing noise or future traffic drift peeking);
      - Solves exact shortest path via O(L * K) backward dynamic programming per triggered cell;
      - Executes only the step-0 satellite decision and submits to unified admission control.
    """

    name = "RH-STG"
    bypass_trigger_gate = False

    def __init__(
        self,
        horizon_s: int = 60,
        dt_s: int = 2,
        lambda_d: float = 1.40,
        lambda_rho: float = 1.00,
        lambda_slot: float = 0.80,
        lambda_h: float = 1.10,
        lambda_pp: float = 2.00,
    ):
        self.horizon_s = max(2, int(horizon_s))
        self.dt_s = max(1, int(dt_s))
        self.num_layers = max(2, self.horizon_s // self.dt_s)
        self.lambda_d = float(lambda_d)
        self.lambda_rho = float(lambda_rho)
        self.lambda_slot = float(lambda_slot)
        self.lambda_h = float(lambda_h)
        self.lambda_pp = float(lambda_pp)

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        cfg = env.cfg
        snap = out.snap
        trig = out.trigger
        n_act, m_cnt = snap.V_geo.shape
        j_cap = float(cfg.J)

        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]

        trig_cells = np.flatnonzero(trig.i_trig > 0)
        if len(trig_cells) == 0 or n_act == 0:
            return prop, fallbacks

        geom = env.geom
        assert geom._vgeo_all is not None and geom._dist_all is not None and geom._window_sat_ids is not None
        max_t = geom._vgeo_all.shape[0] - 1
        t_layers = np.array([min(out.t + l * self.dt_s, max_t) for l in range(self.num_layers)], dtype=np.int32)
        L = len(t_layers)

        win_ids = geom._window_sat_ids
        n_win = len(win_ids)
        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        win_to_local = np.array([sat_id_to_local.get(int(sid), -1) for sid in win_ids], dtype=np.int32)

        cur_local = np.array(
            [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
            dtype=np.int32,
        )

        # Snapshot of current satellite loads & AR(1) shadow states mapped to window indices (no future peeking)
        rho_mean = float(out.sat_rho.mean()) if n_act > 0 else 0.5
        cur_rho_win = np.full(n_win, 0.75 * rho_mean, dtype=np.float64)
        cur_cnt_win = np.zeros(n_win, dtype=np.float64)
        cur_sh_win = np.zeros((n_win, m_cnt), dtype=np.float64)
        active_win_mask = win_to_local >= 0
        if np.any(active_win_mask):
            aw_idx = np.flatnonzero(active_win_mask)
            al_idx = win_to_local[aw_idx]
            sat_counts = out.Z.sum(axis=1).astype(np.float64)
            cur_rho_win[aw_idx] = out.sat_rho[al_idx]
            cur_cnt_win[aw_idx] = sat_counts[al_idx] / j_cap
            cur_sh_win[aw_idx] = out.links["shadow_db"][al_idx]

        # Frozen step-t cell traffic demand snapshot (no future burst/drift peeking)
        tot_bps, _, _ = env.traffic_model.get_offered_rates_bps(out.t)
        eff_q_bits = np.minimum(env.cell_queue_bits, 2.0 * np.maximum(tot_bps, 2.0e6) * cfg.T_H)
        req_bps = tot_bps + eff_q_bits / cfg.T_H

        layer_dt_scale = float(self.dt_s) / 13.5
        lags = np.arange(L, dtype=np.float64) * float(self.dt_s)
        rho_decay = np.exp(-lags / 30.0)[:, None]                # (L, 1)
        sh_decay = (cfg.shadow_rho ** lags)[:, None]             # (L, 1)
        fspl_coeff = (4.0 * math.pi / cfg.LAMBDA) ** 2
        gain_num = cfg.G_MAX_T_LIN * cfg.G_R_LIN
        p_nom = cfg.P_TOT_W / cfg.K_MAX
        recent_mask = (out.t - env._last_ho_time) <= cfg.ping_pong_window_s
        INF = 1e9

        for raw_m in trig_cells:
            m = int(raw_m)
            v_layers = geom._vgeo_all[t_layers, :, m]  # (L, N_win) bool
            cand_w = np.flatnonzero(v_layers.any(axis=0))
            if len(cand_w) == 0:
                continue

            v_sub = v_layers[:, cand_w]                                     # (L, K)
            d_sub = geom._dist_all[t_layers[:, None], cand_w[None, :], m].astype(np.float64)

            # Expected AR(1) shadow attenuation decay (zero future noise)
            sh_exp = sh_decay * cur_sh_win[cand_w, m][None, :]              # (L, K)
            g_det = gain_num / (fspl_coeff * (np.maximum(d_sub, 1.0) ** 2) * (10.0 ** (sh_exp / 10.0)))
            sig = p_nom * g_det
            denom = cfg.noise_full_bw_w + np.maximum(
                env.channel.ema_cell_interf_w[m], 0.5 * cfg.nominal_interf_ratio * sig
            )
            rate_det = cfg.B_TOT * np.log2(1.0 + sig / np.maximum(denom, 1e-30))
            d_hat = req_bps[m] / (rate_det + 1e3)

            # Overwrite layer 0 with exact observed d_nm for currently active satellites
            loc_w = win_to_local[cand_w]
            act_k = loc_w >= 0
            if np.any(act_k):
                d_hat[0, act_k] = out.d_nm[loc_w[act_k], m]

            # Predicted satellite load decays smoothly toward network mean over the 60 s horizon
            rho_hat = np.where(
                act_k[None, :],
                rho_decay * cur_rho_win[cand_w][None, :] + (1.0 - rho_decay) * rho_mean,
                0.75 * rho_mean,
            )
            node_cost = layer_dt_scale * (self.lambda_d * d_hat + self.lambda_rho * rho_hat)
            node_cost[0] += self.lambda_slot * (cur_cnt_win[cand_w] ** 2)

            # Exact O(L * K) backward dynamic programming on the space-time DAG
            ho_pen = self.lambda_h * float(out.cell_ho_cost[m])
            V = np.where(v_sub[-1], node_cost[-1], INF)
            for l in range(L - 2, -1, -1):
                v_min = float(V.min())
                if v_min >= INF * 0.5:
                    cont = np.zeros_like(V)
                else:
                    cont = np.minimum(V, v_min + ho_pen)
                V = np.where(v_sub[l], node_cost[l] + cont, INF)

            # Extract step-0 feasible active satellites and apply initial transition / expiring penalties
            old_n = int(cur_local[m])
            vis_l0_k = np.flatnonzero(v_sub[0] & act_k)
            if len(vis_l0_k) == 0:
                vis_fb = np.flatnonzero(snap.V_geo[:, m] == 1)
                if len(vis_fb) > 0:
                    order_fb = vis_fb[np.argsort(-snap.rvt_s[vis_fb, m])]
                    prop[m] = int(order_fb[0])
                    fallbacks[m] = order_fb.tolist()
                continue

            l0_locals = loc_w[vis_l0_k]
            has_healthy = any((int(n) != old_n) and (snap.rvt_s[n, m] > cfg.T_cov_s) for n in l0_locals)

            scored: list[tuple[float, int]] = []
            for k_idx, loc_n in zip(vis_l0_k, l0_locals):
                n = int(loc_n)
                total_c = float(V[k_idx])
                is_switch = n != old_n
                if is_switch:
                    total_c += ho_pen
                    if recent_mask[m] and int(snap.active_sat_ids[n]) == int(env._last_ho_from_sat[m]):
                        total_c += self.lambda_pp
                    if snap.rvt_s[n, m] <= cfg.T_cov_s and has_healthy:
                        total_c += 25.0
                else:
                    if (not trig.stay_allowed[m]) or (
                        trig.i_force[m] == 1 and snap.rvt_s[n, m] <= cfg.T_cov_s and has_healthy
                    ):
                        total_c += 80.0
                scored.append((total_c, n))

            scored.sort(key=lambda x: x[0])
            best_n = scored[0][1]
            fallbacks[m] = [n for _, n in scored]
            # Append any remaining visible sats (edge case safety)
            fb_set = set(fallbacks[m])
            for n in np.flatnonzero(snap.V_geo[:, m] == 1):
                if int(n) not in fb_set:
                    fallbacks[m].append(int(n))

            if (
                best_n == old_n
                and bool(trig.stay_allowed[m])
                and not (trig.i_force[m] == 1 and snap.rvt_s[best_n, m] <= cfg.T_cov_s and has_healthy)
            ):
                prop[m] = -1
            else:
                prop[m] = best_n

        return prop, fallbacks
