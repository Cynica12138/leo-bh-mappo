# -*- coding: utf-8 -*-
"""Traditional handover baselines: Max-RVT, ISA-HO (Chapter 3 periodic ISA), and Swap-HO (Sun et al. TCOM 2025)."""

from abc import ABC, abstractmethod
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

    def __init__(self, rel_gain_margin: float = 0.15, min_target_rvt_s: float = 35.0):
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

            # Composite utility: projected beam load + quadratic slot-headroom penalty + strong RVT term + SINR
            proj_rho = rho[cands] + out.d_nm[cands, m] / k_max
            slot_frac = counts[cands] / float(j_cap)
            cap_pen = np.where(counts[cands] >= j_cap, 20.0, 2.2 * (slot_frac ** 2))
            rvt_bonus = np.minimum(snap.rvt_s[cands, m] / 120.0, 1.0)
            sinr_bonus = np.clip((out.links["nominal_sinr_db"][cands, m] - 5.0) / 20.0, -0.5, 1.0)
            cost = 0.85 * proj_rho + cap_pen - 1.35 * rvt_bonus - 0.25 * sinr_bonus
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
                # Combined beam-load + slot-occupancy energy (encourages offloading full 14/14 sats onto rising sats)
                old_e = rho[old_n] ** 2 + rho[dst] ** 2 + 0.8 * ((counts[old_n] / j_cap) ** 2 + (counts[dst] / j_cap) ** 2)
                new_e = (
                    max(0.0, rho[old_n] - dm_old) ** 2
                    + (rho[dst] + dm_new) ** 2
                    + 0.8 * (((counts[old_n] - 1) / j_cap) ** 2 + ((counts[dst] + 1) / j_cap) ** 2)
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
