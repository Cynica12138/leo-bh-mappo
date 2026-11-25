# -*- coding: utf-8 -*-
"""Unified satellite-side concurrent admission control (Make-Before-Break) and P0-safe ISA."""

from dataclasses import dataclass
import math
import numpy as np

from handover.config import HandoverConfig
from handover.geometry import TopologySnapshot


@dataclass
class AdmissionResult:
    """Outcome of unified satellite-side concurrent handover admission."""

    Z_new: np.ndarray            # (N_act, M) int32 binary strictly feasible association
    new_cell_sat_id: np.ndarray  # (M,) int32 global sat_id, -1 if outage
    handover_flags: np.ndarray           # (M,) int32 {0, 1} switched to a different satellite
    outage_flags: np.ndarray             # (M,) int32 {0, 1} unconnected (Z[:, m].sum() == 0)
    rejected_proactive_flags: np.ndarray # (M,) int32 {0, 1} proactive switch rejected by admission
    rejected_mandatory_flags: np.ndarray # (M,) int32 {0, 1} mandatory primary choice rejected by admission
    rejected_proactive: int
    rejected_mandatory: int


def solve_isa_feasible(
    cell_demands: np.ndarray,
    V_geo: np.ndarray,
    J: int = 14,
    elev_rad: np.ndarray | None = None,
    T_0: float = 100.0,
    T_min: float = 1e-3,
    rho: float = 0.95,
    L_max: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P0-safe Improved Simulated Annealing (ISA) strictly respecting counts[n] <= J and empty visibility.

    When elev_rad is provided, greedy initialization prefers higher-elevation satellites among
    low-load candidates so geographically clustered cells start on nearby overhead satellites.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_sats, m_cells = V_geo.shape
    Z = np.zeros((n_sats, m_cells), dtype=np.int32)
    if n_sats == 0:
        return Z

    assoc = np.full(m_cells, -1, dtype=np.int32)
    loads = np.zeros(n_sats, dtype=np.float64)
    counts = np.zeros(n_sats, dtype=np.int32)

    vis_deg = V_geo.sum(axis=0)
    order = np.lexsort((-cell_demands, vis_deg))
    mean_dem = max(float(cell_demands.mean()), 1.0)
    for m in order:
        cands = np.flatnonzero((V_geo[:, m] == 1) & (counts < J))
        if len(cands) == 0:
            continue
        if elev_rad is not None:
            # Balance load while favoring higher-elevation local satellites
            score = loads[cands] - 0.35 * mean_dem * (elev_rad[cands, m] / (math.pi / 2.0))
            best_n = int(cands[np.argmin(score)])
        else:
            best_n = int(cands[np.argmin(loads[cands])])
        assoc[m] = best_n
        loads[best_n] += float(cell_demands[m])
        counts[best_n] += 1

    connected_cells = np.flatnonzero(assoc >= 0)
    vis_sats = [np.flatnonzero(V_geo[:, m] == 1) for m in range(m_cells)]
    multi_vis_cells = np.array([m for m in connected_cells if len(vis_sats[m]) > 1], dtype=np.int32)
    if len(multi_vis_cells) == 0:
        for m in connected_cells:
            Z[assoc[m], m] = 1
        return Z

    sat_cells = [set() for _ in range(n_sats)]
    for m in connected_cells:
        sat_cells[assoc[m]].add(int(m))

    overlap_pairs = []
    for i in range(n_sats):
        for j in range(i + 1, n_sats):
            if np.any((V_geo[i] == 1) & (V_geo[j] == 1)):
                overlap_pairs.append((i, j))

    cur_E = float(np.dot(loads, loads))
    best_E = cur_E
    best_assoc = assoc.copy()

    scale = max(1.0, cur_E / max(n_sats, 1) * 0.05)
    T = T_0
    while T > T_min:
        eff_T = T * scale / T_0
        for _ in range(L_max):
            if rng.random() < 0.5 or len(overlap_pairs) == 0:
                m = int(multi_vis_cells[rng.integers(0, len(multi_vis_cells))])
                old_n = int(assoc[m])
                cands = [int(n) for n in vis_sats[m] if n != old_n and counts[n] < J]
                if not cands:
                    continue
                new_n = cands[int(rng.integers(0, len(cands)))]
                dm = float(cell_demands[m])
                dE = (loads[old_n] - dm) ** 2 + (loads[new_n] + dm) ** 2 - loads[old_n] ** 2 - loads[new_n] ** 2
                if dE < 0.0 or rng.random() < math.exp(-dE / max(eff_T, 1e-12)):
                    assoc[m] = new_n
                    loads[old_n] -= dm
                    loads[new_n] += dm
                    counts[old_n] -= 1
                    counts[new_n] += 1
                    sat_cells[old_n].remove(m)
                    sat_cells[new_n].add(m)
                    cur_E += dE
                    if cur_E < best_E:
                        best_E = cur_E
                        best_assoc = assoc.copy()
            else:
                i, j = overlap_pairs[int(rng.integers(0, len(overlap_pairs)))]
                cand_i = [m for m in sat_cells[i] if V_geo[j, m] == 1]
                cand_j = [m for m in sat_cells[j] if V_geo[i, m] == 1]
                if not cand_i or not cand_j:
                    continue
                mi = cand_i[int(rng.integers(0, len(cand_i)))]
                mj = cand_j[int(rng.integers(0, len(cand_j)))]
                dmi, dmj = float(cell_demands[mi]), float(cell_demands[mj])
                diff = dmj - dmi
                dE = (loads[i] + diff) ** 2 + (loads[j] - diff) ** 2 - loads[i] ** 2 - loads[j] ** 2
                if dE < 0.0 or rng.random() < math.exp(-dE / max(eff_T, 1e-12)):
                    assoc[mi] = j
                    assoc[mj] = i
                    loads[i] += diff
                    loads[j] -= diff
                    sat_cells[i].remove(mi)
                    sat_cells[i].add(mj)
                    sat_cells[j].remove(mj)
                    sat_cells[j].add(mi)
                    cur_E += dE
                    if cur_E < best_E:
                        best_E = cur_E
                        best_assoc = assoc.copy()
        T *= rho

    for m in range(m_cells):
        if best_assoc[m] >= 0:
            Z[best_assoc[m], m] = 1
    return Z


class UnifiedAdmissionController:
    """Satellite-side concurrent admission controller (Make-Before-Break) shared by all algorithms.

    Existing visible connections hold their serving slot until an incoming handover to a new
    target satellite is ACCEPTED, so rejected proactive requests cleanly stay on their serving
    satellite without suffering artificial outages.
    """

    def __init__(self, cfg: HandoverConfig):
        self.cfg = cfg

    def admit(
        self,
        snap: TopologySnapshot,
        prev_cell_sat_id: np.ndarray,
        proposed_target_local: np.ndarray,
        is_mandatory: np.ndarray,
        rt_urgency: np.ndarray,
        d_nm: np.ndarray,
        fallback_rankings: list[list[int]] | None = None,
    ) -> AdmissionResult:
        """Resolve concurrent cell handover requests and return strictly feasible Z_new."""
        n_act = len(snap.active_sat_ids)
        m_cnt = self.cfg.M
        j_cap = self.cfg.J
        k_max = float(self.cfg.K_MAX)
        rho_soft = self.cfg.rho_max_soft

        Z_new = np.zeros((n_act, m_cnt), dtype=np.int32)
        cur_local = np.full(m_cnt, -1, dtype=np.int32)
        counts = np.zeros(n_act, dtype=np.int32)
        rho = np.zeros(n_act, dtype=np.float64)

        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        prev_local = np.array(
            [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in prev_cell_sat_id],
            dtype=np.int32,
        )
        prev_vis = np.array(
            [(prev_local[m] >= 0) and (snap.V_geo[prev_local[m], m] == 1) for m in range(m_cnt)],
            dtype=bool,
        )

        # Step 1 (Make-Before-Break): Hold existing visible connections until switched
        for m in range(m_cnt):
            if prev_vis[m]:
                loc = int(prev_local[m])
                if counts[loc] < j_cap:
                    cur_local[m] = loc
                    counts[loc] += 1
                    rho[loc] += float(d_nm[loc, m]) / k_max

        # Step 2: Collect all cells requesting a satellite change or currently disconnected
        requests: list[int] = []
        for m in range(m_cnt):
            prop = int(proposed_target_local[m])
            if cur_local[m] < 0:
                requests.append(m)
            elif prop >= 0 and prop != cur_local[m]:
                requests.append(m)

        # Sort requests: mandatory holding a slot first (so departing cells free slots!),
        # then mandatory currently-outage cells, then proactive; tie-break by descending RT urgency.
        requests.sort(
            key=lambda m: (
                0 if (is_mandatory[m] and cur_local[m] >= 0) else (1 if is_mandatory[m] else 2),
                -float(rt_urgency[m]),
                -float(d_nm[int(proposed_target_local[m]), m]) if 0 <= int(proposed_target_local[m]) < n_act else 0.0,
            )
        )

        rejected_pro = 0
        rejected_mand = 0
        rejected_pro_flags = np.zeros(m_cnt, dtype=np.int32)
        rejected_mand_flags = np.zeros(m_cnt, dtype=np.int32)

        def _commit_switch(cell_m: int, new_loc: int) -> None:
            old_loc = int(cur_local[cell_m])
            if old_loc >= 0:
                counts[old_loc] -= 1
                rho[old_loc] = max(0.0, rho[old_loc] - float(d_nm[old_loc, cell_m]) / k_max)
            cur_local[cell_m] = new_loc
            counts[new_loc] += 1
            rho[new_loc] += float(d_nm[new_loc, cell_m]) / k_max

        # Step 3: Process requests sequentially under hard (J) and soft (rho_max / load-improving) admission rules
        for m in requests:
            mand = bool(is_mandatory[m])
            prop = int(proposed_target_local[m])
            old_loc = int(cur_local[m])

            if 0 <= prop < n_act and prop != old_loc and snap.V_geo[prop, m] == 1:
                hard_ok = counts[prop] < j_cap
                new_target_rho = rho[prop] + float(d_nm[prop, m]) / k_max
                old_source_rho = rho[old_loc] if old_loc >= 0 else 1e9
                soft_ok = (new_target_rho <= rho_soft) or (new_target_rho < old_source_rho - 0.10)
                if hard_ok and (mand or soft_ok):
                    _commit_switch(m, prop)
                    continue

            # Rejected primary choice -> fallback
            if not mand:
                rejected_pro += 1
                rejected_pro_flags[m] = 1
                # Proactive request keeps its reserved old_loc cleanly!
                continue

            rejected_mand += 1
            rejected_mand_flags[m] = 1
            if fallback_rankings is not None and len(fallback_rankings[m]) > 0:
                cands = [c for c in fallback_rankings[m] if 0 <= c < n_act and c != old_loc and snap.V_geo[c, m] == 1]
            else:
                vis_c = np.flatnonzero(snap.V_geo[:, m] == 1)
                cands = [int(c) for c in vis_c[np.argsort(-snap.rvt_s[vis_c, m])] if int(c) != old_loc]

            # Pass A: prefer candidate satisfying both hard J and soft rho_max
            chosen = -1
            for c in cands:
                if counts[c] < j_cap and (rho[c] + float(d_nm[c, m]) / k_max) <= rho_soft:
                    chosen = c
                    break
            # Pass B: relax soft rho_max for mandatory requests to prevent artificial outages
            if chosen < 0:
                for c in cands:
                    if counts[c] < j_cap:
                        chosen = c
                        break
            if chosen >= 0:
                _commit_switch(m, chosen)
            # Else: if old_loc >= 0 (e.g., RVT <= T_cov still visible), keep old_loc until it actually sets;
            # if old_loc < 0, cell stays in outage (-1).

        # Step 3b (Post-release outage recovery): if earlier disconnected cells failed before later
        # cells vacated full satellites in the same step, let them claim newly freed visible slots.
        for m in range(m_cnt):
            if cur_local[m] >= 0:
                continue
            if fallback_rankings is not None and len(fallback_rankings[m]) > 0:
                cands = [c for c in fallback_rankings[m] if 0 <= c < n_act and snap.V_geo[c, m] == 1]
            else:
                vis_c = np.flatnonzero(snap.V_geo[:, m] == 1)
                cands = [int(c) for c in vis_c[np.argsort(-snap.rvt_s[vis_c, m])]]
            for c in cands:
                if counts[c] < j_cap:
                    _commit_switch(m, c)
                    break

        # Step 4: Assemble Z_new, global sat IDs, handover flags, and outage flags
        new_sat_id = np.full(m_cnt, -1, dtype=np.int32)
        for m in range(m_cnt):
            loc = int(cur_local[m])
            if loc >= 0:
                Z_new[loc, m] = 1
                new_sat_id[m] = int(snap.active_sat_ids[loc])

        outage_flags = (new_sat_id < 0).astype(np.int32)
        handover_flags = (
            (new_sat_id >= 0) & (prev_cell_sat_id >= 0) & (new_sat_id != prev_cell_sat_id)
        ).astype(np.int32)

        return AdmissionResult(
            Z_new=Z_new,
            new_cell_sat_id=new_sat_id,
            handover_flags=handover_flags,
            outage_flags=outage_flags,
            rejected_proactive_flags=rejected_pro_flags,
            rejected_mandatory_flags=rejected_mand_flags,
            rejected_proactive=rejected_pro,
            rejected_mandatory=rejected_mand,
        )
