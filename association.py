# -*- coding: utf-8 -*-
"""Cell-satellite association algorithms: ISA, NLB, MEA, and MMTG."""

import math
import numpy as np
import config as cfg


def _energy(loads: np.ndarray) -> float:
    return float(np.dot(loads, loads))


def greedy_init(cell_demands: np.ndarray, V: np.ndarray, J: int = cfg.J) -> np.ndarray:
    """Greedy initialization under visibility and capacity (<= J) constraints."""
    N, M = V.shape
    assoc = np.full(M, -1, dtype=np.int32)
    loads = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)

    # Sort cells by descending demand for better feasible packing
    order = np.argsort(-cell_demands)
    for m in order:
        vis = np.where((V[:, m] == 1) & (counts < J))[0]
        if len(vis) == 0:
            # Fallback: pick visible satellite with smallest count
            vis = np.where(V[:, m] == 1)[0]
        best_n = vis[np.argmin(loads[vis])]
        assoc[m] = best_n
        loads[best_n] += cell_demands[m]
        counts[best_n] += 1
    return assoc


def solve_isa(
    cell_demands: np.ndarray,
    V: np.ndarray,
    J: int = cfg.J,
    T_0: float = cfg.T_0,
    T_min: float = cfg.T_MIN,
    rho: float = cfg.RHO,
    L_max: int = cfg.L_MAX,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Improved Simulated Annealing (ISA) for cell-satellite load-balancing association.

    Returns binary association matrix Z of shape (N, M).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N, M = V.shape
    assoc = greedy_init(cell_demands, V, J)
    loads = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)
    for m in range(M):
        n = assoc[m]
        loads[n] += cell_demands[m]
        counts[n] += 1

    # Precompute visible satellites per cell and satellite cells lists
    vis_sats = [np.where(V[:, m] == 1)[0] for m in range(M)]
    multi_vis_cells = np.array([m for m in range(M) if len(vis_sats[m]) > 1], dtype=np.int32)
    if len(multi_vis_cells) == 0:
        Z = np.zeros((N, M), dtype=np.int32)
        Z[assoc, np.arange(M)] = 1
        return Z

    sat_cells = [set() for _ in range(N)]
    for m in range(M):
        sat_cells[assoc[m]].add(m)

    # Pairs of satellites with overlapping visibility
    overlap_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.any((V[i] == 1) & (V[j] == 1)):
                overlap_pairs.append((i, j))

    cur_E = _energy(loads)
    best_E = cur_E
    best_assoc = assoc.copy()

    # Scale temperature relative to energy scale for effective annealing
    scale = max(1.0, cur_E / max(N, 1) * 0.05)
    T = T_0
    while T > T_min:
        eff_T = T * scale / T_0
        for _ in range(L_max):
            if rng.random() < 0.5 or len(overlap_pairs) == 0:
                # Shift operator
                m = int(multi_vis_cells[rng.integers(0, len(multi_vis_cells))])
                old_n = int(assoc[m])
                cands = [n for n in vis_sats[m] if n != old_n and counts[n] < J]
                if not cands:
                    continue
                new_n = int(cands[rng.integers(0, len(cands))])
                dm = cell_demands[m]
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
                # Swap operator
                i, j = overlap_pairs[rng.integers(0, len(overlap_pairs))]
                cand_i = [m for m in sat_cells[i] if V[j, m] == 1]
                cand_j = [m for m in sat_cells[j] if V[i, m] == 1]
                if not cand_i or not cand_j:
                    continue
                mi = cand_i[rng.integers(0, len(cand_i))]
                mj = cand_j[rng.integers(0, len(cand_j))]
                dmi, dmj = cell_demands[mi], cell_demands[mj]
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

    Z = np.zeros((N, M), dtype=np.int32)
    Z[best_assoc, np.arange(M)] = 1
    return Z


def solve_nlb(V: np.ndarray, J: int = cfg.J, rng: np.random.Generator | None = None) -> np.ndarray:
    """No Load Balancing (NLB): random feasible association."""
    if rng is None:
        rng = np.random.default_rng(42)
    N, M = V.shape
    assoc = np.full(M, -1, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    for m in rng.permutation(M):
        cands = np.where((V[:, m] == 1) & (counts < J))[0]
        if len(cands) == 0:
            cands = np.where(V[:, m] == 1)[0]
        n = int(cands[rng.integers(0, len(cands))])
        assoc[m] = n
        counts[n] += 1
    Z = np.zeros((N, M), dtype=np.int32)
    Z[assoc, np.arange(M)] = 1
    return Z


def solve_mea(elev_rad: np.ndarray, V: np.ndarray, J: int = cfg.J) -> np.ndarray:
    """Maximum Elevation Angle (MEA): assign each cell to highest-elevation feasible satellite."""
    N, M = V.shape
    assoc = np.full(M, -1, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    # Assign cells with fewest visible satellites first to guarantee feasibility
    order = np.argsort(V.sum(axis=0))
    for m in order:
        cands = np.where((V[:, m] == 1) & (counts < J))[0]
        if len(cands) == 0:
            cands = np.where(V[:, m] == 1)[0]
        n = int(cands[np.argmax(elev_rad[cands, m])])
        assoc[m] = n
        counts[n] += 1
    Z = np.zeros((N, M), dtype=np.int32)
    Z[assoc, np.arange(M)] = 1
    return Z


def solve_mmtg(cell_demands: np.ndarray, V: np.ndarray, J: int = cfg.J) -> np.ndarray:
    """Max-Min Transfer Greedy (MMTG) association."""
    N, M = V.shape
    assoc = greedy_init(cell_demands, V, J)
    loads = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)
    for m in range(M):
        loads[assoc[m]] += cell_demands[m]
        counts[assoc[m]] += 1

    for _ in range(200):
        max_n = int(np.argmax(loads))
        cells_max = np.where(assoc == max_n)[0]
        best_gain = 0.0
        best_move = None
        for m in cells_max:
            dm = cell_demands[m]
            cands = np.where((V[:, m] == 1) & (counts < J))[0]
            for dst in cands:
                if dst == max_n:
                    continue
                dE = (loads[max_n] - dm) ** 2 + (loads[dst] + dm) ** 2 - loads[max_n] ** 2 - loads[dst] ** 2
                if dE < best_gain:
                    best_gain = dE
                    best_move = (m, max_n, int(dst))
        if best_move is None:
            break
        m, src, dst = best_move
        dm = cell_demands[m]
        assoc[m] = dst
        loads[src] -= dm
        loads[dst] += dm
        counts[src] -= 1
        counts[dst] += 1

    Z = np.zeros((N, M), dtype=np.int32)
    Z[assoc, np.arange(M)] = 1
    return Z
