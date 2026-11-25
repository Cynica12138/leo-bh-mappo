# -*- coding: utf-8 -*-
"""Vectorized local multi-dimensional heterogeneous graph observation builder for triggered ground cells."""

from dataclasses import dataclass
import math
import numpy as np

from handover.dynamic_env import DynamicLEOHandoverEnv, EnvStepOutput

SAT_FEAT_DIM = 7
CELL_FEAT_DIM = 7
COV_EDGE_DIM = 6
SERV_EDGE_DIM = 3
COMP_EDGE_DIM = 2


@dataclass
class CellGraphObs:
    """Fixed-shape dense local heterogeneous graph observation for a single cell m."""

    cell_id: int
    cand_local_sats: np.ndarray  # (C_MAX,) int32 local satellite indices (-1 padded)
    cur_sat_local: int           # local index of currently serving visible sat (-1 if none)
    cur_sat_cand_pos: int        # position in 0..C_MAX-1 if cur_sat in cand_local_sats else -1
    x_sat: np.ndarray            # (C_MAX, SAT_FEAT_DIM) float32
    sat_mask: np.ndarray         # (C_MAX,) float32 {0, 1}
    x_cell: np.ndarray           # (1 + M_nbr, CELL_FEAT_DIM) float32 (idx 0 is self)
    cell_mask: np.ndarray        # (1 + M_nbr,) float32 {0, 1}
    e_cov: np.ndarray            # (C_MAX, 1 + M_nbr, COV_EDGE_DIM) float32
    m_cov: np.ndarray            # (C_MAX, 1 + M_nbr) float32
    e_serv: np.ndarray           # (C_MAX, 1 + M_nbr, SERV_EDGE_DIM) float32
    m_serv: np.ndarray           # (C_MAX, 1 + M_nbr) float32
    e_comp: np.ndarray           # (1 + M_nbr, 1 + M_nbr, COMP_EDGE_DIM) float32
    m_comp: np.ndarray           # (1 + M_nbr, 1 + M_nbr) float32
    action_mask: np.ndarray      # (1 + C_MAX,) float32 (idx 0 = Stay, 1..C_MAX = switch to cand c)
    rt_frac: float               # p_m^RT(t)
    rt_urg: float                # u_m^RT(t)
    is_mandatory: bool

    def copy(self) -> "CellGraphObs":
        return CellGraphObs(
            cell_id=self.cell_id,
            cand_local_sats=self.cand_local_sats.copy(),
            cur_sat_local=self.cur_sat_local,
            cur_sat_cand_pos=self.cur_sat_cand_pos,
            x_sat=self.x_sat.copy(),
            sat_mask=self.sat_mask.copy(),
            x_cell=self.x_cell.copy(),
            cell_mask=self.cell_mask.copy(),
            e_cov=self.e_cov.copy(),
            m_cov=self.m_cov.copy(),
            e_serv=self.e_serv.copy(),
            m_serv=self.m_serv.copy(),
            e_comp=self.e_comp.copy(),
            m_comp=self.m_comp.copy(),
            action_mask=self.action_mask.copy(),
            rt_frac=self.rt_frac,
            rt_urg=self.rt_urg,
            is_mandatory=self.is_mandatory,
        )


def build_cell_graph_obs_batch(
    cell_ids: list[int] | np.ndarray,
    out: EnvStepOutput,
    env: DynamicLEOHandoverEnv,
    traffic_aware: bool = True,
) -> list[CellGraphObs]:
    """Vectorized step-shared builder producing CellGraphObs for all requested cells at once."""
    if len(cell_ids) == 0:
        return []

    cfg = env.cfg
    snap = out.snap
    trig = out.trigger
    c_max = cfg.C_MAX
    m_nbr = cfg.M_nbr
    n_cells_loc = 1 + m_nbr
    n_act, m_cnt = snap.V_geo.shape
    half_pi = math.pi / 2.0
    assert env.traffic is not None

    # --- Precompute step-level matrices ONCE across all M=144 cells ---
    sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
    cur_cell_local = np.array(
        [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
        dtype=np.int32,
    )
    # Mask out serving sat if no longer geometrically visible
    valid_cur = cur_cell_local >= 0
    if np.any(valid_cur):
        vc_idx = np.flatnonzero(valid_cur)
        still_vis = snap.V_geo[cur_cell_local[vc_idx], vc_idx] == 1
        cur_cell_local[vc_idx[~still_vis]] = -1

    tot_bps, rt_bps, _ = env.traffic_model.get_offered_rates_bps(out.t)
    tot_bps_norm = np.minimum(tot_bps / 150e6, 2.5).astype(np.float32)

    sat_counts = out.Z.sum(axis=1).astype(np.float32)
    z_f = out.Z.astype(np.float64)
    sat_tot_bps = z_f @ tot_bps
    sat_rt_bps = z_f @ rt_bps
    sat_rt_share = np.where(sat_tot_bps > 1e-3, sat_rt_bps / np.maximum(sat_tot_bps, 1e-3), cfg.R_RT_FRAC).astype(np.float32)

    # Full-network Jaccard candidate overlap matrix (M, M) via single BLAS matmul!
    v_f = snap.V_geo.astype(np.float32)
    inter_mat = v_f.T @ v_f                                        # (M, M)
    deg = v_f.sum(axis=0)                                          # (M,)
    union_mat = np.maximum(deg[:, None] + deg[None, :] - inter_mat, 1.0)
    jacc_mat = inter_mat / union_mat                               # (M, M)
    np.fill_diagonal(jacc_mat, 0.0)

    # Neighbor selection priority weight per cell (M,)
    nbr_w = (1.0 + 1.5 * trig.i_trig.astype(np.float32) + (tot_bps_norm if traffic_aware else 0.0))

    # Precompute global cell node feature table (M, 7) (col 6 is_self filled per query)
    all_cell_feat = np.zeros((m_cnt, CELL_FEAT_DIM), dtype=np.float32)
    cur_rvt_all = np.zeros(m_cnt, dtype=np.float32)
    vc_idx = np.flatnonzero(cur_cell_local >= 0)
    if len(vc_idx) > 0:
        cur_rvt_all[vc_idx] = np.clip(snap.rvt_s[cur_cell_local[vc_idx], vc_idx] / 150.0, 0.0, 1.5)
    all_cell_feat[:, 4] = cur_rvt_all
    all_cell_feat[:, 5] = trig.i_trig.astype(np.float32)
    if traffic_aware:
        all_cell_feat[:, 0] = np.minimum(env.traffic.cell_user_counts.astype(np.float32) / 8.0, 2.0)
        all_cell_feat[:, 1] = tot_bps_norm
        all_cell_feat[:, 2] = env.traffic.cell_rt_frac.astype(np.float32)
        all_cell_feat[:, 3] = env.cell_rt_urg.astype(np.float32)

    # Precompute global edge feature tables (N_act, M, 6) and per-cell service features (M, 3)
    cov_table = np.zeros((n_act, m_cnt, COV_EDGE_DIM), dtype=np.float32)
    if n_act > 0:
        cov_table[:, :, 0] = np.clip(snap.rvt_s / 150.0, 0.0, 1.5)
        cov_table[:, :, 1] = snap.elev_rad / half_pi
        cov_table[:, :, 2] = np.clip(snap.elev_rate_rad * 100.0, -2.0, 2.0)
        cov_table[:, :, 3] = np.clip((out.links["nominal_sinr_db"] - 8.0) / 15.0, -2.0, 2.0)
        cov_table[:, :, 4] = np.clip(out.links["nominal_rate_bps"] / 350e6, 0.0, 2.0)
        if traffic_aware:
            cov_table[:, :, 5] = np.clip(out.d_nm / 2.5, 0.0, 2.0)

    serv_cell_feat = np.zeros((m_cnt, SERV_EDGE_DIM), dtype=np.float32)
    serv_cell_feat[:, 0] = np.minimum(trig.dwell_s.astype(np.float32) / 60.0, 2.0)
    if traffic_aware:
        serv_cell_feat[:, 1] = np.clip(env.cell_hist_tput_ratio, 0.0, 1.5)
        serv_cell_feat[:, 2] = np.clip(env.cell_recent_rt_drop, 0.0, 1.0)

    sat_rho_norm = np.clip(out.sat_rho / 2.5, 0.0, 2.0).astype(np.float32)
    sat_zeta_norm = np.clip(out.sat_zeta / 2.5, 0.0, 2.0).astype(np.float32)
    sat_cnt_norm = (sat_counts / float(cfg.J)).astype(np.float32)

    obs_list: list[CellGraphObs] = []
    for raw_m in cell_ids:
        cell_id = int(raw_m)
        cur_local = int(cur_cell_local[cell_id])
        cand_sats = snap.cand_sats_per_cell[cell_id].copy()
        valid_c = np.flatnonzero(cand_sats >= 0)
        n_valid_c = len(valid_c)
        c_sats_v = cand_sats[valid_c]

        cur_sat_cand_pos = -1
        if cur_local >= 0 and n_valid_c > 0:
            matches = np.flatnonzero(c_sats_v == cur_local)
            if len(matches) > 0:
                cur_sat_cand_pos = int(valid_c[matches[0]])

        # Select top-M_nbr competing neighbors using precomputed Jaccard row
        scores = jacc_mat[cell_id] * nbr_w
        pos_nbr = np.flatnonzero(scores > 0.0)
        if len(pos_nbr) > m_nbr:
            top_sub = np.argpartition(-scores[pos_nbr], m_nbr)[:m_nbr]
            top_nbr = pos_nbr[top_sub[np.argsort(-scores[pos_nbr[top_sub]])]]
        elif len(pos_nbr) > 0:
            top_nbr = pos_nbr[np.argsort(-scores[pos_nbr])]
        else:
            top_nbr = np.empty(0, dtype=np.int32)

        nbr_arr = np.concatenate([[cell_id], top_nbr]).astype(np.int32)
        k_loc = len(nbr_arr)

        # Satellite node features (C_MAX, 7)
        x_sat = np.zeros((c_max, SAT_FEAT_DIM), dtype=np.float32)
        sat_mask = np.zeros(c_max, dtype=np.float32)
        if n_valid_c > 0:
            sat_mask[valid_c] = 1.0
            x_sat[valid_c, 0] = cov_table[c_sats_v, cell_id, 1]
            x_sat[valid_c, 1] = cov_table[c_sats_v, cell_id, 2]
            x_sat[valid_c, 2] = cov_table[c_sats_v, cell_id, 0]
            x_sat[valid_c, 5] = sat_cnt_norm[c_sats_v]
            if traffic_aware:
                x_sat[valid_c, 3] = sat_rho_norm[c_sats_v]
                x_sat[valid_c, 4] = sat_zeta_norm[c_sats_v]
                x_sat[valid_c, 6] = sat_rt_share[c_sats_v]

        # Cell node features (1 + M_nbr, 7)
        x_cell = np.zeros((n_cells_loc, CELL_FEAT_DIM), dtype=np.float32)
        cell_mask = np.zeros(n_cells_loc, dtype=np.float32)
        cell_mask[:k_loc] = 1.0
        x_cell[:k_loc] = all_cell_feat[nbr_arr]
        x_cell[0, 6] = 1.0

        # Directed edge feature tensors via pure NumPy fancy indexing
        e_cov = np.zeros((c_max, n_cells_loc, COV_EDGE_DIM), dtype=np.float32)
        m_cov = np.zeros((c_max, n_cells_loc), dtype=np.float32)
        e_serv = np.zeros((c_max, n_cells_loc, SERV_EDGE_DIM), dtype=np.float32)
        m_serv = np.zeros((c_max, n_cells_loc), dtype=np.float32)

        if n_valid_c > 0:
            ix_sn = np.ix_(c_sats_v, nbr_arr)
            v_sub = snap.V_geo[ix_sn].astype(np.float32)  # (n_valid_c, k_loc)
            m_cov[np.ix_(valid_c, np.arange(k_loc))] = v_sub
            e_cov[np.ix_(valid_c, np.arange(k_loc))] = cov_table[ix_sn] * v_sub[:, :, None]

            serv_match = (c_sats_v[:, None] == cur_cell_local[nbr_arr][None, :]).astype(np.float32)
            m_serv[np.ix_(valid_c, np.arange(k_loc))] = serv_match
            e_serv[np.ix_(valid_c, np.arange(k_loc))] = serv_match[:, :, None] * serv_cell_feat[nbr_arr][None, :, :]

        e_comp = np.zeros((n_cells_loc, n_cells_loc, COMP_EDGE_DIM), dtype=np.float32)
        m_comp = np.zeros((n_cells_loc, n_cells_loc), dtype=np.float32)
        j_sub = jacc_mat[np.ix_(nbr_arr, nbr_arr)]
        m_c_sub = (j_sub > 0.0).astype(np.float32)
        m_comp[:k_loc, :k_loc] = m_c_sub
        e_comp[:k_loc, :k_loc, 0] = j_sub
        if traffic_aware:
            e_comp[:k_loc, :k_loc, 1] = j_sub * tot_bps_norm[nbr_arr][None, :]

        # Dynamic action mask: action 0 = Stay, action 1+c = switch to cand_sats[c]
        action_mask = np.zeros(1 + c_max, dtype=np.float32)
        has_other_cand = False
        if n_valid_c > 0:
            other = c_sats_v != cur_local
            if np.any(other):
                action_mask[1 + valid_c[other]] = 1.0
                has_other_cand = True

        stay_ok = bool(trig.stay_allowed[cell_id])
        if trig.i_force[cell_id] == 1 and cur_local >= 0 and snap.rvt_s[cur_local, cell_id] <= cfg.T_cov_s and has_other_cand:
            stay_ok = False
        if stay_ok or (not has_other_cand and trig.stay_allowed[cell_id]):
            action_mask[0] = 1.0
        if action_mask.sum() < 0.5:
            action_mask[0] = 1.0

        obs_list.append(
            CellGraphObs(
                cell_id=cell_id,
                cand_local_sats=cand_sats,
                cur_sat_local=cur_local,
                cur_sat_cand_pos=cur_sat_cand_pos,
                x_sat=x_sat,
                sat_mask=sat_mask,
                x_cell=x_cell,
                cell_mask=cell_mask,
                e_cov=e_cov,
                m_cov=m_cov,
                e_serv=e_serv,
                m_serv=m_serv,
                e_comp=e_comp,
                m_comp=m_comp,
                action_mask=action_mask,
                rt_frac=float(env.traffic.cell_rt_frac[cell_id]),
                rt_urg=float(env.cell_rt_urg[cell_id]),
                is_mandatory=bool(trig.i_force[cell_id] > 0),
            )
        )
    return obs_list


def build_cell_graph_obs(
    cell_id: int,
    out: EnvStepOutput,
    env: DynamicLEOHandoverEnv,
    traffic_aware: bool = True,
) -> CellGraphObs:
    """Single-cell convenience wrapper around build_cell_graph_obs_batch."""
    return build_cell_graph_obs_batch([cell_id], out, env, traffic_aware=traffic_aware)[0]


def collate_cell_graphs(obs_list: list[CellGraphObs]) -> dict[str, np.ndarray]:
    """Stack a batch of CellGraphObs into contiguous numpy arrays for fast batched PyTorch forward."""
    return {
        "x_sat": np.stack([o.x_sat for o in obs_list], axis=0),
        "sat_mask": np.stack([o.sat_mask for o in obs_list], axis=0),
        "x_cell": np.stack([o.x_cell for o in obs_list], axis=0),
        "cell_mask": np.stack([o.cell_mask for o in obs_list], axis=0),
        "e_cov": np.stack([o.e_cov for o in obs_list], axis=0),
        "m_cov": np.stack([o.m_cov for o in obs_list], axis=0),
        "e_serv": np.stack([o.e_serv for o in obs_list], axis=0),
        "m_serv": np.stack([o.m_serv for o in obs_list], axis=0),
        "e_comp": np.stack([o.e_comp for o in obs_list], axis=0),
        "m_comp": np.stack([o.m_comp for o in obs_list], axis=0),
        "action_mask": np.stack([o.action_mask for o in obs_list], axis=0),
        "cur_sat_cand_pos": np.array([o.cur_sat_cand_pos for o in obs_list], dtype=np.int64),
        "rt_frac": np.array([o.rt_frac for o in obs_list], dtype=np.float32),
        "rt_urg": np.array([o.rt_urg for o in obs_list], dtype=np.float32),
    }
