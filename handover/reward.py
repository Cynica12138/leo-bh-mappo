# -*- coding: utf-8 -*-
"""Per-cell local traffic reward and O(1) analytical counterfactual cooperation reward."""

import numpy as np

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv, EnvStepOutput


def compute_step_rewards(
    prev_out: EnvStepOutput,
    step_out: EnvStepOutput,
    env: DynamicLEOHandoverEnv,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell (total_reward, local_reward, coop_reward) for macro step t.

    Local reward:
      r_m^local = w_1 * R_tilde_m - w_2 * P_RT_drop_m - w_3 * H_m * C_HO_m - w_4 * F_m
      (Ping-pong adds an extra handover penalty to discourage oscillation.)

    O(1) Counterfactual Cooperation Reward (Negative Mean-Squared Load + Slot-Balance Potential):
      G(rho, c) = -(1 / N_t) * sum_n [ rho_n^2 + 0.25 * (c_n / J)^2 ]
      r_m^coop  = G(a) - G(a_m^ref, a_{-m})
      Computed in O(1) per cell via incremental updates to old_sat and new_sat only.
    """
    cfg = env.cfg
    adm = step_out.admission
    srv = step_out.service
    assert adm is not None and srv is not None

    m_cnt = cfg.M
    tot_bps, _, _ = env.traffic_model.get_offered_rates_bps(prev_out.t)

    # 1. Local business-aware reward (no double-penalizing RT drops!)
    tput_sat = np.where(
        tot_bps > 1e3,
        np.clip(srv.cell_delivered_bits / np.maximum(tot_bps * cfg.T_H, 1e3), 0.0, 1.2),
        np.where(adm.outage_flags == 0, 1.0, 0.0),
    )
    ho_pen = adm.handover_flags.astype(np.float64) * step_out.cell_ho_cost * (
        1.0 + 2.5 * step_out.ping_pong_flags.astype(np.float64)
    )
    rej_flags = np.maximum(adm.rejected_proactive_flags, adm.rejected_mandatory_flags).astype(np.float64)
    r_local = (
        cfg.w_tput * (tput_sat - 0.85)
        - cfg.w_rt_drop * srv.cell_rt_drop_rate
        - cfg.w_ho_cost * ho_pen
        - cfg.w_outage * adm.outage_flags.astype(np.float64)
        - cfg.w_reject * rej_flags
    )

    # 2. O(1) Analytical Counterfactual Cooperation Reward on post-admission topology (prev_out.snap)
    snap = prev_out.snap
    n_act = len(snap.active_sat_ids)
    r_coop = np.zeros(m_cnt, dtype=np.float64)
    if n_act > 0:
        Z_post = adm.Z_new
        counts = Z_post.sum(axis=1).astype(np.float64)
        k_max = float(cfg.K_MAX)
        j_cap = float(cfg.J)
        d_nm = prev_out.d_nm  # (N_act, M) evaluated on the decision topology

        rho = np.zeros(n_act, dtype=np.float64)
        for n in range(n_act):
            ms = np.flatnonzero(Z_post[n] == 1)
            if len(ms) > 0:
                rho[n] = float(d_nm[n, ms].sum()) / k_max

        slot_w = 1.2
        barrier_w = 0.35
        barrier_start = max(0.0, j_cap - 2.0)

        # Barrier-convex potential G(a) = -(1/N_t) * sum_n [rho_n^2 + slot_w*(c_n/J)^2 + barrier_w*max(0, c_n-(J-2))^2]
        # Incentivizes vacating 14/14 saturated satellites while keeping Stay exact counterfactual zero.
        def _term(r_val: float, c_val: float) -> float:
            excess = max(0.0, c_val - barrier_start)
            return r_val * r_val + slot_w * ((c_val / j_cap) ** 2) + barrier_w * (excess * excess)

        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
        inv_n = 1.0 / float(n_act)
        mean_term = float(
            np.mean(
                rho * rho
                + slot_w * ((counts / j_cap) ** 2)
                + barrier_w * (np.maximum(0.0, counts - barrier_start) ** 2)
            )
        )

        for m in range(m_cnt):
            new_sid = int(adm.new_cell_sat_id[m])
            old_sid = int(prev_out.cell_sat_id[m])
            new_loc = sat_id_to_local.get(new_sid, -1) if new_sid >= 0 else -1
            old_loc = sat_id_to_local.get(old_sid, -1) if old_sid >= 0 else -1
            old_vis = (old_loc >= 0) and (snap.V_geo[old_loc, m] == 1)

            if new_loc < 0:
                # Outage already penalized heavily by w_outage
                r_coop[m] = 0.0
                continue

            dm_new = float(d_nm[new_loc, m]) / k_max
            if old_vis:
                if old_loc == new_loc:
                    # Reference action is Stay == actual action -> counterfactual diff = 0
                    r_coop[m] = 0.0
                else:
                    # O(1) delta between actual (m on new_loc) and reference (m stays on old_loc)
                    dm_old = float(d_nm[old_loc, m]) / k_max
                    cur_pair = _term(rho[new_loc], counts[new_loc]) + _term(rho[old_loc], counts[old_loc])
                    ref_pair = (
                        _term(max(0.0, rho[new_loc] - dm_new), max(0.0, counts[new_loc] - 1.0))
                        + _term(rho[old_loc] + dm_old, counts[old_loc] + 1.0)
                    )
                    # G(actual) - G(ref) = (ref_pair - cur_pair) / N_t
                    r_coop[m] = (ref_pair - cur_pair) * inv_n
            else:
                # Mandatory handover (old sat unavailable): compare placing m on new_loc vs average active sat load
                marginal_cost = _term(rho[new_loc], counts[new_loc]) - _term(
                    max(0.0, rho[new_loc] - dm_new), max(0.0, counts[new_loc] - 1.0)
                )
                # Positive if placed on a lighter-than-average satellite, negative if piled onto an overloaded one
                ref_marginal = 2.0 * math.sqrt(max(mean_term, 1e-6)) * dm_new + slot_w * (2.0 / j_cap)
                r_coop[m] = (ref_marginal - marginal_cost) * inv_n

        # Scale counterfactual delta (divided by N_t ~ 15) to match local reward magnitude cleanly
        r_coop = np.clip(r_coop * float(n_act) * 1.0, -2.5, 2.5)

    r_total = r_local + cfg.eta_coop * r_coop
    return r_total.astype(np.float32), r_local.astype(np.float32), r_coop.astype(np.float32)


import math  # noqa: E402
