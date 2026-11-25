# -*- coding: utf-8 -*-
"""Dual-stage CHO-inspired adaptive handover trigger (mandatory coverage + hysteretic proactive risk)."""

from dataclasses import dataclass
import numpy as np

from handover.config import HandoverConfig
from handover.geometry import TopologySnapshot


@dataclass
class TriggerOutput:
    """Per-cell trigger indicators and action feasibility flags at macro step t."""

    i_force: np.ndarray        # (M,) int32 {0, 1} mandatory trigger
    i_pro: np.ndarray          # (M,) int32 {0, 1} proactive traffic/link trigger
    i_trig: np.ndarray         # (M,) int32 {0, 1} max(i_force, i_pro)
    stay_allowed: np.ndarray   # (M,) bool whether keeping current satellite is physically valid
    dwell_s: np.ndarray        # (M,) int32 current dwell time on serving satellite (s)


class DualStageTrigger:
    """CHO-inspired dual-stage handover trigger with dwell timer and L_TTT hysteresis."""

    def __init__(self, cfg: HandoverConfig, force_only: bool = False, traffic_aware: bool = True):
        self.cfg = cfg
        self.force_only = force_only
        self.traffic_aware = traffic_aware
        self.dwell_s = np.zeros(cfg.M, dtype=np.int32)
        self.proactive_streak = np.zeros(cfg.M, dtype=np.int32)
        self.proactive_cooldown = np.zeros(cfg.M, dtype=np.int32)

    def reset(self) -> None:
        self.dwell_s.fill(0)
        self.proactive_streak.fill(0)
        self.proactive_cooldown.fill(0)

    def update_dwell(self, prev_sat_id: np.ndarray, new_sat_id: np.ndarray) -> None:
        """Advance dwell counters and decay proactive cooldown after handover execution."""
        stayed = (new_sat_id >= 0) & (new_sat_id == prev_sat_id)
        connected = new_sat_id >= 0
        self.dwell_s = np.where(stayed, self.dwell_s + 1, np.where(connected, 1, 0)).astype(np.int32)
        # Reset proactive hysteresis streak & cooldown upon handover or outage
        self.proactive_streak = np.where(stayed, self.proactive_streak, 0).astype(np.int32)
        self.proactive_cooldown = np.where(stayed, np.maximum(0, self.proactive_cooldown - 1), 0).astype(np.int32)

    def evaluate(
        self,
        snap: TopologySnapshot,
        cell_sat_id: np.ndarray,
        nominal_sinr_db: np.ndarray,
        sat_rho: np.ndarray,
        sat_zeta: np.ndarray,
        cell_rt_urg: np.ndarray,
        cell_recent_rt_drop: np.ndarray,
        sat_counts: np.ndarray | None = None,
    ) -> TriggerOutput:
        """Compute mandatory and proactive triggers for all M=144 cells."""
        m_cnt = self.cfg.M
        sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}

        i_force = np.zeros(m_cnt, dtype=np.int32)
        i_pro = np.zeros(m_cnt, dtype=np.int32)
        stay_allowed = np.zeros(m_cnt, dtype=bool)
        sinr_trig_th = self.cfg.sinr_th_db - self.cfg.sinr_hys_db

        for m in range(m_cnt):
            sid = int(cell_sat_id[m])
            cur_local = sat_id_to_local.get(sid, -1) if sid >= 0 else -1
            is_vis = (cur_local >= 0) and (snap.V_geo[cur_local, m] == 1)
            stay_allowed[m] = is_vis

            if not is_vis:
                i_force[m] = 1
                self.proactive_streak[m] = 0
                continue

            if snap.rvt_s[cur_local, m] <= self.cfg.T_cov_s:
                i_force[m] = 1
                self.proactive_streak[m] = 0
                continue

            if self.force_only:
                self.proactive_streak[m] = 0
                continue

            link_risk = float(nominal_sinr_db[cur_local, m]) < sinr_trig_th
            if self.traffic_aware:
                slot_pressure = (sat_counts is not None) and (int(sat_counts[cur_local]) >= self.cfg.J)
                traffic_risk = (
                    slot_pressure
                    or (float(sat_rho[cur_local]) > self.cfg.rho_th)
                    or (float(sat_zeta[cur_local]) > self.cfg.zeta_th)
                    or (float(cell_rt_urg[m]) > self.cfg.rt_urg_th)
                    or (float(cell_recent_rt_drop[m]) > self.cfg.rt_drop_trig_th)
                )
            else:
                traffic_risk = False

            if link_risk or traffic_risk:
                self.proactive_streak[m] += 1
            else:
                self.proactive_streak[m] = 0

            if (
                self.dwell_s[m] >= self.cfg.T_dwell_min_s
                and self.proactive_streak[m] >= self.cfg.L_ttt_s
                and self.proactive_cooldown[m] == 0
            ):
                i_pro[m] = 1
                self.proactive_streak[m] = 0
                self.proactive_cooldown[m] = self.cfg.proactive_cooldown_s

        i_trig = np.maximum(i_force, i_pro).astype(np.int32)
        return TriggerOutput(
            i_force=i_force,
            i_pro=i_pro,
            i_trig=i_trig,
            stay_allowed=stay_allowed,
            dwell_s=self.dwell_s.copy(),
        )
