# -*- coding: utf-8 -*-
"""Spatially heterogeneous multi-hotspot traffic model with differentiated RT share and optional hotspot drift."""

from dataclasses import dataclass
import numpy as np

from handover.config import HandoverConfig
from handover.geometry import GroundCellLayout, place_users_on_sphere


@dataclass
class TrafficSnapshot:
    """User placement and per-cell traffic profiles for an episode."""

    U: int
    user_cell: np.ndarray           # (U,) int32 cell index in 0..M-1
    user_enu: np.ndarray            # (U, 2) m
    user_ecef: np.ndarray           # (U, 3) m
    cell_users_arr: list[np.ndarray]
    cell_user_counts: np.ndarray    # (M,) int32
    cell_rt_frac: np.ndarray        # (M,) float64 in [rt_frac_min, rt_frac_max], traffic-weighted mean == R_RT_FRAC
    user_rt_frac: np.ndarray        # (U,) float64
    base_user_rate_bps: np.ndarray  # (U,) float64
    base_cell_rate_bps: np.ndarray  # (M,) float64
    drift_cell_masks: list[np.ndarray]  # per drift event boolean mask (M,)


class DynamicTrafficModel:
    """Generates non-uniform hotspot user distributions, differentiated RT shares, and macro hotspot drift."""

    def __init__(self, cfg: HandoverConfig, layout: GroundCellLayout):
        self.cfg = cfg
        self.layout = layout
        self.state: TrafficSnapshot | None = None

    def reset(
        self,
        rng: np.random.Generator,
        num_users: int | None = None,
        input_rate_mbps: float | None = None,
    ) -> TrafficSnapshot:
        u_cnt = int(self.cfg.num_users if num_users is None else num_users)
        rate_mbps = float(self.cfg.input_rate_mbps if input_rate_mbps is None else input_rate_mbps)
        m_cnt = self.cfg.M
        enu = self.layout.cell_pos_enu  # (M, 2)

        # 1. Spatially non-uniform multi-hotspot user density
        span = float(np.max(np.abs(enu))) * 0.78
        centers = rng.uniform(-span, span, size=(self.cfg.num_hotspots, 2))
        d2 = np.sum((enu[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        field = np.sum(np.exp(-d2 / (2.0 * (self.cfg.hotspot_sigma_m ** 2))), axis=1)
        field_norm = field / max(float(field.max()), 1e-9)

        # Background + hotspot boost (~25% hotspot cells gather ~55-60% users)
        weights = 0.25 + self.cfg.hotspot_weight_boost * (field_norm ** 1.6)
        weights = weights * rng.gamma(shape=2.5, scale=0.4, size=m_cnt)
        weights /= weights.sum()

        # Assign users to cells
        user_cell = rng.choice(m_cnt, size=u_cnt, p=weights).astype(np.int32)
        user_enu, user_ecef = place_users_on_sphere(self.layout, user_cell, self.cfg.CELL_RADIUS_M, rng)
        cell_users_arr = [np.flatnonzero(user_cell == m).astype(np.int32) for m in range(m_cnt)]
        cell_user_counts = np.array([len(arr) for arr in cell_users_arr], dtype=np.int32)

        # 2. Spatially differentiated RT share across cells (preserving network-wide mean == R_RT_FRAC)
        # Correlate partially with hotspot clusters + independent cell variation so RT urgency varies meaningfully
        rt_raw = self.cfg.rt_frac_min + (self.cfg.rt_frac_max - self.cfg.rt_frac_min) * (
            0.55 * field_norm + 0.45 * rng.uniform(0.0, 1.0, size=m_cnt)
        )
        # Calibrate traffic-weighted network mean RT fraction to cfg.R_RT_FRAC (0.20)
        w_cell = np.maximum(cell_user_counts.astype(np.float64), 0.0)
        if w_cell.sum() > 0:
            cur_mean = float(np.dot(w_cell, rt_raw) / w_cell.sum())
            if cur_mean > 1e-6:
                rt_cal = np.clip(rt_raw * (self.cfg.R_RT_FRAC / cur_mean), self.cfg.rt_frac_min, self.cfg.rt_frac_max)
            else:
                rt_cal = np.full(m_cnt, self.cfg.R_RT_FRAC, dtype=np.float64)
        else:
            rt_cal = np.full(m_cnt, self.cfg.R_RT_FRAC, dtype=np.float64)

        user_rt_frac = rt_cal[user_cell]
        base_user_rate_bps = np.full(u_cnt, rate_mbps * 1e6, dtype=np.float64)
        base_cell_rate_bps = cell_user_counts.astype(np.float64) * (rate_mbps * 1e6)

        # 3. Optional macro hotspot drift target masks (distinct clusters activated at drift_times_s)
        drift_masks: list[np.ndarray] = []
        for _ in self.cfg.drift_times_s:
            dc = rng.uniform(-span, span, size=(2, 2))
            dd = np.min(np.sum((enu[:, None, :] - dc[None, :, :]) ** 2, axis=-1), axis=1)
            thresh = np.percentile(dd, 22)
            drift_masks.append(dd <= thresh)

        self.state = TrafficSnapshot(
            U=u_cnt,
            user_cell=user_cell,
            user_enu=user_enu,
            user_ecef=user_ecef,
            cell_users_arr=cell_users_arr,
            cell_user_counts=cell_user_counts,
            cell_rt_frac=rt_cal,
            user_rt_frac=user_rt_frac,
            base_user_rate_bps=base_user_rate_bps,
            base_cell_rate_bps=base_cell_rate_bps,
            drift_cell_masks=drift_masks,
        )
        return self.state

    def cell_rate_multiplier(self, t: int) -> np.ndarray:
        """Return per-cell offered-rate multiplier (M,) at macro step t (1.0 unless hotspot drift is active)."""
        mult = np.ones(self.cfg.M, dtype=np.float64)
        if not self.cfg.enable_hotspot_drift or self.state is None:
            return mult
        for start_t, mask in zip(self.cfg.drift_times_s, self.state.drift_cell_masks):
            if start_t <= t < start_t + self.cfg.drift_burst_duration_s:
                mult[mask] *= self.cfg.drift_rate_mult
        return mult

    def get_offered_rates_bps(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cell_tot_bps, cell_rt_bps, cell_bulk_bps) at macro step t."""
        if self.state is None:
            raise RuntimeError("Call reset() before get_offered_rates_bps().")
        mult = self.cell_rate_multiplier(t)
        tot = self.state.base_cell_rate_bps * mult
        rt = tot * self.state.cell_rt_frac
        bulk = tot - rt
        return tot, rt, bulk
