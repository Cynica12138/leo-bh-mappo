# -*- coding: utf-8 -*-
"""Spherical ground cell layout (paper-simulation 16x3x3 clusters), visibility V_geo, and exact RVT."""

from dataclasses import dataclass
import math
import numpy as np

from handover.config import EARTH_RADIUS_M, HandoverConfig
from handover.orbit import WalkerConstellation


@dataclass
class GroundCellLayout:
    """144 Earth-fixed ground cells mapped onto the Earth sphere in ECEF."""

    cell_pos_enu: np.ndarray     # (M, 2) local east/north offsets (m)
    cell_pos_ecef: np.ndarray    # (M, 3) ECEF coordinates on Earth sphere (m)
    cell_up: np.ndarray          # (M, 3) outward unit normal vectors
    center_ecef: np.ndarray      # (3,) region center ECEF
    east_vec: np.ndarray         # (3,) local East unit vector at region center
    north_vec: np.ndarray        # (3,) local North unit vector at region center
    up_vec: np.ndarray           # (3,) local Up unit vector at region center


def build_ground_cells(
    lat0_deg: float,
    lon0_deg: float,
    cell_spacing_m: float,
    sat_spacing_m: float,
) -> GroundCellLayout:
    """Build paper-simulation's 144 fixed cells (16 local 3x3 multibeam clusters) on the Earth sphere."""
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    up0 = np.array(
        [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)],
        dtype=np.float64,
    )
    east0 = np.array([-math.sin(lon0), math.cos(lon0), 0.0], dtype=np.float64)
    north0 = np.array(
        [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)],
        dtype=np.float64,
    )
    center_ecef = EARTH_RADIUS_M * up0

    sx = (np.arange(4, dtype=np.float64) - 1.5) * sat_spacing_m
    sy = (np.arange(4, dtype=np.float64) - 1.5) * sat_spacing_m
    enu = np.zeros((144, 2), dtype=np.float64)
    idx_m = 0
    for ix in range(4):
        for iy in range(4):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    enu[idx_m, 0] = sx[ix] + dx * cell_spacing_m
                    enu[idx_m, 1] = sy[iy] + dy * cell_spacing_m
                    idx_m += 1

    raw = center_ecef[None, :] + enu[:, 0:1] * east0[None, :] + enu[:, 1:2] * north0[None, :]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cell_up = raw / norms
    cell_pos_ecef = cell_up * EARTH_RADIUS_M
    return GroundCellLayout(
        cell_pos_enu=enu,
        cell_pos_ecef=cell_pos_ecef,
        cell_up=cell_up,
        center_ecef=center_ecef,
        east_vec=east0,
        north_vec=north0,
        up_vec=up0,
    )


def place_users_on_sphere(
    layout: GroundCellLayout,
    user_cell: np.ndarray,
    cell_radius_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Place U users uniformly inside their cell's disk (radius cell_radius_m) and project onto sphere."""
    u_count = len(user_cell)
    ang = rng.uniform(0.0, 2.0 * math.pi, size=u_count)
    rad = cell_radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=u_count))
    dx = rad * np.cos(ang)
    dy = rad * np.sin(ang)

    user_enu = layout.cell_pos_enu[user_cell] + np.stack([dx, dy], axis=-1)
    raw = (
        layout.center_ecef[None, :]
        + user_enu[:, 0:1] * layout.east_vec[None, :]
        + user_enu[:, 1:2] * layout.north_vec[None, :]
    )
    user_ecef = (raw / np.linalg.norm(raw, axis=1, keepdims=True)) * EARTH_RADIUS_M
    return user_enu, user_ecef


@dataclass
class TopologySnapshot:
    """Dynamic geometry snapshot at macro step t (seconds)."""

    t: int
    active_sat_ids: np.ndarray   # (N_act,) global Walker sat IDs
    sat_pos_ecef: np.ndarray     # (N_act, 3)
    sat_vel_ecef: np.ndarray     # (N_act, 3)
    elev_rad: np.ndarray         # (N_act, M)
    elev_rate_rad: np.ndarray    # (N_act, M) rad/s
    slant_dist_m: np.ndarray     # (N_act, M)
    V_geo: np.ndarray            # (N_act, M) int32 binary strictly elev >= THETA_MIN_DEG
    rvt_s: np.ndarray            # (N_act, M) float32 exact remaining visibility time (s)
    cand_sats_per_cell: np.ndarray  # (M, C_MAX) local index into active_sat_ids, -1 padded


class DynamicGeometry:
    """Precomputed episode-level orbital geometry and exact 1 s RVT lookup engine."""

    def __init__(self, cfg: HandoverConfig):
        self.cfg = cfg
        self.spec = cfg.constellation
        self.constellation = WalkerConstellation(self.spec)
        self.layout = build_ground_cells(
            lat0_deg=self.spec.region_lat_deg,
            lon0_deg=self.spec.region_lon_deg,
            cell_spacing_m=cfg.CELL_SPACING_M,
            sat_spacing_m=cfg.SAT_SPACING_M,
        )
        self.sin_min_elev = math.sin(math.radians(self.spec.min_elev_deg))
        # Broad spatial culling radius around region center (~1800 km covers 541km + 385km diag + 515km alt safely)
        self._cull_radius_m = 2100.0e3

        # Episode precomputed buffers
        self._t0_sec: float = 0.0
        self._duration_s: int = 0
        self._window_sat_ids: np.ndarray | None = None
        self._pos_all: np.ndarray | None = None        # (T_tot, N_win, 3)
        self._vel_all: np.ndarray | None = None        # (T_tot, N_win, 3)
        self._elev_all: np.ndarray | None = None       # (T_tot, N_win, M)
        self._dist_all: np.ndarray | None = None       # (T_tot, N_win, M)
        self._vgeo_all: np.ndarray | None = None       # (T_tot, N_win, M) bool
        self._rvt_all: np.ndarray | None = None        # (T_tot, N_win, M) float32

    def precompute_episode(self, t0_sec: float, duration_s: int) -> None:
        """Precompute positions, elevations, V_geo, and exact RVT for t in [0, duration_s]."""
        self._t0_sec = float(t0_sec)
        self._duration_s = int(duration_s)
        t_tot = self._duration_s + self.cfg.rvt_lookahead_s + 2

        # Stage 1: coarse filter over every 10 s to collect all satellites that EVER approach the region
        seen_mask = np.zeros(self.constellation.total_sats, dtype=bool)
        center = self.layout.center_ecef[None, :]
        sample_ts = list(range(0, t_tot, 10))
        if sample_ts[-1] != t_tot - 1:
            sample_ts.append(t_tot - 1)
        for ts in sample_ts:
            pos_full, _ = self.constellation.propagate(float(ts), self._t0_sec)
            d0 = np.linalg.norm(pos_full - center, axis=-1)
            seen_mask |= d0 < (self._cull_radius_m + 400.0e3)

        win_ids = np.flatnonzero(seen_mask).astype(np.int32)
        n_win = len(win_ids)
        m_cnt = self.cfg.M

        pos_all = np.empty((t_tot, n_win, 3), dtype=np.float64)
        vel_all = np.empty((t_tot, n_win, 3), dtype=np.float64)
        elev_all = np.empty((t_tot, n_win, m_cnt), dtype=np.float32)
        dist_all = np.empty((t_tot, n_win, m_cnt), dtype=np.float32)
        vgeo_all = np.empty((t_tot, n_win, m_cnt), dtype=bool)

        cell_ecef = self.layout.cell_pos_ecef[None, :, :]  # (1, M, 3)
        cell_up = self.layout.cell_up                      # (M, 3)

        for t in range(t_tot):
            pos, vel = self.constellation.propagate_subset(win_ids, float(t), self._t0_sec)
            pos_all[t] = pos
            vel_all[t] = vel
            vec = pos[:, None, :] - cell_ecef              # (N_win, M, 3)
            dist = np.linalg.norm(vec, axis=-1)            # (N_win, M)
            sin_el = np.einsum("nmc,mc->nm", vec, cell_up) / np.maximum(dist, 1.0)
            sin_el = np.clip(sin_el, -1.0, 1.0)
            elev_all[t] = np.arcsin(sin_el).astype(np.float32)
            dist_all[t] = dist.astype(np.float32)
            vgeo_all[t] = sin_el >= self.sin_min_elev

        # Stage 2: exact backward sweep to compute remaining visibility time RVT (s)
        # If visible at t, rvt[t] = 1 + rvt[t+1] (remaining seconds including step t until dropping below 40 deg)
        rvt_all = np.zeros((t_tot, n_win, m_cnt), dtype=np.float32)
        rvt_all[-1] = vgeo_all[-1].astype(np.float32)
        for t in range(t_tot - 2, -1, -1):
            rvt_all[t] = np.where(vgeo_all[t], rvt_all[t + 1] + 1.0, 0.0)

        self._window_sat_ids = win_ids
        self._pos_all = pos_all
        self._vel_all = vel_all
        self._elev_all = elev_all
        self._dist_all = dist_all
        self._vgeo_all = vgeo_all
        self._rvt_all = rvt_all

    def get_snapshot(self, t: int, current_cell_sat_id: np.ndarray | None = None) -> TopologySnapshot:
        """Return TopologySnapshot at integer macro step t in [0, duration_s]."""
        if self._vgeo_all is None:
            raise RuntimeError("Call precompute_episode() before get_snapshot().")
        if t < 0 or t > self._duration_s:
            raise IndexError(f"Step t={t} outside [0, {self._duration_s}]")

        v_t = self._vgeo_all[t]                            # (N_win, M)
        act_mask = v_t.any(axis=1)
        act_idx = np.flatnonzero(act_mask)

        active_sat_ids = self._window_sat_ids[act_idx]
        sat_pos = self._pos_all[t, act_idx]
        sat_vel = self._vel_all[t, act_idx]
        elev = self._elev_all[t, act_idx]
        dist = self._dist_all[t, act_idx]
        v_geo = v_t[act_idx].astype(np.int32)
        rvt = self._rvt_all[t, act_idx]

        # Elevation rate (rad/s)
        if t > 0:
            elev_rate = (self._elev_all[t, act_idx] - self._elev_all[t - 1, act_idx]).astype(np.float32)
        else:
            elev_rate = (self._elev_all[1, act_idx] - self._elev_all[0, act_idx]).astype(np.float32)

        # Build per-cell Top-C_MAX candidate satellites (local indices into active_sat_ids)
        c_max = self.cfg.C_MAX
        m_cnt = self.cfg.M
        cand_sats = np.full((m_cnt, c_max), -1, dtype=np.int32)
        sat_id_to_local = {int(sid): i for i, sid in enumerate(active_sat_ids)}

        for m in range(m_cnt):
            vis_local = np.flatnonzero(v_geo[:, m] == 1)
            if len(vis_local) == 0:
                continue
            # Score by RVT primarily, elevation tie-break
            order = vis_local[np.lexsort((-elev[vis_local, m], -rvt[vis_local, m]))]
            chosen = order[:c_max].tolist()
            # Ensure currently serving visible satellite is always retained in candidate set
            if current_cell_sat_id is not None and current_cell_sat_id[m] >= 0:
                cur_local = sat_id_to_local.get(int(current_cell_sat_id[m]), -1)
                if cur_local >= 0 and v_geo[cur_local, m] == 1 and cur_local not in chosen:
                    if len(chosen) < c_max:
                        chosen.append(cur_local)
                    else:
                        chosen[-1] = cur_local
            cand_sats[m, : len(chosen)] = chosen

        return TopologySnapshot(
            t=t,
            active_sat_ids=active_sat_ids,
            sat_pos_ecef=sat_pos,
            sat_vel_ecef=sat_vel,
            elev_rad=elev,
            elev_rate_rad=elev_rate,
            slant_dist_m=dist,
            V_geo=v_geo,
            rvt_s=rvt,
            cand_sats_per_cell=cand_sats,
        )

    def summarize_episode(self) -> dict:
        """Compute coverage and capacity diagnostics over [0, duration_s]."""
        if self._vgeo_all is None:
            raise RuntimeError("Call precompute_episode() first.")
        v_ep = self._vgeo_all[: self._duration_s + 1]      # (T+1, N_win, M)
        n_act_t = v_ep.any(axis=2).sum(axis=1)             # (T+1,)
        vis_per_cell = v_ep.sum(axis=1)                    # (T+1, M)
        cap_ratio = (n_act_t * self.cfg.J) / float(self.cfg.M)
        rvt_vis = self._rvt_all[: self._duration_s + 1][v_ep]
        return {
            "preset": self.spec.name,
            "duration_s": self._duration_s,
            "active_sats_min": int(n_act_t.min()),
            "active_sats_mean": float(n_act_t.mean()),
            "active_sats_max": int(n_act_t.max()),
            "cap_ratio_min": float(cap_ratio.min()),
            "cap_ratio_mean": float(cap_ratio.mean()),
            "cap_ratio_max": float(cap_ratio.max()),
            "vis_per_cell_min": int(vis_per_cell.min()),
            "vis_per_cell_mean": float(vis_per_cell.mean()),
            "vis_per_cell_max": int(vis_per_cell.max()),
            "rvt_mean_s": float(rvt_vis.mean()) if rvt_vis.size > 0 else 0.0,
            "rvt_max_s": float(rvt_vis.max()) if rvt_vis.size > 0 else 0.0,
        }
