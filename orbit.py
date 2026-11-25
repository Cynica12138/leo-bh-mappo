# -*- coding: utf-8 -*-
"""Real Starlink SGP4 orbital propagation and WGS-84 ECEF ground geometry."""

import datetime
import json
import math
import os
import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, jday

import config as cfg

# WGS-84 ellipsoid constants
A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

DEFAULT_GP_PATH = os.path.join(os.path.dirname(__file__), "data", "starlink_shell_gp.json")
DEFAULT_EPOCH_UTC = datetime.datetime(2026, 11, 25, 10, 0, 0, tzinfo=datetime.timezone.utc)
DEFAULT_REGION_LAT_DEG = 40.0
DEFAULT_REGION_LON_DEG = 116.4
DEFAULT_CELL_SPACING_M = cfg.CELL_SPACING_M    # 25 km contiguous 3 dB footprint (2 * 485km * tan(1.5 deg) = 25.4 km)
DEFAULT_CLUSTER_SPACING_M = cfg.SAT_SPACING_M  # 165 km regional 3x3 cluster spacing matching overhead Starlink shell


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: float | np.ndarray = 0.0) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates (deg, deg, m) to ECEF meters."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - E2_WGS84) + alt_m) * sin_lat
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def enu_basis(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Return local East, North, Up unit vectors in ECEF at given geodetic coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.stack([-sin_lon, cos_lon, np.zeros_like(lat)], axis=-1)
    north = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    up = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)
    return east.astype(np.float64), north.astype(np.float64), up.astype(np.float64)


def gmst_rad(jd: float, fr: float) -> float:
    """IAU-1982 Greenwich Mean Sidereal Time (radians) for TEME -> ECEF rotation."""
    tut1 = ((jd - 2451545.0) + fr) / 36525.0
    sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 0.093104 * (tut1 ** 2)
        - 6.2e-6 * (tut1 ** 3)
    )
    return (sec % 86400.0) * (2.0 * math.pi / 86400.0)


def teme_to_ecef(r_teme_m: np.ndarray, jd: float, fr: float) -> np.ndarray:
    """Rotate TEME position vectors (m) into WGS-84 ECEF (m) via GMST."""
    theta = gmst_rad(jd, fr)
    c, s = math.cos(theta), math.sin(theta)
    x = c * r_teme_m[..., 0] + s * r_teme_m[..., 1]
    y = -s * r_teme_m[..., 0] + c * r_teme_m[..., 1]
    z = r_teme_m[..., 2]
    return np.stack([x, y, z], axis=-1)


def datetime_to_jday(dt: datetime.datetime):
    """Convert UTC datetime to SGP4 (jd, fr) pair."""
    sec = dt.second + dt.microsecond * 1e-6
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec)


# Module-level cache so repeated env resets do not re-parse the JSON file
_SHELL_CACHE: dict[str, tuple[list[str], list[Satrec]]] = {}


def load_starlink_shell(gp_path: str = DEFAULT_GP_PATH) -> tuple[list[str], list[Satrec]]:
    abs_path = os.path.abspath(gp_path)
    if abs_path not in _SHELL_CACHE:
        with open(abs_path, "r") as f:
            records = json.load(f)
        names, sats = [], []
        for rec in records:
            sat = Satrec()
            omm.initialize(sat, rec)
            names.append(rec.get("OBJECT_NAME", "STARLINK"))
            sats.append(sat)
        _SHELL_CACHE[abs_path] = (names, sats)
    return _SHELL_CACHE[abs_path]


class StarlinkOrbitProvider:
    """Propagates real CelesTrak Starlink OMM/SGP4 ephemerides over 144 fixed WGS-84 ground cells."""

    def __init__(
        self,
        gp_path: str = DEFAULT_GP_PATH,
        center_lat_deg: float = DEFAULT_REGION_LAT_DEG,
        center_lon_deg: float = DEFAULT_REGION_LON_DEG,
        cell_spacing_m: float = DEFAULT_CELL_SPACING_M,
        cluster_spacing_m: float = DEFAULT_CLUSTER_SPACING_M,
        base_epoch: datetime.datetime = DEFAULT_EPOCH_UTC,
    ):
        self.gp_path = gp_path
        self.center_lat_deg = center_lat_deg
        self.center_lon_deg = center_lon_deg
        self.cell_spacing_m = cell_spacing_m
        self.cluster_spacing_m = cluster_spacing_m
        self.base_epoch = base_epoch
        self.names, self.sats = load_starlink_shell(gp_path)

        # Build 144 fixed geographic ground cells (16 regional 3x3 clusters with 25 km contiguous 3 dB footprints)
        # projected onto the WGS-84 ellipsoid around (center_lat_deg, center_lon_deg).
        lat_r = math.radians(center_lat_deg)
        m_per_deg_lat = 111035.0
        m_per_deg_lon = 111320.0 * math.cos(lat_r)
        cx = (np.arange(4) - 1.5) * cluster_spacing_m
        cy = (np.arange(4) - 1.5) * cluster_spacing_m
        lats, lons = [], []
        for ix in range(4):
            for iy in range(4):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        de = cx[ix] + dx * cell_spacing_m
                        dn = cy[iy] + dy * cell_spacing_m
                        lats.append(center_lat_deg + dn / m_per_deg_lat)
                        lons.append(center_lon_deg + de / m_per_deg_lon)
        self.cell_lat = np.array(lats, dtype=np.float64)
        self.cell_lon = np.array(lons, dtype=np.float64)
        self.cell_pos = geodetic_to_ecef(self.cell_lat, self.cell_lon, 0.0)  # (M, 3)
        self.cell_east, self.cell_north, self.cell_up = enu_basis(self.cell_lat, self.cell_lon)

        self.center_ecef = geodetic_to_ecef(np.array(center_lat_deg), np.array(center_lon_deg), 0.0)
        _, _, self.center_up = enu_basis(np.array(center_lat_deg), np.array(center_lon_deg))

    def __deepcopy__(self, memo):
        # Immutable ephemeris provider; safe to share across deepcopied environments
        return self

    def select_overhead_constellation(self, t_offset_sec: float, num_sats: int = cfg.N):
        """Propagate full shell to base_epoch + t_offset_sec and select num_sats regional serving satellites."""
        dt = self.base_epoch + datetime.timedelta(seconds=float(t_offset_sec))
        jd, fr = datetime_to_jday(dt)
        valid_idx, pos_teme = [], []
        for idx, sat in enumerate(self.sats):
            err, r, _ = sat.sgp4(jd, fr)
            if err == 0 and math.isfinite(r[0]) and math.isfinite(r[1]) and math.isfinite(r[2]):
                valid_idx.append(idx)
                pos_teme.append(r)
        pos_ecef = teme_to_ecef(np.asarray(pos_teme, dtype=np.float64) * 1000.0, jd, fr)

        # Filter to regional overhead satellites above 30 deg at region center
        diff0 = pos_ecef - self.center_ecef[None, :]
        dist0 = np.linalg.norm(diff0, axis=-1)
        elev0 = np.degrees(np.arcsin(np.clip((diff0 @ self.center_up) / np.maximum(dist0, 1.0), -1.0, 1.0)))
        pool_local = np.where(elev0 >= 30.0)[0]
        if len(pool_local) < num_sats:
            pool_local = np.argsort(-elev0)[: max(num_sats, 24)]
        pool_pos = pos_ecef[pool_local]

        # Greedy max-coverage / k-medoids selection across the 144 ground cells
        d_pm = np.linalg.norm(pool_pos[:, None, :] - self.cell_pos[None, :, :], axis=-1)  # (P, M)
        chosen = [int(np.argmin(d_pm.mean(axis=1)))]
        min_d = d_pm[chosen[0]].copy()
        for _ in range(num_sats - 1):
            gains = np.maximum(min_d[None, :] - d_pm, 0.0).sum(axis=1)
            gains[chosen] = -1.0
            nxt = int(np.argmax(gains))
            chosen.append(nxt)
            min_d = np.minimum(min_d, d_pm[nxt])

        active_indices = [valid_idx[pool_local[c]] for c in chosen]
        active_names = [self.names[i] for i in active_indices]
        sat_pos = pool_pos[chosen].copy()
        return active_indices, active_names, sat_pos

    def propagate_active(self, active_indices: list[int], t_offset_sec: float) -> np.ndarray:
        """Exact SGP4 propagation of the N active satellites to base_epoch + t_offset_sec (ECEF meters)."""
        dt = self.base_epoch + datetime.timedelta(seconds=float(t_offset_sec))
        jd, fr = datetime_to_jday(dt)
        pos_teme = np.zeros((len(active_indices), 3), dtype=np.float64)
        for i, sat_idx in enumerate(active_indices):
            err, r, _ = self.sats[sat_idx].sgp4(jd, fr)
            if err == 0:
                pos_teme[i] = r
        return teme_to_ecef(pos_teme * 1000.0, jd, fr)

    def elevation_and_visibility(self, sat_pos: np.ndarray, J: int = cfg.J, min_elev_deg: float = 35.0):
        """Compute exact Earth-curved elevation angles (rad) and capacity-feasible visibility matrix V (N, M)."""
        diff = sat_pos[:, None, :] - self.cell_pos[None, :, :]  # (N, M, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (N, M)
        sin_elev = np.clip(np.einsum("nmc,mc->nm", diff, self.cell_up) / np.maximum(dist, 1.0), -1.0, 1.0)
        elev_rad = np.arcsin(sin_elev)
        elev_deg = np.degrees(elev_rad)

        N, M = sat_pos.shape[0], self.cell_pos.shape[0]
        V = np.zeros((N, M), dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        # Step 1: guarantee every cell is covered by at least one capacity-feasible satellite (hardest cells first)
        for m in np.argsort(-dist.min(axis=0)):
            cands = np.where(counts < J)[0]
            best_n = cands[int(np.argmin(dist[cands, m]))]
            V[best_n, m] = 1
            counts[best_n] += 1
        # Step 2: fill each satellite up to J candidate cells with closest remaining visible cells
        for n in range(N):
            rem = J - counts[n]
            if rem <= 0:
                continue
            unadded = np.where((V[n] == 0) & (elev_deg[n] >= min_elev_deg))[0]
            if len(unadded) < rem:
                unadded = np.where(V[n] == 0)[0]
            add = unadded[np.argsort(dist[n, unadded])[:rem]]
            V[n, add] = 1
            counts[n] += len(add)
        return elev_rad, V

    def sample_users_on_wgs84(self, user_cell: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample users uniformly inside each cell's 7.5 km tangent footprint and project onto WGS-84 surface."""
        U = len(user_cell)
        ang = rng.uniform(0.0, 2.0 * math.pi, size=U)
        rad = cfg.CELL_RADIUS_M * np.sqrt(rng.uniform(0.0, 1.0, size=U))
        de = rad * np.cos(ang)
        dn = rad * np.sin(ang)
        c_lat = self.cell_lat[user_cell]
        c_lon = self.cell_lon[user_cell]
        u_lat = c_lat + dn / 111035.0
        u_lon = c_lon + de / (111320.0 * np.cos(np.radians(c_lat)))
        return geodetic_to_ecef(u_lat, u_lon, 0.0)
