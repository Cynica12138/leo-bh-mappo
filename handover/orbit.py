# -*- coding: utf-8 -*-
"""Analytical circular Walker Delta LEO constellation propagator (ECI -> ECEF)."""

import math
import numpy as np

from handover.config import ConstellationSpec, EARTH_OMEGA_RAD_S


class WalkerConstellation:
    """Analytical two-body circular Walker Delta constellation with Earth rotation."""

    def __init__(self, spec: ConstellationSpec):
        self.spec = spec
        self.P = spec.num_planes
        self.S = spec.sats_per_plane
        self.F = spec.phase_factor
        self.total_sats = spec.total_sats
        self.r_orb = spec.orbit_radius_m
        self.omega_orb = spec.orbit_omega_rad_s
        self.inc_rad = math.radians(spec.inclination_deg)
        self.cos_inc = math.cos(self.inc_rad)
        self.sin_inc = math.sin(self.inc_rad)

        # Orbital elements for every satellite id in 0..total_sats-1
        p_idx = np.repeat(np.arange(self.P, dtype=np.float64), self.S)
        s_idx = np.tile(np.arange(self.S, dtype=np.float64), self.P)
        self.raan = (2.0 * math.pi * p_idx) / self.P
        self.u0 = (2.0 * math.pi * s_idx) / self.S + (2.0 * math.pi * self.F * p_idx) / self.total_sats
        self.cos_raan = np.cos(self.raan)
        self.sin_raan = np.sin(self.raan)

    def propagate(self, t_sec: float, t0_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Compute ECEF positions (N_tot, 3) and velocities (N_tot, 3) at time t_sec + t0_sec."""
        t = float(t_sec + t0_sec)
        u = self.u0 + self.omega_orb * t
        cu = np.cos(u)
        su = np.sin(u)

        # ECI position
        x_eci = self.r_orb * (self.cos_raan * cu - self.sin_raan * su * self.cos_inc)
        y_eci = self.r_orb * (self.sin_raan * cu + self.cos_raan * su * self.cos_inc)
        z_eci = self.r_orb * (su * self.sin_inc)

        # ECI velocity (d/dt of position)
        v_orb = self.r_orb * self.omega_orb
        vx_eci = v_orb * (-self.cos_raan * su - self.sin_raan * cu * self.cos_inc)
        vy_eci = v_orb * (-self.sin_raan * su + self.cos_raan * cu * self.cos_inc)
        vz_eci = v_orb * (cu * self.sin_inc)

        # Rotate ECI -> ECEF by GMST = omega_e * t
        gmst = EARTH_OMEGA_RAD_S * t
        cg = math.cos(gmst)
        sg = math.sin(gmst)

        x_e = cg * x_eci + sg * y_eci
        y_e = -sg * x_eci + cg * y_eci
        z_e = z_eci

        # ECEF velocity: R(t) v_eci - omega_e x r_ecef
        vx_rot = cg * vx_eci + sg * vy_eci
        vy_rot = -sg * vx_eci + cg * vy_eci
        vx_e = vx_rot + EARTH_OMEGA_RAD_S * y_e
        vy_e = vy_rot - EARTH_OMEGA_RAD_S * x_e
        vz_e = vz_eci

        pos = np.stack([x_e, y_e, z_e], axis=-1)
        vel = np.stack([vx_e, vy_e, vz_e], axis=-1)
        return pos, vel

    def propagate_subset(
        self, sat_ids: np.ndarray, t_sec: float, t0_sec: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast ECEF propagation for a subset of satellite IDs."""
        t = float(t_sec + t0_sec)
        c_r = self.cos_raan[sat_ids]
        s_r = self.sin_raan[sat_ids]
        u = self.u0[sat_ids] + self.omega_orb * t
        cu = np.cos(u)
        su = np.sin(u)

        x_eci = self.r_orb * (c_r * cu - s_r * su * self.cos_inc)
        y_eci = self.r_orb * (s_r * cu + c_r * su * self.cos_inc)
        z_eci = self.r_orb * (su * self.sin_inc)

        v_orb = self.r_orb * self.omega_orb
        vx_eci = v_orb * (-c_r * su - s_r * cu * self.cos_inc)
        vy_eci = v_orb * (-s_r * su + c_r * cu * self.cos_inc)
        vz_eci = v_orb * (cu * self.sin_inc)

        gmst = EARTH_OMEGA_RAD_S * t
        cg = math.cos(gmst)
        sg = math.sin(gmst)

        x_e = cg * x_eci + sg * y_eci
        y_e = -sg * x_eci + cg * y_eci
        z_e = z_eci

        vx_e = (cg * vx_eci + sg * vy_eci) + EARTH_OMEGA_RAD_S * y_e
        vy_e = (-sg * vx_eci + cg * vy_eci) - EARTH_OMEGA_RAD_S * x_e
        vz_e = vz_eci

        return np.stack([x_e, y_e, z_e], axis=-1), np.stack([vx_e, vy_e, vz_e], axis=-1)
