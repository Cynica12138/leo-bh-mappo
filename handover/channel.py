# -*- coding: utf-8 -*-
"""Dynamic 3D Bessel channel model with per-link (sat_id, cell_id) AR(1) shadow fading."""

import math
import numpy as np

from env import bessel_beam_gain
from handover.config import HandoverConfig
from handover.geometry import GroundCellLayout, TopologySnapshot


class LinkAR1Shadowing:
    """Per-link (sat_id, cell_id) Gaussian-Markov AR(1) shadow fading in dB.

    Follows 3GPP TR 38.811 elevation-dependent shadow std sigma_sh(theta):
      L_sh_dB(t) = rho * L_sh_dB(t-1) + sqrt(1 - rho^2) * sigma_sh(theta(t)) * eps(t)
    """

    def __init__(self, cfg: HandoverConfig):
        self.cfg = cfg
        self.rho = cfg.shadow_rho
        self.innov_scale = math.sqrt(max(1.0 - self.rho ** 2, 0.0))
        # Dict keyed by global sat_id -> float64 array of shape (M,) in dB
        self._state_db: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._state_db.clear()

    def sigma_db(self, elev_rad: np.ndarray) -> np.ndarray:
        """Elevation-dependent shadow fading std (dB), decreasing linearly from 40 deg to 90 deg."""
        elev_deg = np.degrees(elev_rad)
        frac = np.clip((elev_deg - self.cfg.THETA_MIN_DEG) / max(90.0 - self.cfg.THETA_MIN_DEG, 1.0), 0.0, 1.0)
        base_sig = self.cfg.shadow_sigma_low_elev_db * (1.0 - frac) + self.cfg.shadow_sigma_high_elev_db * frac
        return base_sig * self.cfg.shadow_sigma_scale

    def step(self, snap: TopologySnapshot, rng: np.random.Generator) -> np.ndarray:
        """Advance AR(1) shadowing by 1 macro step and return shadow loss (N_act, M) in dB."""
        active_ids = snap.active_sat_ids
        active_set = {int(s) for s in active_ids}
        # Prune departed satellites
        for stale in list(self._state_db.keys()):
            if stale not in active_set:
                del self._state_db[stale]

        sig = self.sigma_db(snap.elev_rad)  # (N_act, M)
        out_db = np.empty((len(active_ids), self.cfg.M), dtype=np.float64)
        for idx, sid in enumerate(active_ids):
            sid_i = int(sid)
            eps = rng.normal(0.0, 1.0, size=self.cfg.M)
            if sid_i not in self._state_db:
                # Stationary initialization for newly entering satellite
                val = sig[idx] * eps
            else:
                val = self.rho * self._state_db[sid_i] + self.innov_scale * sig[idx] * eps
            self._state_db[sid_i] = val
            out_db[idx] = val
        return out_db


class DynamicChannelModel:
    """Dynamic forward-link channel gains, nominal rates, and 3D Bessel H_sq / I_mat tensors."""

    def __init__(self, cfg: HandoverConfig, layout: GroundCellLayout):
        self.cfg = cfg
        self.layout = layout
        self.shadowing = LinkAR1Shadowing(cfg)
        self.user_shadow_offset_db: np.ndarray | None = None  # (U,) fixed intra-cell spread
        # Smoothed historical interference per cell (W) for nominal rate estimation
        self.ema_cell_interf_w = np.full(cfg.M, cfg.noise_full_bw_w * cfg.nominal_interf_ratio, dtype=np.float64)

    def reset(self, num_users: int, rng: np.random.Generator) -> None:
        self.shadowing.reset()
        self.user_shadow_offset_db = rng.normal(0.0, self.cfg.user_shadow_offset_db, size=num_users)
        self.ema_cell_interf_w.fill(self.cfg.noise_full_bw_w * self.cfg.nominal_interf_ratio)

    def update_nominal_links(
        self, snap: TopologySnapshot, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Advance link-level AR(1) shadowing and compute nominal link metrics for (N_act, M).

        Returns dict with:
          - shadow_db: (N_act, M) link-level shadow loss in dB
          - cell_path_gain: (N_act, M) linear channel gain at boresight (cell center)
          - nominal_sinr: (N_act, M) linear nominal SINR
          - nominal_sinr_db: (N_act, M) nominal SINR in dB
          - nominal_rate_bps: (N_act, M) nominal Shannon rate R_hat_{n,m}(t) in bps
        """
        shadow_db = self.shadowing.step(snap, rng)
        l_sh = 10.0 ** (shadow_db / 10.0)
        dist = np.maximum(snap.slant_dist_m.astype(np.float64), 1.0)
        fspl = (4.0 * math.pi * dist / self.cfg.LAMBDA) ** 2
        # Boresight gain at cell center (off-axis = 0 -> bessel gain = 1.0)
        cell_path_gain = (self.cfg.G_MAX_T_LIN * self.cfg.G_R_LIN) / (fspl * l_sh)

        p_nom = self.cfg.P_TOT_W / self.cfg.K_MAX
        # Combine smoothed measured CCI with a signal-proportional CCI floor so nominal SINR reflects full frequency reuse
        sig_nom = p_nom * cell_path_gain
        cci_floor = self.cfg.nominal_interf_ratio * sig_nom
        denom = self.cfg.noise_full_bw_w + np.maximum(self.ema_cell_interf_w[None, :], 0.5 * cci_floor)
        nominal_sinr = sig_nom / np.maximum(denom, 1e-30)
        nominal_sinr_db = 10.0 * np.log10(np.maximum(nominal_sinr, 1e-12))
        nominal_rate_bps = self.cfg.B_TOT * np.log2(1.0 + nominal_sinr)

        return {
            "shadow_db": shadow_db,
            "cell_path_gain": cell_path_gain,
            "nominal_sinr": nominal_sinr,
            "nominal_sinr_db": nominal_sinr_db,
            "nominal_rate_bps": nominal_rate_bps,
        }

    def update_ema_interference(self, measured_cell_interf_w: np.ndarray, alpha_ema: float = 0.25) -> None:
        """Update smoothed per-cell interference estimate from actual/surrogate slot measurements."""
        valid = measured_cell_interf_w >= 0.0
        self.ema_cell_interf_w[valid] = (
            (1.0 - alpha_ema) * self.ema_cell_interf_w[valid] + alpha_ema * measured_cell_interf_w[valid]
        )

    def build_user_channel_tensors(
        self,
        snap: TopologySnapshot,
        shadow_db: np.ndarray,
        cand_cells: np.ndarray,
        user_ecef: np.ndarray,
        user_cell: np.ndarray,
        cell_users_arr: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build exact 3D Bessel channel gain tensor H_sq (N_act, J, U) and intra-sat coupling I_mat (N_act, J, J).

        Used for Mode B (frozen MAPPO-BH / real 2ms PF simulation) and Mode A offline calibration.
        """
        n_act, j_dim = cand_cells.shape
        u_cnt = user_ecef.shape[0]
        if n_act == 0:
            return (
                np.zeros((0, j_dim, u_cnt), dtype=np.float64),
                np.zeros((0, j_dim, j_dim), dtype=np.float32),
            )

        # User-to-satellite vectors in ECEF
        vec_su = user_ecef[None, :, :] - snap.sat_pos_ecef[:, None, :]  # (N_act, U, 3)
        dist_su = np.linalg.norm(vec_su, axis=-1)                        # (N_act, U)
        unit_su = vec_su / np.maximum(dist_su[:, :, None], 1.0)          # (N_act, U, 3)

        # Per-user shadow fading = link-level cell shadow + fixed small intra-cell offset
        u_sh_db = shadow_db[:, user_cell]
        if self.user_shadow_offset_db is not None:
            u_sh_db = u_sh_db + self.user_shadow_offset_db[None, :]
        l_sh = 10.0 ** (u_sh_db / 10.0)
        fspl = (4.0 * math.pi * np.maximum(dist_su, 1.0) / self.cfg.LAMBDA) ** 2
        base_path_gain = (self.cfg.G_MAX_T_LIN * self.cfg.G_R_LIN) / (fspl * l_sh)  # (N_act, U)

        H_sq = np.zeros((n_act, j_dim, u_cnt), dtype=np.float64)
        for n in range(n_act):
            s_pos = snap.sat_pos_ecef[n]
            for j in range(j_dim):
                m = int(cand_cells[n, j])
                if m < 0:
                    continue
                vec_sb = self.layout.cell_pos_ecef[m] - s_pos
                unit_sb = vec_sb / max(float(np.linalg.norm(vec_sb)), 1.0)
                cos_ang = np.clip(unit_su[n] @ unit_sb, -1.0, 1.0)
                off_axis = np.arccos(cos_ang)
                H_sq[n, j, :] = base_path_gain[n] * bessel_beam_gain(off_axis)

        # Normalized intra-satellite beam interference coupling matrix I_mat (N_act, J, J)
        I_mat = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)
        for n in range(n_act):
            for j in range(j_dim):
                mj = int(cand_cells[n, j])
                if mj < 0:
                    continue
                us = cell_users_arr[mj]
                if len(us) == 0:
                    continue
                sig_j = max(float(H_sq[n, j, us].mean()), 1e-30)
                for i in range(j_dim):
                    if i == j or int(cand_cells[n, i]) < 0:
                        continue
                    I_mat[n, i, j] = float(np.clip(H_sq[n, i, us].mean() / sig_j, 0.0, 2.0))

        return H_sq, I_mat
