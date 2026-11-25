# -*- coding: utf-8 -*-
"""Configuration and presets for Chapter 4 dynamic LEO handover (aligned with paper-simulation)."""

from dataclasses import dataclass, field
import math
import config as base_cfg


# Physical constants
EARTH_RADIUS_M = 6371.0e3
EARTH_MU = 3.986004418e14          # m^3 / s^2
EARTH_OMEGA_RAD_S = 7.2921159e-5   # Earth rotation rate (rad/s)


@dataclass(frozen=True)
class ConstellationSpec:
    """Walker Delta constellation and ground region definition."""

    name: str
    num_planes: int          # P
    sats_per_plane: int      # S
    phase_factor: int        # F
    inclination_deg: float   # i (deg)
    altitude_m: float = base_cfg.H
    min_elev_deg: float = base_cfg.THETA_MIN_DEG
    region_lat_deg: float = 45.0
    region_lon_deg: float = 125.0

    @property
    def total_sats(self) -> int:
        return self.num_planes * self.sats_per_plane

    @property
    def orbit_radius_m(self) -> float:
        return EARTH_RADIUS_M + self.altitude_m

    @property
    def orbit_period_s(self) -> float:
        return 2.0 * math.pi * math.sqrt((self.orbit_radius_m ** 3) / EARTH_MU)

    @property
    def orbit_omega_rad_s(self) -> float:
        return 2.0 * math.pi / self.orbit_period_s


CONSTELLATION_PRESETS: dict[str, ConstellationSpec] = {
    # Tight-capacity primary benchmark: 12~18 active sats (mean 14.6), cap 1.17x~1.75x, min vis 2
    "tight_45n": ConstellationSpec(
        name="tight_45n",
        num_planes=72,
        sats_per_plane=22,
        phase_factor=17,
        inclination_deg=53.0,
        altitude_m=base_cfg.H,
        min_elev_deg=base_cfg.THETA_MIN_DEG,
        region_lat_deg=45.0,
        region_lon_deg=125.0,
    ),
    # Standard-redundancy benchmark: 16~21 active sats (mean 18.2), cap 1.56x~2.04x, min vis 3
    "standard_47n": ConstellationSpec(
        name="standard_47n",
        num_planes=72,
        sats_per_plane=22,
        phase_factor=17,
        inclination_deg=53.0,
        altitude_m=base_cfg.H,
        min_elev_deg=base_cfg.THETA_MIN_DEG,
        region_lat_deg=47.0,
        region_lon_deg=125.0,
    ),
    # Dense constellation generalization: 72x44 = 3168 sats @ 35N, 12~20 active sats (mean 17.2)
    "dense_35n": ConstellationSpec(
        name="dense_35n",
        num_planes=72,
        sats_per_plane=44,
        phase_factor=17,
        inclination_deg=53.0,
        altitude_m=base_cfg.H,
        min_elev_deg=base_cfg.THETA_MIN_DEG,
        region_lat_deg=35.0,
        region_lon_deg=116.0,
    ),
}


def get_constellation_preset(name: str = "tight_45n") -> ConstellationSpec:
    if name not in CONSTELLATION_PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(CONSTELLATION_PRESETS)}")
    return CONSTELLATION_PRESETS[name]


@dataclass
class HandoverConfig:
    """Unified Chapter 4 simulation, trigger, admission, and TRA-HGDRL hyperparameters."""

    # Inherited Chapter 3 (paper-simulation) physical & queue parameters
    M: int = base_cfg.M
    J: int = base_cfg.J
    K_MAX: int = base_cfg.K_MAX
    H: float = base_cfg.H
    THETA_MIN_DEG: float = base_cfg.THETA_MIN_DEG
    THETA_3DB_DEG: float = base_cfg.THETA_3DB_DEG
    P_TOT_DBW: float = base_cfg.P_TOT_DBW
    P_TOT_W: float = base_cfg.P_TOT_W
    G_MAX_T_DBI: float = base_cfg.G_MAX_T_DBI
    G_MAX_T_LIN: float = base_cfg.G_MAX_T_LIN
    G_R_LIN: float = base_cfg.G_R_LIN
    B_TOT: float = base_cfg.B_TOT
    N_RB: int = base_cfg.N_RB
    F_C: float = base_cfg.F_C
    LAMBDA: float = base_cfg.LAMBDA
    K_BOLTZMANN: float = base_cfg.K_BOLTZMANN
    T_NOISE: float = base_cfg.T_NOISE
    UE_LOSS_NF_DB: float = base_cfg.UE_LOSS_NF_DB
    L_P: float = base_cfg.L_P
    T_SLOT: float = base_cfg.T_SLOT
    T_TH_R: int = base_cfg.T_TH_R
    PF_WINDOW: int = base_cfg.PF_WINDOW
    CELL_SPACING_M: float = base_cfg.CELL_SPACING_M
    SAT_SPACING_M: float = base_cfg.SAT_SPACING_M
    CELL_RADIUS_M: float = base_cfg.CELL_RADIUS_M
    D_TH_M: float = base_cfg.D_TH_M
    ETA_M: float = base_cfg.ETA_M
    R_RT_FRAC: float = base_cfg.R_RT_FRAC
    R_BULK_FRAC: float = base_cfg.R_BULK_FRAC
    MMPP_BURST_MULT: float = base_cfg.MMPP_BURST_MULT
    MMPP_P_ON: float = base_cfg.MMPP_P_ON
    MMPP_STAY_ON: float = base_cfg.MMPP_STAY_ON

    # Chapter 4 macro timescale & constellation
    constellation_name: str = "tight_45n"
    T_H: float = 1.0                    # Macro handover step (s)
    slots_per_macro: int = 500          # 1 s / 2 ms = 500 micro slots
    episode_duration_s: int = 600       # Evaluation episode horizon (s)
    train_duration_s: int = 300         # Training episode horizon (s)
    rvt_lookahead_s: int = 180          # Lookahead window for exact RVT (> max 153.7s pass)
    C_MAX: int = 6                      # Max candidate satellites per cell

    # Link-level AR(1) shadow fading (3GPP TR 38.811 elevation-dependent)
    shadow_corr_time_s: float = 10.0    # Correlation time tau_corr (s)
    shadow_sigma_low_elev_db: float = 4.0   # Std at 40 deg elevation (dB)
    shadow_sigma_high_elev_db: float = 2.0  # Std at 90 deg elevation (dB)
    shadow_sigma_scale: float = 1.0         # Global multiplier for sensitivity sweep (e.g., 0.67..2.0)
    user_shadow_offset_db: float = 1.0      # Small intra-cell per-user shadow spread (dB)
    nominal_interf_ratio: float = 0.25      # Baseline nominal CCI-to-noise+sig proxy floor

    # Traffic & spatial heterogeneity
    num_users: int = base_cfg.EVAL_USERS    # Default 200 (sweep 200..500)
    input_rate_mbps: float = 35.0           # Default mean per-user offered rate (Mbps)
    num_hotspots: int = 4
    hotspot_sigma_m: float = 75.0e3
    hotspot_weight_boost: float = 3.5
    rt_frac_min: float = 0.05
    rt_frac_max: float = 0.45
    enable_hotspot_drift: bool = False
    drift_times_s: tuple[int, ...] = (200, 400)
    drift_burst_duration_s: int = 70
    drift_rate_mult: float = 1.75

    # Handover interruption & signaling cost
    tau_ho_ms: float = 30.0             # Default 30 ms = 15 micro slots
    ho_cost_common: float = 1.0         # C^com
    ho_cost_per_user: float = 0.25      # C^ter
    ping_pong_window_s: int = 10        # Ping-pong detection horizon

    # Dual-stage CHO-inspired trigger
    T_cov_s: float = 10.0               # Mandatory RVT threshold (s)
    T_dwell_min_s: int = 6              # Min dwell before proactive trigger (s)
    L_ttt_s: int = 2                    # Consecutive seconds hysteresis for proactive trigger
    sinr_th_db: float = 8.0             # Proactive link SINR threshold (dB)
    sinr_hys_db: float = 1.5            # Hysteresis margin (dB)
    rho_th: float = 0.85                # Proactive satellite beam load threshold
    zeta_th: float = 0.90               # Proactive single-cell overload threshold
    rt_urg_th: float = 0.45             # Proactive RT HOL urgency / drop trigger threshold
    rt_drop_trig_th: float = 0.03       # Recent RT drop rate proactive trigger

    # Satellite-side unified admission control
    rho_max_soft: float = 1.20          # Soft beam-resource load cap

    # Reward weights
    w_tput: float = 1.0
    w_rt_drop: float = 2.8
    w_ho_cost: float = 0.95
    w_outage: float = 4.5
    eta_coop: float = 0.9

    # HGAT & QR-DQN / CVaR hyperparameters
    M_nbr: int = 8                      # Max competing neighbor cells in local graph
    hgat_hidden_dim: int = 64
    hgat_layers: int = 2
    num_quantiles: int = 32
    huber_kappa: float = 1.0
    gamma: float = 0.99
    smdp_max_step: int = 20
    beta_rt_share: float = 0.45         # beta_1 in alpha_m(t)
    beta_rt_urg: float = 0.35           # beta_2 in alpha_m(t)
    alpha_min: float = 0.20
    lr: float = 3e-4
    batch_size: int = 128
    target_tau: float = 0.005

    # Default frozen Chapter 3 MAPPO checkpoints
    mappo_local_ckpt: str = "checkpoints/mappo-apg-local/mappo_bh.pt"
    mappo_coop_ckpt: str = "checkpoints/mappo-coop-gnn/mappo_bh.pt"

    @property
    def constellation(self) -> ConstellationSpec:
        return get_constellation_preset(self.constellation_name)

    @property
    def ho_slots(self) -> int:
        return max(0, int(round((self.tau_ho_ms * 1e-3) / self.T_SLOT)))

    @property
    def shadow_rho(self) -> float:
        if self.shadow_corr_time_s <= 0.0:
            return 0.0
        return math.exp(-self.T_H / self.shadow_corr_time_s)

    @property
    def noise_full_bw_w(self) -> float:
        eff_t = self.T_NOISE * (10.0 ** (self.UE_LOSS_NF_DB / 10.0))
        return self.K_BOLTZMANN * eff_t * self.B_TOT
