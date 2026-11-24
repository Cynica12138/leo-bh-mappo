# -*- coding: utf-8 -*-
"""Simulation and algorithm hyperparameters for LEO multi-satellite beam hopping."""

import math

# 1. System parameters
N = 16                      # Number of LEO satellites
M = 144                     # Number of ground cells (12x12 grid)
J = 14                      # Candidate cells per satellite
K_MAX = 4                   # Max active beams per satellite
H = 515e3                   # Satellite altitude (m)
THETA_MIN_DEG = 40.0        # Minimum elevation angle (deg)
THETA_3DB_DEG = 3.0         # 3 dB beamwidth (deg)
P_TOT_DBW = 34.0            # Total transmit power per satellite (dBW)
P_TOT_W = 10.0 ** (P_TOT_DBW / 10.0)
G_MAX_T_DBI = 34.0          # Max transmit antenna gain (dBi)
G_MAX_T_LIN = 10.0 ** (G_MAX_T_DBI / 10.0)
G_R_LIN = 1.0               # Handheld UE receive antenna gain (0 dBi)
B_TOT = 40e6                # System bandwidth (Hz)
N_RB = 80                   # Resource blocks per beam
F_C = 2e9                   # Carrier frequency (Hz)
C_LIGHT = 3e8               # Speed of light (m/s)
LAMBDA = C_LIGHT / F_C      # Carrier wavelength (0.15 m)
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
T_NOISE = 300.0             # Noise temperature (K)
SHADOW_STD_DB = 3.0         # Log-normal shadow fading std (dB)
MMPP_BURST_MULT = 5.5       # Bursty RT ON-state rate multiplier (mean RT still equals 20%)
MMPP_P_ON = 0.16            # Stationary probability a cell is in an RT burst state
MMPP_STAY_ON = 0.94         # Markov persistence of RT burst state (~17 slots avg burst)

U_RANGE = (100, 500)        # Number of users range
R_RT_FRAC = 0.20            # RT share of per-user input rate (2-10 of 10-50 Mbps)
R_BULK_FRAC = 0.80          # BULK share of per-user input rate (8-40 of 10-50 Mbps)
L_P = 1000.0                # Packet size (bits)
T_SLOT = 2e-3               # Slot duration (2 ms)
ASSOC_PERIOD = 100          # Association update period (slots)
T_TH_R = 40                 # RT packet TTL (slots)
PF_WINDOW = 50              # PF historical throughput window (slots)

# Geometry calibration:
# 3 dB beamwidth 3 deg -> 2*pi*r_ap/lambda * sin(1.5 deg) = 1.61633 (first -3 dB point of |2 J1(u)/u|^2)
KA_APERTURE = 1.61633 / math.sin(math.radians(THETA_3DB_DEG / 2.0))
CELL_SPACING_M = 54e3       # 54 km inter-cell center spacing
CELL_RADIUS_M = 9.5e3       # 9.5 km user radius inside 3 dB beam footprint
D_TH_M = 108e3              # Distance prior threshold d_th (2 cell hops)
ETA_M = 35e3                # Distance prior smoothing factor eta

# 2. ISA association parameters
T_0 = 100.0
T_MIN = 1e-3
RHO = 0.95
L_MAX = 100

# 3. MAPPO parameters
NODE_FEAT_DIM = 5
D_G = 8
HIDDEN_DIM = 128
N_MB = 64
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2
EP = 1000
ST = 128
N_EPO = 4
LR_ACTOR = 2e-4
LR_CRITIC = 3e-4
ALPHA = 0.7
XI_1 = 0.5                  # Distance prior weight in GAT attention
XI_2 = 0.5                  # Common-visibility prior weight in GAT attention
BETA_D = 0.01               # Discrete entropy weight
BETA_C = 0.002              # Continuous entropy weight
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0
ZETA_NORM = 2.0e7           # Slot delivered bits normalization (~10 Gbps * 2 ms = 20 Mbits)
GAMMA_NORM = 10.0           # Slot average RT delay normalization (slots)

EVAL_RATES_MBPS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
