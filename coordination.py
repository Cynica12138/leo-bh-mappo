"""Neighbor interference features derived from link gains and aggregate queues."""

import numpy as np

import config as cfg

FEATURE_COUNT = 3


class NeighborInterference:
    """Summarize incoming interference and expected harm to other active cells."""

    def __init__(self, env):
        n, j, u = env.H_sq.shape
        members = np.eye(cfg.M, dtype=np.float64)[env.user_cell]
        self.members = members / np.maximum(members.sum(axis=0), 1)
        self.gains = env.H_sq_base.reshape(n * j, u) @ self.members
        self.candidate_cells = env.cand_cells.clip(min=0)
        self.valid = env.cand_cells >= 0
        self.home_gains = np.zeros(cfg.M)
        self.serving_beams = np.flatnonzero(env.action_mask.ravel())
        self.served_cells = self.candidate_cells.ravel()[self.serving_beams]
        self.home_gains[self.served_cells] = self.gains[self.serving_beams, self.served_cells]
        self.foreign = np.arange(n).repeat(j)[:, None] != env.cell_sat[None, :]
        self.nominal_power = cfg.P_TOT_W / cfg.K_MAX
        self.noise = cfg.K_BOLTZMANN * cfg.T_NOISE * 10 ** (cfg.UE_LOSS_NF_DB / 10) * cfg.B_TOT
        # Precompute normalized beam-to-beam interference coupling E_mat[src, dst] for cooperative power control
        M_tot = n * j
        self.E_mat = np.zeros((M_tot, M_tot), dtype=np.float32)
        if self.serving_beams.size > 0:
            denom = np.maximum(self.home_gains[self.served_cells], 1e-30)
            self.E_mat[:, self.serving_beams] = np.clip(
                self.gains[:, self.served_cells] / denom[None, :], 0.0, 5.0
            ).astype(np.float32)

    def features(self, env, nodes):
        # Predict other satellites' beam use using aggregate demand, capped by their beam budgets.
        demand = (nodes[..., 0] + nodes[..., 1]) * env.action_mask
        occupied = demand > 1e-6
        priority = np.where(occupied, np.sqrt(demand), 0.0)
        active_count = np.minimum(occupied.sum(axis=1, keepdims=True), cfg.K_MAX)
        probability = np.minimum(
            active_count * priority / np.maximum(priority.sum(axis=1, keepdims=True), 1e-9), 1.0
        )
        expected_power = cfg.P_TOT_W * probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-9)

        # Mobility fading acts on the receiving cell for all transmitters.
        scale = 10 ** (env.cell_fade_db / 10)
        gains = self.gains * scale
        home_gains = self.home_gains * scale
        cross_gains = gains * self.foreign
        incoming = expected_power.ravel() @ cross_gains
        own_received = expected_power.ravel() @ (gains - cross_gains)
        service_p = expected_power.ravel()[self.serving_beams]
        signal = np.zeros(cfg.M)
        signal[self.served_cells] = service_p * home_gains[self.served_cells]
        denominator = self.noise + np.maximum(own_received - signal, 0.0) + incoming

        # Compute the marginal spectral efficiency loss from a candidate beam at nominal power.
        direct_ratio = incoming[self.candidate_cells] / np.maximum(
            self.nominal_power * gains[np.arange(gains.shape[0]), self.candidate_cells.ravel()].reshape(nodes.shape[:2]),
            self.noise,
        )
        rt_deadline = env.rt_buckets[:, -8:].sum(axis=1)
        near_expiry = np.bincount(env.user_cell, weights=rt_deadline, minlength=cfg.M)
        victim_weight = np.zeros(cfg.M)
        flat = nodes.reshape(-1, nodes.shape[-1])[self.serving_beams]
        victim_weight[self.served_cells] = np.minimum(0.35 + flat[:, 0] * 2 + flat[:, 1] + flat[:, 3], 3.0)
        leak = self.nominal_power * cross_gains
        harm = (np.log1p(signal / denominator) - np.log1p(signal / (denominator + leak))) @ victim_weight
        messages = np.stack([
            np.log1p(direct_ratio),
            np.log1p(harm.reshape(nodes.shape[:2])),
            np.log1p(near_expiry[self.candidate_cells] / 40),
        ], axis=-1)
        messages *= self.valid[..., None]
        return np.concatenate([nodes, messages.astype(np.float32)], axis=-1)
