# -*- coding: utf-8 -*-
"""Heuristic beam hopping baselines: P-BH, QLF-BH, USWGP-BH."""

import numpy as np
import config as cfg


def _equal_power(beam_sel: np.ndarray) -> np.ndarray:
    counts = beam_sel.sum(axis=1, keepdims=True)
    return np.where(counts > 0, beam_sel / np.maximum(counts, 1.0), 0.0)


class PeriodicBH:
    """P-BH: Fixed periodic polling over associated cells with equal power."""

    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, obs: dict):
        mask = obs["action_mask"]
        N, J = mask.shape
        beam_sel = np.zeros((N, J), dtype=np.float32)
        for n in range(N):
            avail = np.where(mask[n] > 0.5)[0]
            k = min(cfg.K_MAX, len(avail))
            if k == 0:
                continue
            start = (self.t * k) % len(avail)
            chosen = [avail[(start + i) % len(avail)] for i in range(k)]
            beam_sel[n, chosen] = 1.0
        self.t += 1
        return beam_sel, _equal_power(beam_sel)


class QLFBeamHopping:
    """QLF-BH: Prioritize associated cells with largest current queue length, equal power."""

    def reset(self):
        pass

    def act(self, obs: dict):
        F = obs["F"]          # (N, J, 5); col 0 = normalized queue length
        mask = obs["action_mask"]
        N, J = mask.shape
        beam_sel = np.zeros((N, J), dtype=np.float32)
        for n in range(N):
            avail = np.flatnonzero((mask[n] > 0.5) & (F[n, :, 0] > 1e-6))
            k = min(cfg.K_MAX, len(avail))
            if k == 0:
                continue
            scores = F[n, avail, 0]
            top_idx = avail[np.argsort(-scores)[:k]]
            beam_sel[n, top_idx] = 1.0
        return beam_sel, _equal_power(beam_sel)


class USWGPBeamHopping:
    """USWGP-BH: Service-weight gain priority combining queue demand and RT delay, equal power."""

    def reset(self):
        pass

    def act(self, obs: dict):
        F = obs["F"]          # (N, J, 5): [q, avg_d, hol_w, hist_tput, v_cov]
        mask = obs["action_mask"]
        N, J = mask.shape
        beam_sel = np.zeros((N, J), dtype=np.float32)
        for n in range(N):
            avail = np.flatnonzero((mask[n] > 0.5) & (F[n, :, 0] > 1e-6))
            k = min(cfg.K_MAX, len(avail))
            if k == 0:
                continue
            # Combine queue backlog, RT delay, and HOL urgency
            scores = 0.5 * F[n, avail, 0] + 1.2 * F[n, avail, 1] + 0.8 * F[n, avail, 2]
            top_idx = avail[np.argsort(-scores)[:k]]
            beam_sel[n, top_idx] = 1.0
        return beam_sel, _equal_power(beam_sel)
