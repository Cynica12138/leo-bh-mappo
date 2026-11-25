"""One-slot coordinate search using the simulator's PF and queue rules."""

import numpy as np

import config as cfg


class SlotService:
    """Evaluate batches of feasible schedules against a fixed queue snapshot."""

    def __init__(self, env):
        self.h = env.H_sq.reshape(cfg.N * cfg.J, env.U)
        self.noise = cfg.K_BOLTZMANN * cfg.T_NOISE * 10 ** (cfg.UE_LOSS_NF_DB / 10) * cfg.B_TOT
        self.rt = env.rt_buckets.sum(axis=1)
        self.backlog = self.rt + env.bulk_queue
        self.urgent = env.rt_buckets[:, -8:].sum(axis=1)
        self.pf_groups = []
        groups = {}
        for n, j in zip(*np.where(env.action_mask > 0.5)):
            users = env.cell_users_arr[env.cand_cells[n, j]]
            users = users[self.backlog[users] > 1e-9]
            if users.size:
                groups.setdefault(users.size, []).append((n * cfg.J + j, users))
        self.user_beam = np.zeros(env.U, dtype=np.intp)
        for entries in groups.values():
            for beam, users in entries:
                self.user_beam[users] = beam
        self.direct_gain = self.h[self.user_beam, np.arange(env.U)]
        for size, group in groups.items():
            users = np.stack([users for _, users in group])
            hist = np.maximum(env.hist_tput[users], 1e4)
            self.pf_groups.append((size, users, hist))

    def capacity(self, watts, total_rx=None):
        sig = watts[:, self.user_beam] * self.direct_gain
        if total_rx is None:
            total_rx = watts @ self.h
        se = np.log2(1 + sig / (self.noise + np.maximum(total_rx - sig, 0.0)))
        packets = np.zeros_like(se)
        rb_bw = cfg.B_TOT / cfg.N_RB
        for size, users, hist in self.pf_groups:
            if size == 1:
                packets[:, users.ravel()] = se[:, users.ravel()] * cfg.B_TOT
                continue
            eff = se[:, users]
            est = np.broadcast_to(hist, eff.shape).copy()
            assigned = np.zeros_like(eff)
            batch_idx = np.arange(eff.shape[0])[:, None]
            cell_idx = np.arange(eff.shape[1])[None, :]
            for _ in range(cfg.N_RB // 4):
                winner = np.argmax(eff / est, axis=-1)
                index = (batch_idx, cell_idx, winner)
                est[index] += eff[index] * (4 * rb_bw / cfg.PF_WINDOW)
                assigned[index] += 4
            packets[:, users] = assigned * eff * rb_bw
        return packets * (cfg.T_SLOT / cfg.L_P)

    def scores(self, watts, total_rx, rt_weight):
        capacity = self.capacity(watts, total_rx)
        delivered = np.minimum(capacity, self.backlog).sum(axis=1)
        priority = np.minimum(capacity, self.rt).sum(axis=1) + np.minimum(capacity, self.urgent).sum(axis=1)
        return delivered + rt_weight * priority


def best_global_swap(env, beam_sel, power_ratios, rt_weight=0.5, shortlist=8):
    """Rank swaps by estimated service, then verify the shortlist with the exact PF rule."""
    model = SlotService(env)
    watts = power_ratios.astype(np.float64).ravel() * cfg.P_TOT_W
    cell_backlog = np.bincount(env.user_cell, weights=model.backlog, minlength=cfg.M)
    available = (env.action_mask > 0.5) & (cell_backlog[env.cand_cells.clip(min=0)] > 1e-9)
    lit = watts.reshape(cfg.N, cfg.J) > 0
    n, old, new = np.where(lit[:, :, None] & (available & ~lit)[:, None, :])
    if not n.size:
        return beam_sel, power_ratios
    remove = n * cfg.J + old
    add = n * cfg.J + new
    rx0 = watts @ model.h
    delta = watts[remove, None]
    rx = rx0 + delta * (model.h[add] - model.h[remove])
    unchanged_signal = watts[model.user_beam] * model.direct_gain
    sig = np.where(model.user_beam == remove[:, None], 0.0, unchanged_signal)
    sig += (model.user_beam == add[:, None]) * delta * model.direct_gain
    se = np.log2(1 + sig / (model.noise + np.maximum(rx - sig, 0)))
    # Equal RB shares give a cheap ranking. The acceptance score below uses exact PF.
    counts = np.bincount(env.user_cell, weights=model.backlog > 1e-9, minlength=cfg.M)
    cap = se * (cfg.B_TOT * cfg.T_SLOT / cfg.L_P) / np.maximum(counts[env.user_cell], 1)
    score = np.minimum(cap, model.backlog).sum(axis=1) + rt_weight * (
        np.minimum(cap, model.rt).sum(axis=1) + np.minimum(cap, model.urgent).sum(axis=1)
    )
    k = min(shortlist, n.size)
    top = np.argpartition(score, -k)[-k:]
    candidates = np.broadcast_to(watts, (k + 1, watts.size)).copy()
    rows = np.arange(1, k + 1)
    candidates[rows, add[top]] = watts[remove[top]]
    candidates[rows, remove[top]] = 0
    scores = model.scores(candidates, np.vstack([rx0, rx[top]]), rt_weight)
    best = int(np.argmax(scores))
    if scores[best] - scores[0] <= 1e-6:
        return beam_sel, power_ratios
    ratios = candidates[best].reshape(cfg.N, cfg.J) / cfg.P_TOT_W
    return (ratios > 0).astype(beam_sel.dtype), ratios
