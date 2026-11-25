# -*- coding: utf-8 -*-
"""Event-triggered Semi-MDP (SMDP) transition tracker and stratified experience replay buffer."""

from dataclasses import dataclass
import numpy as np

from handover.graph import CellGraphObs, collate_cell_graphs


@dataclass
class SMDPTransition:
    """Event-triggered multi-step transition with deep-copied observations and next_action_mask."""

    obs: CellGraphObs
    action: int                  # 0 = Stay, 1..C_MAX = cand satellite slot
    reward_acc: float            # sum_{k=0}^{delta_t-1} gamma^k r(t+k)
    next_obs: CellGraphObs
    delta_t: int                 # steps elapsed between triggers (1..smdp_max_step)
    done: bool
    is_handover_event: bool      # True if action != 0 or obs.is_mandatory


class SMDPTracker:
    """Tracks per-cell pending triggered decisions and emits SMDP multi-step transitions."""

    def __init__(self, num_cells: int, gamma: float, max_step: int = 20):
        self.num_cells = num_cells
        self.gamma = float(gamma)
        self.max_step = int(max_step)
        self._pending: list[dict | None] = [None] * num_cells

    def reset(self) -> None:
        self._pending = [None] * self.num_cells

    def record_decision(self, cell_id: int, obs: CellGraphObs, action: int) -> None:
        self._pending[cell_id] = {
            "obs": obs.copy(),
            "action": int(action),
            "reward_acc": 0.0,
            "steps": 0,
            "discount": 1.0,
            "is_ho": (int(action) != 0) or obs.is_mandatory,
        }

    def accumulate_step_rewards(self, rewards: np.ndarray) -> None:
        for m in range(self.num_cells):
            pend = self._pending[m]
            if pend is not None:
                pend["reward_acc"] += pend["discount"] * float(rewards[m])
                pend["discount"] *= self.gamma
                pend["steps"] += 1

    def pop_ready_cells(self, triggered_mask: np.ndarray, done: bool) -> list[int]:
        """Return cell IDs whose pending transition should finalize at the current step."""
        ready = []
        for m in range(self.num_cells):
            pend = self._pending[m]
            if pend is None or pend["steps"] == 0:
                continue
            if done or (triggered_mask[m] > 0) or (pend["steps"] >= self.max_step):
                ready.append(m)
        return ready

    def finalize(self, cell_id: int, next_obs: CellGraphObs, done: bool) -> SMDPTransition | None:
        pend = self._pending[cell_id]
        if pend is None or pend["steps"] == 0:
            return None
        tr = SMDPTransition(
            obs=pend["obs"],
            action=pend["action"],
            reward_acc=float(pend["reward_acc"]),
            next_obs=next_obs.copy(),
            delta_t=int(pend["steps"]),
            done=bool(done),
            is_handover_event=bool(pend["is_ho"]),
        )
        self._pending[cell_id] = None
        return tr


class StratifiedSMDPReplayBuffer:
    """Two-pool stratified replay buffer separating handover/mandatory events from routine Stay events."""

    def __init__(self, capacity: int = 80000, ho_sample_frac: float = 0.75):
        self.cap_ho = int(capacity * 0.7)
        self.cap_stay = max(1, capacity - self.cap_ho)
        self.ho_sample_frac = float(ho_sample_frac)
        self._ho_buf: list[SMDPTransition] = []
        self._stay_buf: list[SMDPTransition] = []
        self._ho_ptr = 0
        self._stay_ptr = 0

    def __len__(self) -> int:
        return len(self._ho_buf) + len(self._stay_buf)

    def push(self, tr: SMDPTransition) -> None:
        if tr.is_handover_event:
            if len(self._ho_buf) < self.cap_ho:
                self._ho_buf.append(tr)
            else:
                self._ho_buf[self._ho_ptr] = tr
            self._ho_ptr = (self._ho_ptr + 1) % self.cap_ho
        else:
            if len(self._stay_buf) < self.cap_stay:
                self._stay_buf.append(tr)
            else:
                self._stay_buf[self._stay_ptr] = tr
            self._stay_ptr = (self._stay_ptr + 1) % self.cap_stay

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n_ho_avail = len(self._ho_buf)
        n_st_avail = len(self._stay_buf)
        if n_ho_avail + n_st_avail == 0:
            raise RuntimeError("Cannot sample from empty replay buffer.")

        if n_ho_avail > 0 and n_st_avail > 0:
            n_ho = min(n_ho_avail, max(1, int(round(batch_size * self.ho_sample_frac))))
            n_st = batch_size - n_ho
            idx_ho = rng.integers(0, n_ho_avail, size=n_ho)
            idx_st = rng.integers(0, n_st_avail, size=n_st)
            batch = [self._ho_buf[i] for i in idx_ho] + [self._stay_buf[i] for i in idx_st]
        elif n_ho_avail > 0:
            idx = rng.integers(0, n_ho_avail, size=batch_size)
            batch = [self._ho_buf[i] for i in idx]
        else:
            idx = rng.integers(0, n_st_avail, size=batch_size)
            batch = [self._stay_buf[i] for i in idx]

        obs_b = collate_cell_graphs([tr.obs for tr in batch])
        next_obs_b = collate_cell_graphs([tr.next_obs for tr in batch])
        return {
            "obs": obs_b,
            "next_obs": next_obs_b,
            "action": np.array([tr.action for tr in batch], dtype=np.int64),
            "reward": np.array([tr.reward_acc for tr in batch], dtype=np.float32),
            "delta_t": np.array([tr.delta_t for tr in batch], dtype=np.float32),
            "done": np.array([1.0 if tr.done else 0.0 for tr in batch], dtype=np.float32),
        }
