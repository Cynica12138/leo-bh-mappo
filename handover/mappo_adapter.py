# -*- coding: utf-8 -*-
"""Dynamic adapters for Chapter 3 (paper-simulation) frozen MAPPO-BH and heuristics.

Supports variable active satellite sets N_act(t) via sat_id-indexed GRU hidden states
and dynamic F_rich / I_mat / optional E_mat tensors.
"""

from typing import Sequence
import numpy as np
import torch

import config as base_cfg
from models import MAPPOActor


def equal_power_np(b_mask: np.ndarray) -> np.ndarray:
    counts = b_mask.sum(axis=1, keepdims=True)
    return np.where(counts > 0, b_mask / np.maximum(counts, 1.0), 0.0).astype(np.float32)


class DynamicMAPPOAdapter:
    """Wrap a frozen paper-simulation MAPPOActor for dynamic N_act(t) constellations.

    Compatible with both:
      - checkpoints/mappo-apg-local/mappo_bh.pt (pure local 8-dim F_rich + I_mat)
      - checkpoints/mappo-coop-gnn/mappo_bh.pt  (11-dim F_rich + I_mat + E_mat)
      - checkpoints/reference/mappo_bh.pt       (8-dim F_rich + I_mat)
    """

    def __init__(self, ckpt_path: str, device: str = "cpu", bc_only: bool = False):
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cross_satellite: bool = bool(ckpt.get("cross_satellite", False))
        self.power_conditioned: bool = bool(ckpt.get("power_conditioned", False))
        self.coop_power_gnn: bool = bool(ckpt.get("coop_power_gnn", False))
        if self.coop_power_gnn:
            self.cross_satellite = True

        state = ckpt["bc_actor_state_dict"] if bc_only else (ckpt.get("best_actor_state_dict") or ckpt["actor_state_dict"])
        self.node_dim = int(state["node_proj.weight"].shape[1])
        self.hidden_dim = int(state["gru.weight_hh_l0"].shape[1])

        self.actor = MAPPOActor(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            power_conditioned=self.power_conditioned,
            coop_power_gnn=self.coop_power_gnn,
        ).to(self.device)
        self.actor.load_state_dict(state)
        self.actor.eval()

        # Per-satellite GRU hidden states keyed by globally unique satellite ID
        self.hidden_states: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.hidden_states.clear()

    def sync_active_satellites(self, active_sat_ids: Sequence[int]) -> None:
        """Initialize GRU states for newly entering satellites and prune departed ones."""
        active_set = {int(s) for s in active_sat_ids}
        for stale in list(self.hidden_states.keys()):
            if stale not in active_set:
                del self.hidden_states[stale]
        for sid in active_sat_ids:
            sid_i = int(sid)
            if sid_i not in self.hidden_states:
                self.hidden_states[sid_i] = torch.zeros(1, 1, self.hidden_dim, device=self.device)

    def _gather_hidden(self, active_sat_ids: Sequence[int]) -> torch.Tensor:
        self.sync_active_satellites(active_sat_ids)
        return torch.cat([self.hidden_states[int(sid)] for sid in active_sat_ids], dim=1)

    def _scatter_hidden(self, active_sat_ids: Sequence[int], h_next: torch.Tensor) -> None:
        for idx, sid in enumerate(active_sat_ids):
            self.hidden_states[int(sid)] = h_next[:, idx : idx + 1, :].detach()

    @torch.no_grad()
    def act(
        self,
        active_sat_ids: Sequence[int],
        F_rich: np.ndarray,
        I_mat: np.ndarray,
        Omega: np.ndarray,
        C: np.ndarray,
        real_mask: np.ndarray,
        action_mask: np.ndarray,
        E_mat: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute beam selection (N_act, J) and power ratios (N_act, J) for current active satellites."""
        n_act = len(active_sat_ids)
        j_dim = action_mask.shape[1]
        if n_act == 0:
            return np.zeros((0, j_dim), dtype=np.float32), np.zeros((0, j_dim), dtype=np.float32)

        if F_rich.shape[-1] < self.node_dim:
            raise ValueError(
                f"MAPPO checkpoint expects node_dim={self.node_dim}, but received F_rich with dim={F_rich.shape[-1]}. "
                f"(cross_satellite={self.cross_satellite}, coop_power_gnn={self.coop_power_gnn})"
            )

        h_in = self._gather_hidden(active_sat_ids)
        f_t = torch.from_numpy(np.ascontiguousarray(F_rich[..., : self.node_dim], dtype=np.float32)).to(self.device)
        i_t = torch.from_numpy(np.ascontiguousarray(I_mat, dtype=np.float32)).to(self.device)
        om_t = torch.from_numpy(np.ascontiguousarray(Omega, dtype=np.float32)).to(self.device)
        c_t = torch.from_numpy(np.ascontiguousarray(C, dtype=np.float32)).to(self.device)
        rm_t = torch.from_numpy(np.ascontiguousarray(real_mask, dtype=np.float32)).to(self.device)
        am_t = torch.from_numpy(np.ascontiguousarray(action_mask, dtype=np.float32)).to(self.device)
        e_t = (
            torch.from_numpy(np.ascontiguousarray(E_mat, dtype=np.float32)).to(self.device)
            if (self.coop_power_gnn and E_mat is not None)
            else None
        )

        b, p, _, _, _, h_next = self.actor.act(
            f_t,
            i_t,
            om_t,
            c_t,
            rm_t,
            am_t,
            h_in,
            deterministic=deterministic,
            E_mat=e_t,
            power_deterministic=deterministic,
        )
        self._scatter_hidden(active_sat_ids, h_next)
        return b.cpu().numpy().astype(np.float32), p.cpu().numpy().astype(np.float32)


class DynamicUSWGPAdapter:
    """Dynamic N_act(t) wrapper for Chapter 3 USWGP-BH heuristic (equal power, skips empty queues)."""

    def reset(self) -> None:
        pass

    def act(
        self,
        active_sat_ids: Sequence[int],
        F_basic: np.ndarray,
        action_mask: np.ndarray,
        k_max: int = base_cfg.K_MAX,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_act, j_dim = action_mask.shape
        beam_sel = np.zeros((n_act, j_dim), dtype=np.float32)
        for n in range(n_act):
            avail = np.flatnonzero((action_mask[n] > 0.5) & (F_basic[n, :, 0] > 1e-6))
            k = min(k_max, len(avail))
            if k == 0:
                continue
            scores = 0.5 * F_basic[n, avail, 0] + 1.2 * F_basic[n, avail, 1] + 0.8 * F_basic[n, avail, 2]
            top_idx = avail[np.argsort(-scores)[:k]]
            beam_sel[n, top_idx] = 1.0
        return beam_sel, equal_power_np(beam_sel)


class DynamicQLFAdapter:
    """Dynamic N_act(t) wrapper for Chapter 3 QLF-BH heuristic."""

    def reset(self) -> None:
        pass

    def act(
        self,
        active_sat_ids: Sequence[int],
        F_basic: np.ndarray,
        action_mask: np.ndarray,
        k_max: int = base_cfg.K_MAX,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_act, j_dim = action_mask.shape
        beam_sel = np.zeros((n_act, j_dim), dtype=np.float32)
        for n in range(n_act):
            avail = np.flatnonzero((action_mask[n] > 0.5) & (F_basic[n, :, 0] > 1e-6))
            k = min(k_max, len(avail))
            if k == 0:
                continue
            scores = F_basic[n, avail, 0]
            top_idx = avail[np.argsort(-scores)[:k]]
            beam_sel[n, top_idx] = 1.0
        return beam_sel, equal_power_np(beam_sel)
