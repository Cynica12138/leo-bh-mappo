# -*- coding: utf-8 -*-
"""TRA-HGDRL networks: 2-layer multi-relation HGAT, permutation-equivariant QR-DQN, and adaptive CVaR.

Also supports all DRL baselines and ablations via clean configuration flags:
  - TRA-HGDRL (full):      use_hgat=True,  distributional=True,  use_cvar=True
  - w/o Risk:              use_hgat=True,  distributional=True,  use_cvar=False (alpha == 1)
  - w/o DistRL / MA-HGDQN: use_hgat=True,  distributional=False, use_cvar=False (Double DQN)
  - w/o HGAT / Cell-RL:    use_hgat=False, distributional=True/False
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv, EnvStepOutput
from handover.baselines import BaseHandoverPolicy
from handover.graph import (
    SAT_FEAT_DIM,
    CELL_FEAT_DIM,
    COV_EDGE_DIM,
    SERV_EDGE_DIM,
    COMP_EDGE_DIM,
    CellGraphObs,
    build_cell_graph_obs_batch,
    collate_cell_graphs,
)


def compute_adaptive_alpha(
    rt_frac: torch.Tensor,
    rt_urg: torch.Tensor,
    beta_1: float,
    beta_2: float,
    alpha_min: float,
    cvar_blend: float = 1.0,
) -> torch.Tensor:
    """Traffic-adaptive CVaR risk level alpha_m(t) = clip(1 - beta_1*p_RT - beta_2*u_RT, alpha_min, 1).

    cvar_blend in [0, 1] enables two-stage annealing from risk-neutral (0 -> alpha=1) to full CVaR (1).
    """
    raw = (1.0 - beta_1 * rt_frac - beta_2 * rt_urg).clamp(min=alpha_min, max=1.0)
    return 1.0 - cvar_blend * (1.0 - raw)


def compute_approx_cvar(quantiles: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Continuous linear-interpolation approximate CVaR for non-integer alpha * N_q.

    Args:
        quantiles: (B, A, N_q) predicted quantile values (unsorted allowed; sorted ascending internally)
        alpha:     (B,) risk levels in (0, 1]
    Returns:
        cvar:      (B, A) approximate CVaR_alpha scores
    """
    q_sorted, _ = torch.sort(quantiles, dim=-1)  # (B, A, N_q)
    nq = q_sorted.shape[-1]
    a_nq = (alpha.clamp(min=1e-4, max=1.0) * float(nq)).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    # Weight of quantile index i (0-based): w_i = clamp(alpha*N_q - i, 0, 1) / (alpha*N_q)
    idx = torch.arange(nq, device=quantiles.device, dtype=quantiles.dtype).view(1, 1, nq)
    weights = (a_nq - idx).clamp(min=0.0, max=1.0)
    return (q_sorted * weights).sum(dim=-1) / a_nq.squeeze(-1).clamp(min=1e-6)


def quantile_huber_loss(pred_q: torch.Tensor, target_q: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """QR-DQN Quantile Huber Loss between predicted (B, N_q) and target (B, N_q) quantiles."""
    nq = pred_q.shape[-1]
    taus = (torch.arange(nq, device=pred_q.device, dtype=pred_q.dtype) + 0.5) / float(nq)  # (N_q,)
    # Pairwise TD error u_{i,j} = target_j - pred_i -> (B, N_q_pred, N_q_target)
    td_err = target_q.unsqueeze(1) - pred_q.unsqueeze(2)
    abs_u = td_err.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * td_err * td_err, kappa * (abs_u - 0.5 * kappa))
    weight = (taus.view(1, nq, 1) - (td_err.detach() < 0.0).to(pred_q.dtype)).abs()
    # Normalize over both target and predicted quantiles for stable gradient scale independent of N_q
    loss = (weight * huber).mean(dim=-1).mean(dim=-1)
    return loss.mean()


class BipartiteEdgeGAT(nn.Module):
    """Edge-conditioned graph attention from source nodes (B, N_src, D) to target nodes (B, N_dst, D)."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.w_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.w_val = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        h_dst: torch.Tensor,
        h_src: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_dst:     (B, N_dst, D)
            h_src:     (B, N_src, D)
            edge_feat: (B, N_dst, N_src, E)
            edge_mask: (B, N_dst, N_src) {0, 1}
        """
        q = self.w_dst(h_dst).unsqueeze(2)          # (B, N_dst, 1, D)
        k = self.w_src(h_src).unsqueeze(1)          # (B, 1, N_src, D)
        e = self.w_edge(edge_feat)                  # (B, N_dst, N_src, D)
        logits = self.attn_a(F.leaky_relu(q + k + e, negative_slope=0.2)).squeeze(-1)  # (B, N_dst, N_src)
        has_edge = edge_mask > 0.5
        any_edge = has_edge.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~has_edge, -1e9)
        attn = F.softmax(logits, dim=-1) * has_edge.to(logits.dtype)
        attn = torch.where(any_edge, attn, torch.zeros_like(attn))
        v = self.w_val(h_src)                       # (B, N_src, D)
        return torch.bmm(attn, v)                   # (B, N_dst, D)


class HGATLayer(nn.Module):
    """Single multi-relation heterogeneous GAT layer over Coverage, Service, and Competition subgraphs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Messages into cell nodes
        self.cov_s2c = BipartiteEdgeGAT(hidden_dim, COV_EDGE_DIM)
        self.serv_s2c = BipartiteEdgeGAT(hidden_dim, SERV_EDGE_DIM)
        self.comp_c2c = BipartiteEdgeGAT(hidden_dim, COMP_EDGE_DIM)
        # Messages into satellite nodes
        self.cov_c2s = BipartiteEdgeGAT(hidden_dim, COV_EDGE_DIM)
        self.serv_c2s = BipartiteEdgeGAT(hidden_dim, SERV_EDGE_DIM)

        self.norm_cell = nn.LayerNorm(hidden_dim)
        self.norm_sat = nn.LayerNorm(hidden_dim)
        self.gate_cell = nn.Parameter(torch.tensor(0.25))
        self.gate_sat = nn.Parameter(torch.tensor(0.25))

    def forward(self, h_sat, h_cell, e_cov, m_cov, e_serv, m_serv, e_comp, m_comp, sat_mask, cell_mask):
        # e_cov/e_serv have shape (B, C, K, E); transpose to (B, K, C, E) for sat -> cell messages
        e_cov_c = e_cov.transpose(1, 2)
        m_cov_c = m_cov.transpose(1, 2)
        e_serv_c = e_serv.transpose(1, 2)
        m_serv_c = m_serv.transpose(1, 2)

        msg_cell = (
            self.cov_s2c(h_cell, h_sat, e_cov_c, m_cov_c)
            + self.serv_s2c(h_cell, h_sat, e_serv_c, m_serv_c)
            + self.comp_c2c(h_cell, h_cell, e_comp, m_comp)
        )
        msg_sat = (
            self.cov_c2s(h_sat, h_cell, e_cov, m_cov)
            + self.serv_c2s(h_sat, h_cell, e_serv, m_serv)
        )
        h_cell_next = self.norm_cell(h_cell + self.gate_cell * F.elu(msg_cell)) * cell_mask.unsqueeze(-1)
        h_sat_next = self.norm_sat(h_sat + self.gate_sat * F.elu(msg_sat)) * sat_mask.unsqueeze(-1)
        return h_sat_next, h_cell_next


class TRAHGDRLNet(nn.Module):
    """Unified network for TRA-HGDRL, MA-HGDQN, Cell-RL, and all ablations.

    Uses a permutation-equivariant per-candidate scoring head so candidate satellite
    ordering changes never cause action-slot semantic drift.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_quantiles: int = 32,
        use_hgat: bool = True,
        distributional: bool = True,
        twin: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hgat = use_hgat
        self.distributional = distributional
        self.num_quantiles = num_quantiles if distributional else 1
        self.twin = twin

        self.sat_enc = nn.Sequential(nn.Linear(SAT_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.cell_enc = nn.Sequential(nn.Linear(CELL_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

        if use_hgat:
            self.layers = nn.ModuleList([HGATLayer(hidden_dim) for _ in range(num_layers)])
        else:
            # Flat MLP baseline (Cell-RL / w/o HGAT): aggregates local context without heterogeneous attention
            self.layers = nn.ModuleList()
            self.flat_ctx = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        # Shared permutation-equivariant scoring head for every candidate action (and Stay)
        head_in = hidden_dim * 2 + COV_EDGE_DIM + SERV_EDGE_DIM + 1

        def _make_head():
            return nn.Sequential(
                nn.Linear(head_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_quantiles),
            )

        self.q_head_1 = _make_head()
        self.q_head_2 = _make_head() if twin else None

    def _encode_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_sat = batch["x_sat"]            # (B, C, F_s)
        sat_mask = batch["sat_mask"]      # (B, C)
        x_cell = batch["x_cell"]          # (B, K, F_c)
        cell_mask = batch["cell_mask"]    # (B, K)
        e_cov = batch["e_cov"]            # (B, C, K, E_cov)
        m_cov = batch["m_cov"]            # (B, C, K)
        e_serv = batch["e_serv"]          # (B, C, K, E_serv)
        m_serv = batch["m_serv"]          # (B, C, K)
        e_comp = batch["e_comp"]          # (B, K, K, E_comp)
        m_comp = batch["m_comp"]          # (B, K, K)
        cur_pos = batch["cur_sat_cand_pos"]  # (B,) int64

        B, C, _ = x_sat.shape
        h_sat = self.sat_enc(x_sat) * sat_mask.unsqueeze(-1)
        h_cell = self.cell_enc(x_cell) * cell_mask.unsqueeze(-1)

        if self.use_hgat:
            for layer in self.layers:
                h_sat, h_cell = layer(h_sat, h_cell, e_cov, m_cov, e_serv, m_serv, e_comp, m_comp, sat_mask, cell_mask)
            h_self = h_cell[:, 0, :]  # (B, D)
        else:
            sat_pool = h_sat.sum(dim=1) / sat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_self = self.flat_ctx(torch.cat([h_cell[:, 0, :], sat_pool], dim=-1))

        # Action 1..C: switch to candidate satellite c
        h_self_exp = h_self.unsqueeze(1).expand(B, C, self.hidden_dim)
        cov_self = e_cov[:, :, 0, :]     # (B, C, E_cov)
        serv_self = e_serv[:, :, 0, :]   # (B, C, E_serv)
        flag_ho = torch.zeros((B, C, 1), device=x_sat.device, dtype=x_sat.dtype)
        cand_in = torch.cat([h_self_exp, h_sat, cov_self, serv_self, flag_ho], dim=-1)  # (B, C, head_in)

        # Action 0: Stay on current serving satellite (gather embedding if present in cand list, else zero)
        has_cur = (cur_pos >= 0).unsqueeze(-1).to(x_sat.dtype)  # (B, 1)
        safe_pos = cur_pos.clamp(min=0)
        b_idx = torch.arange(B, device=x_sat.device)
        h_cur_sat = h_sat[b_idx, safe_pos] * has_cur
        cov_cur = cov_self[b_idx, safe_pos] * has_cur
        serv_cur = serv_self[b_idx, safe_pos] * has_cur
        flag_stay = torch.ones((B, 1), device=x_sat.device, dtype=x_sat.dtype)
        stay_in = torch.cat([h_self, h_cur_sat, cov_cur, serv_cur, flag_stay], dim=-1).unsqueeze(1)  # (B, 1, head_in)

        return torch.cat([stay_in, cand_in], dim=1)  # (B, 1 + C, head_in)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        act_feat = self._encode_actions(batch)
        q1 = self.q_head_1(act_feat)  # (B, 1 + C, N_q)
        q2 = self.q_head_2(act_feat) if self.q_head_2 is not None else None
        return q1, q2


class DRLHandoverAgent(BaseHandoverPolicy):
    """Parameter-shared multi-agent DRL handover policy supporting TRA-HGDRL, MA-HGDQN, Cell-RL, and ablations."""

    def __init__(
        self,
        cfg: HandoverConfig,
        name: str = "TRA-HGDRL",
        use_hgat: bool = True,
        distributional: bool = True,
        use_cvar: bool = True,
        traffic_aware: bool = True,
        twin: bool = False,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.name = name
        self.use_hgat = use_hgat
        self.distributional = distributional
        self.use_cvar = use_cvar and distributional
        self.traffic_aware = traffic_aware
        self.twin = twin
        self.device = torch.device(device)
        self.bypass_trigger_gate = False

        self.net = TRAHGDRLNet(
            hidden_dim=cfg.hgat_hidden_dim,
            num_layers=cfg.hgat_layers,
            num_quantiles=cfg.num_quantiles,
            use_hgat=use_hgat,
            distributional=distributional,
            twin=twin,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.net).to(self.device)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        # Two-stage CVaR annealing factor (0.0 = risk-neutral warmup, 1.0 = full traffic-adaptive CVaR)
        self.cvar_blend = 1.0 if self.use_cvar else 0.0
        self.epsilon = 0.0

    def _to_tensors(self, batch_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in batch_np.items():
            t = torch.from_numpy(v).to(self.device)
            out[k] = t
        return out

    def _action_scores(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor | None,
        rt_frac: torch.Tensor,
        rt_urg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar decision scores (B, 1+C) via mean or traffic-adaptive CVaR."""
        if self.use_cvar and self.cvar_blend > 0.0 and self.traffic_aware:
            alpha = compute_adaptive_alpha(
                rt_frac, rt_urg, self.cfg.beta_rt_share, self.cfg.beta_rt_urg, self.cfg.alpha_min, self.cvar_blend
            )
            s1 = compute_approx_cvar(q1, alpha)
            if q2 is not None:
                s2 = compute_approx_cvar(q2, alpha)
                return torch.minimum(s1, s2)
            return s1
        s1 = q1.mean(dim=-1)
        if q2 is not None:
            return torch.minimum(s1, q2.mean(dim=-1))
        return s1

    @torch.no_grad()
    def select_actions_with_obs(
        self,
        out: EnvStepOutput,
        env: DynamicLEOHandoverEnv,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[list[int]], dict[int, tuple[CellGraphObs, int]]]:
        """Select actions for all triggered cells and return (proposed_local, fallbacks, triggered_decisions)."""
        m_cnt = self.cfg.M
        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]
        decisions: dict[int, tuple[CellGraphObs, int]] = {}

        trig_cells = np.flatnonzero(out.trigger.i_trig > 0)
        if len(trig_cells) == 0:
            return prop, fallbacks, decisions

        obs_list = build_cell_graph_obs_batch(trig_cells, out, env, traffic_aware=self.traffic_aware)
        batch_t = self._to_tensors(collate_cell_graphs(obs_list))
        self.net.eval()
        q1, q2 = self.net(batch_t)
        scores = self._action_scores(q1, q2, batch_t["rt_frac"], batch_t["rt_urg"])  # (B, 1+C)
        mask = batch_t["action_mask"]
        masked_scores = scores.masked_fill(mask < 0.5, -1e9).cpu().numpy()

        for idx, m in enumerate(trig_cells):
            m = int(m)
            obs_m = obs_list[idx]
            valid_acts = np.flatnonzero(obs_m.action_mask > 0.5)
            if rng is not None and self.epsilon > 0.0 and rng.random() < self.epsilon:
                act = int(rng.choice(valid_acts))
            else:
                act = int(np.argmax(masked_scores[idx]))
            decisions[m] = (obs_m, act)

            # Decode action -> local satellite index (-1 for Stay)
            if act == 0:
                prop[m] = -1
            else:
                prop[m] = int(obs_m.cand_local_sats[act - 1])

            # Build Q-value-ordered fallback ranking over all visible satellites
            cand_scores = masked_scores[idx, 1:]
            cand_order = np.argsort(-cand_scores)
            fb = [int(obs_m.cand_local_sats[c]) for c in cand_order if obs_m.cand_local_sats[c] >= 0]
            vis = np.flatnonzero(out.snap.V_geo[:, m] == 1)
            fb_set = set(fb)
            extra = [int(n) for n in vis if int(n) not in fb_set]
            extra.sort(key=lambda n: -float(out.snap.rvt_s[n, m]))
            fallbacks[m] = fb + extra

        return prop, fallbacks, decisions

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        prop, fb, _ = self.select_actions_with_obs(out, env, rng=None)
        return prop, fb

    def update(self, batch_np: dict[str, np.ndarray]) -> float:
        """Perform one Double QR-DQN (or Double DQN) gradient step using SMDP multi-step discount."""
        self.net.train()
        obs_t = self._to_tensors(batch_np["obs"])
        next_obs_t = self._to_tensors(batch_np["next_obs"])
        act_t = torch.from_numpy(batch_np["action"]).to(self.device)            # (B,)
        rew_t = torch.from_numpy(batch_np["reward"]).to(self.device)            # (B,)
        dt_t = torch.from_numpy(batch_np["delta_t"]).to(self.device)            # (B,)
        done_t = torch.from_numpy(batch_np["done"]).to(self.device)             # (B,)

        B = act_t.shape[0]
        b_idx = torch.arange(B, device=self.device)

        q1_all, q2_all = self.net(obs_t)
        q1_a = q1_all[b_idx, act_t]  # (B, N_q)
        q2_a = q2_all[b_idx, act_t] if q2_all is not None else None

        with torch.no_grad():
            # Double Q-learning: select next action a* using online network under next_action_mask.
            # For QR-DQN stability (Dabney et al.), blend unbiased expectation (0.7) + adaptive CVaR (0.3)
            # in Bellman bootstrapping so quantile estimates stay well-calibrated without recursive pessimism.
            nq1_on, nq2_on = self.net(next_obs_t)
            cvar_scores = self._action_scores(nq1_on, nq2_on, next_obs_t["rt_frac"], next_obs_t["rt_urg"])
            mean_scores = nq1_on.mean(dim=-1)
            if nq2_on is not None:
                mean_scores = torch.minimum(mean_scores, nq2_on.mean(dim=-1))
            next_scores = 0.7 * mean_scores + 0.3 * cvar_scores if self.use_cvar else mean_scores
            next_mask = next_obs_t["action_mask"]
            a_star = next_scores.masked_fill(next_mask < 0.5, -1e9).argmax(dim=-1)  # (B,)

            tq1_all, tq2_all = self.target_net(next_obs_t)
            tq1 = tq1_all[b_idx, a_star]  # (B, N_q)
            if tq2_all is not None:
                # Strict whole-distribution conservative selection (NEVER elementwise min across quantiles!)
                tq2 = tq2_all[b_idx, a_star]
                pick_q1 = (tq1.mean(dim=-1, keepdim=True) <= tq2.mean(dim=-1, keepdim=True))
                tq_next = torch.where(pick_q1, tq1, tq2)
            else:
                tq_next = tq1

            disc = (self.cfg.gamma ** dt_t).unsqueeze(-1) * (1.0 - done_t.unsqueeze(-1))
            # Bounded distributional value support [V_min, V_max] prevents outage bootstrap cliffs
            target_q = (rew_t.unsqueeze(-1) + disc * tq_next).clamp(min=-12.0, max=6.0)  # (B, N_q)

        if self.distributional:
            loss = quantile_huber_loss(q1_a, target_q, kappa=self.cfg.huber_kappa)
            if q2_a is not None:
                loss = loss + quantile_huber_loss(q2_a, target_q, kappa=self.cfg.huber_kappa)
        else:
            loss = F.smooth_l1_loss(q1_a, target_q)
            if q2_a is not None:
                loss = loss + F.smooth_l1_loss(q2_a, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()

        # Polyak soft update of target network
        tau = self.cfg.target_tau
        with torch.no_grad():
            for p_tgt, p_src in zip(self.target_net.parameters(), self.net.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)

        return float(loss.item())

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "name": self.name,
            "use_hgat": self.use_hgat,
            "distributional": self.distributional,
            "use_cvar": self.use_cvar,
            "traffic_aware": self.traffic_aware,
            "twin": self.twin,
            "cvar_blend": self.cvar_blend,
            "net_state_dict": self.net.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.target_net.load_state_dict(ckpt["net_state_dict"])
        self.cvar_blend = float(ckpt.get("cvar_blend", 1.0 if self.use_cvar else 0.0))
        self.net.eval()
        return ckpt
