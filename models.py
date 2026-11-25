# -*- coding: utf-8 -*-
"""PyTorch models for MAPPO-BH, PPO-BH, QMIX-BH, and MADDPG-BH."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


def active_backlog_mask(F_in: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Restrict candidate beams to associated cells with non-zero observed queue backlog."""
    return action_mask * (F_in[..., 0] > 1e-6).to(action_mask.dtype)


def sample_without_replacement_batch(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k_max: int = cfg.K_MAX,
    deterministic: bool = False,
    pair_penalty: torch.Tensor | None = None,
):
    """Sequential sampling without replacement from masked categorical distributions.

    If pair_penalty (B, J, J) is provided, selecting cell j_i updates remaining candidate logits
    by subtracting pair_penalty[:, j_i, :] at subsequent steps to avoid adjacent-beam collisions.
    """
    B, J = logits.shape
    device = logits.device
    avail_mask = mask > 0.5
    j_seq = torch.full((B, k_max), -1, dtype=torch.long, device=device)
    b_mask = torch.zeros((B, J), dtype=torch.float32, device=device)
    total_lp = torch.zeros(B, dtype=torch.float32, device=device)
    total_ent = torch.zeros(B, dtype=torch.float32, device=device)

    cur_avail = avail_mask.clone()
    cur_logits = logits.clone()
    batch_idx = torch.arange(B, device=device)
    for step in range(k_max):
        has_any = cur_avail.any(dim=-1)
        if not has_any.any():
            break
        masked_logits = cur_logits.masked_fill(~cur_avail, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()
        if deterministic:
            chosen = masked_logits.argmax(dim=-1)
        else:
            safe_probs = torch.where(has_any.unsqueeze(-1), probs, torch.full_like(probs, 1.0 / J))
            chosen = torch.multinomial(safe_probs, num_samples=1).squeeze(-1)

        step_lp = log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        step_ent = -(probs * log_probs.clamp(min=-30.0)).sum(dim=-1)

        total_lp = total_lp + torch.where(has_any, step_lp, torch.zeros_like(step_lp))
        total_ent = total_ent + torch.where(has_any, step_ent, torch.zeros_like(step_ent))
        j_seq[:, step] = torch.where(has_any, chosen, torch.full_like(chosen, -1))

        one_hot = F.one_hot(chosen, num_classes=J).bool()
        b_mask = b_mask + (one_hot & has_any.unsqueeze(-1)).float()
        cur_avail = cur_avail & ~(one_hot & has_any.unsqueeze(-1))

        if pair_penalty is not None:
            pen_step = pair_penalty[batch_idx, chosen]  # (B, J)
            cur_logits = cur_logits - torch.where(has_any.unsqueeze(-1), pen_step, torch.zeros_like(pen_step))

    return j_seq, b_mask, total_lp, total_ent


def evaluate_without_replacement_batch(
    logits: torch.Tensor,
    mask: torch.Tensor,
    j_seq: torch.Tensor,
    pair_penalty: torch.Tensor | None = None,
):
    """Reevaluate log-probability and entropy of stored sampled sequences j_seq (B, k_max)."""
    B, J = logits.shape
    k_max = j_seq.shape[1]
    device = logits.device
    cur_avail = mask > 0.5
    cur_logits = logits.clone()
    total_lp = torch.zeros(B, dtype=torch.float32, device=device)
    total_ent = torch.zeros(B, dtype=torch.float32, device=device)
    batch_idx = torch.arange(B, device=device)

    for step in range(k_max):
        chosen = j_seq[:, step]
        valid = chosen >= 0
        if not valid.any():
            break
        masked_logits = cur_logits.masked_fill(~cur_avail, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()
        safe_idx = chosen.clamp(min=0)
        step_lp = log_probs.gather(-1, safe_idx.unsqueeze(-1)).squeeze(-1)
        step_ent = -(probs * log_probs.clamp(min=-30.0)).sum(dim=-1)

        total_lp = total_lp + torch.where(valid, step_lp, torch.zeros_like(step_lp))
        total_ent = total_ent + torch.where(valid, step_ent, torch.zeros_like(step_ent))

        one_hot = F.one_hot(safe_idx, num_classes=J).bool()
        cur_avail = cur_avail & ~(one_hot & valid.unsqueeze(-1))

        if pair_penalty is not None:
            pen_step = pair_penalty[batch_idx, safe_idx]
            cur_logits = cur_logits - torch.where(valid.unsqueeze(-1), pen_step, torch.zeros_like(pen_step))

    return total_lp, total_ent


class MAPPOActor(nn.Module):
    """Shared permutation-equivariant MAPPO Actor with CSI/interference GAT, GRU, and hybrid heads."""

    def __init__(
        self,
        node_dim: int = cfg.RICH_NODE_DIM,
        d_g: int = cfg.D_G_RICH,
        J: int = cfg.J,
        hidden_dim: int = cfg.HIDDEN_DIM,
        xi_1: float = cfg.XI_1,
        xi_2: float = cfg.XI_2,
    ):
        super().__init__()
        self.J = J
        self.d_g = d_g
        self.hidden_dim = hidden_dim
        self.xi_1 = xi_1
        self.xi_2 = xi_2
        self.log_sigma_min = math.log(0.12)
        self.log_sigma_max = math.log(0.85)

        # Permutation-equivariant GAT-GRU actor backbone
        self.node_proj = nn.Linear(node_dim, d_g)
        self.attn_vec = nn.Linear(2 * d_g, 1, bias=False)
        self.w_attn_int = nn.Parameter(torch.tensor(1.0))
        self.spatial_fc = nn.Linear(d_g, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.feat_fc = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_proj = nn.Linear(hidden_dim, 16)

        # Shared per-node hybrid heads (strictly permutation-equivariant across candidate cells)
        in_head = node_dim + 4 + d_g + 16
        self.node_disc_head = nn.Sequential(nn.Linear(in_head, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.node_power_head = nn.Sequential(nn.Linear(in_head, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cont_std_head = nn.Linear(hidden_dim, 1)
        # Learnable sequential interference repulsion weights using exact I_mat, Omega, and C priors
        self.repulse_imat = nn.Parameter(torch.tensor(1.2))
        self.repulse_omega = nn.Parameter(torch.tensor(-0.5))
        self.repulse_cov = nn.Parameter(torch.tensor(-3.0))

    def _pair_penalty(self, I_mat: torch.Tensor, Omega: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        w_i = F.softplus(self.repulse_imat)
        w_om = F.softplus(self.repulse_omega)
        w_c = F.softplus(self.repulse_cov)
        sym_I = 0.5 * (I_mat + I_mat.transpose(-1, -2))
        eye = torch.eye(self.J, device=Omega.device, dtype=Omega.dtype).unsqueeze(0)
        return (w_i * sym_I + w_om * Omega + w_c * C) * (1.0 - eye)

    def _encode_gat(self, F_in: torch.Tensor, I_mat: torch.Tensor, Omega: torch.Tensor, C: torch.Tensor, real_mask: torch.Tensor):
        B, J, _ = F_in.shape
        v = self.node_proj(F_in)  # (B, J, d_g)
        vi = v.unsqueeze(2).expand(B, J, J, self.d_g)
        vj = v.unsqueeze(1).expand(B, J, J, self.d_g)
        e = F.leaky_relu(self.attn_vec(torch.cat([vi, vj], dim=-1)).squeeze(-1), negative_slope=0.2)
        score = e + self.w_attn_int * I_mat + self.xi_1 * Omega + self.xi_2 * C
        nbr_mask = real_mask.unsqueeze(1) > 0.5
        score = score.masked_fill(~nbr_mask, -1e9)
        psi = F.softmax(score, dim=-1) * real_mask.unsqueeze(2)
        o_hat = torch.tanh(torch.bmm(psi, v))  # (B, J, d_g)
        return o_hat

    def forward_step(self, F_in, I_mat, Omega, C, real_mask, h_in):
        B, J, _ = F_in.shape
        o_hat = self._encode_gat(F_in, I_mat, Omega, C, real_mask)  # (B, J, d_g)
        pool_ctx = (o_hat * real_mask.unsqueeze(-1)).sum(dim=1) / real_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sp = F.relu(self.spatial_fc(pool_ctx))  # (B, 128)
        gru_out, h_next = self.gru(sp.unsqueeze(1), h_in)
        feat = F.relu(self.feat_fc(gru_out.squeeze(1)))  # (B, 128)
        ctx_b = self.ctx_proj(feat).unsqueeze(1).expand(B, J, -1)

        log_q = torch.log1p(15.0 * F_in[..., :4].clamp(min=0.0))
        node_cat = torch.cat([F_in, log_q, o_hat, ctx_b], dim=-1)
        disc_logits = self.node_disc_head(node_cat).squeeze(-1)
        mu = 4.0 * torch.tanh(self.node_power_head(node_cat).squeeze(-1))
        log_std = self.log_sigma_min + 0.5 * (self.log_sigma_max - self.log_sigma_min) * (
            torch.tanh(self.cont_std_head(feat) - 0.8) + 1.0
        )
        return disc_logits, mu, log_std.exp().expand(B, J), h_next

    def _eff_mask(self, F_in: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        has_q = (F_in[..., 0] + F_in[..., 1]) > 1e-6
        return action_mask * has_q.to(action_mask.dtype)

    @torch.no_grad()
    def act(self, F_in, I_mat, Omega, C, real_mask, action_mask, h_in, deterministic: bool = False):
        disc_logits, mu, std, h_next = self.forward_step(F_in, I_mat, Omega, C, real_mask, h_in)
        eff_mask = self._eff_mask(F_in, action_mask)
        pen = self._pair_penalty(I_mat, Omega, C)
        j_seq, b_mask, lp_d, _ = sample_without_replacement_batch(
            disc_logits, eff_mask, cfg.K_MAX, deterministic=deterministic, pair_penalty=pen
        )
        if deterministic:
            y = mu
            lp_c = torch.zeros_like(lp_d)
        else:
            dist = torch.distributions.Normal(mu, std)
            y = dist.sample()
            lp_c = (dist.log_prob(y) * b_mask).sum(dim=-1)

        y_masked = y.masked_fill(b_mask < 0.5, -1e9)
        p = F.softmax(y_masked, dim=-1) * b_mask
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return b_mask, p, j_seq, y, lp_d + lp_c, h_next

    def evaluate_actions(self, F_in, I_mat, Omega, C, real_mask, action_mask, h_in, j_seq, y, b_mask):
        disc_logits, mu, std, _ = self.forward_step(F_in, I_mat, Omega, C, real_mask, h_in)
        eff_mask = self._eff_mask(F_in, action_mask)
        pen = self._pair_penalty(I_mat, Omega, C)
        lp_d, ent_d = evaluate_without_replacement_batch(disc_logits, eff_mask, j_seq, pair_penalty=pen)
        dist = torch.distributions.Normal(mu, std)
        lp_c = (dist.log_prob(y) * b_mask).sum(dim=-1)
        ent_c = (dist.entropy() * b_mask).sum(dim=-1)
        return lp_d + lp_c, ent_d, ent_c


class MAPPOCritic(nn.Module):
    """Per-satellite CTDE Centralized Critic taking global state s_t and per-satellite rich features."""

    def __init__(self, state_dim: int = 5 * cfg.M, local_dim: int = 2 * cfg.RICH_NODE_DIM):
        super().__init__()
        self.global_enc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.local_enc = nn.Sequential(nn.Linear(local_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s: torch.Tensor, F_rich: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        B, N, _, _ = F_rich.shape
        g = self.global_enc(s).unsqueeze(1).expand(B, N, -1)
        denom = am.sum(dim=2, keepdim=True).clamp(min=1.0)
        f_mean = (F_rich * am.unsqueeze(-1)).sum(dim=2) / denom
        f_max = (F_rich * am.unsqueeze(-1)).max(dim=2).values
        loc = self.local_enc(torch.cat([f_mean, f_max], dim=-1))
        return self.head(torch.cat([g, loc], dim=-1)).squeeze(-1)


class PPOBHModel(nn.Module):
    """PPO-BH: Centralized single-agent PPO outputting beam logits & power for all N satellites."""

    def __init__(self, state_dim: int = 5 * cfg.M, N: int = cfg.N, J: int = cfg.J):
        super().__init__()
        self.N, self.J = N, J
        self.actor_body = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.disc_head = nn.Linear(256, N * J)
        self.node_scorer = nn.Sequential(nn.Linear(cfg.NODE_FEAT_DIM, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cont_mean = nn.Linear(256, N * J)
        nn.init.zeros_(self.cont_mean.weight)
        nn.init.zeros_(self.cont_mean.bias)
        self.cont_logstd = nn.Parameter(torch.full((N, J), -1.2))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor, F_all: torch.Tensor):
        # s: (B, 5M), F_all: (B, N, J, 5)
        h = self.actor_body(s)
        B = s.shape[0]
        logits = 0.04 * self.disc_head(h).view(B, self.N, self.J) + self.node_scorer(F_all).squeeze(-1)
        mu = 1.5 * torch.tanh(self.cont_mean(h)).view(B, self.N, self.J)
        std = self.cont_logstd.clamp(-2.3, 0.0).exp().unsqueeze(0).expand(B, -1, -1)
        val = self.critic(s).squeeze(-1)
        return logits, mu, std, val

    @torch.no_grad()
    def act(self, s: torch.Tensor, F_all: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
        logits, mu, std, val = self.forward(s, F_all)
        logits_n, mu_n, std_n = logits.squeeze(0), mu.squeeze(0), std.squeeze(0)
        eff_mask = active_backlog_mask(F_all.squeeze(0), action_mask)
        j_seq, b_mask, lp_d, _ = sample_without_replacement_batch(
            logits_n, eff_mask, cfg.K_MAX, deterministic=deterministic
        )
        if deterministic:
            y = mu_n
            lp_c = torch.zeros_like(lp_d)
        else:
            dist = torch.distributions.Normal(mu_n, std_n)
            y = dist.sample()
            lp_c = (dist.log_prob(y) * b_mask).sum(dim=-1)
        y_masked = y.masked_fill(b_mask < 0.5, -1e9)
        p = F.softmax(y_masked, dim=-1) * b_mask
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return b_mask, p, j_seq, y, (lp_d + lp_c).sum(), val.squeeze(0)


class QMIXAgent(nn.Module):
    """Per-satellite Q-network for QMIX-BH (equal power allocation, permutation-equivariant)."""

    def __init__(self, J: int = cfg.J, node_dim: int = cfg.NODE_FEAT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ctx_proj = nn.Linear(hidden_dim, 16)
        self.node_q = nn.Sequential(nn.Linear(node_dim + 16, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, F_in: torch.Tensor, h_in: torch.Tensor):
        B, J, _ = F_in.shape
        x = F.relu(self.fc1(F_in.mean(dim=1)))
        out, h_next = self.gru(x.unsqueeze(1), h_in)
        ctx = self.ctx_proj(out.squeeze(1)).unsqueeze(1).expand(B, J, -1)
        q = self.node_q(torch.cat([F_in, ctx], dim=-1)).squeeze(-1)
        return q, h_next


class QMIXMixer(nn.Module):
    """Monotonic QMIX mixing network conditioned on global state s_t."""

    def __init__(self, N: int = cfg.N, state_dim: int = 5 * cfg.M, mix_dim: int = 32):
        super().__init__()
        self.N, self.mix_dim = N, mix_dim
        self.hyper_w1 = nn.Linear(state_dim, N * mix_dim)
        self.hyper_b1 = nn.Linear(state_dim, mix_dim)
        self.hyper_w2 = nn.Linear(state_dim, mix_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, 1))

    def forward(self, q_agents: torch.Tensor, s: torch.Tensor):
        B = q_agents.shape[0]
        w1 = torch.abs(self.hyper_w1(s)).view(B, self.N, self.mix_dim)
        b1 = self.hyper_b1(s).view(B, 1, self.mix_dim)
        hidden = F.elu(torch.bmm(q_agents.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(s)).view(B, self.mix_dim, 1)
        b2 = self.hyper_b2(s).view(B, 1, 1)
        return (torch.bmm(hidden, w2) + b2).squeeze(-1).squeeze(-1)


class MADDPGActor(nn.Module):
    """Per-satellite actor for MADDPG-BH (equal power allocation, permutation-equivariant)."""

    def __init__(self, J: int = cfg.J, node_dim: int = cfg.NODE_FEAT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.ctx_net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
        )
        self.node_net = nn.Sequential(nn.Linear(node_dim + 16, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, F_in: torch.Tensor):
        B, J, _ = F_in.shape
        ctx = self.ctx_net(F_in.mean(dim=1)).unsqueeze(1).expand(B, J, -1)
        return self.node_net(torch.cat([F_in, ctx], dim=-1)).squeeze(-1)


class MADDPGCritic(nn.Module):
    """Centralized critic for MADDPG-BH taking (s_t, bounded actions in [0,1])."""

    def __init__(self, state_dim: int = 5 * cfg.M, act_dim: int = cfg.N * cfg.J):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor, a_flat: torch.Tensor):
        return self.net(torch.cat([s, a_flat], dim=-1)).squeeze(-1)
