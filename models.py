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


class IterativeCoopPowerGNN(nn.Module):
    """3-step physics-informed iterative GNN over active beams (all N satellites) using beam-to-beam coupling."""

    def __init__(self, node_dim: int = cfg.RICH_NODE_DIM, hidden: int = 48, steps: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.steps = steps
        in_d = node_dim + 4
        self.enc = nn.Sequential(nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.msg_in = nn.ModuleList([nn.Linear(hidden + 3, hidden) for _ in range(steps)])
        self.msg_out = nn.ModuleList([nn.Linear(hidden + 3, hidden) for _ in range(steps)])
        self.sat_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(steps)])
        self.upd = nn.ModuleList(
            [nn.Sequential(nn.Linear(4 * hidden + 8, hidden), nn.LayerNorm(hidden), nn.ReLU()) for _ in range(steps)]
        )
        self.step_head = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(steps)]
        )

    def forward(self, F_rich: torch.Tensor, b_mask: torch.Tensor, E_mat: torch.Tensor) -> torch.Tensor:
        # F_rich: (N, J, D), b_mask: (N, J), E_mat: (N*J, N*J)
        N, J, D = F_rich.shape
        M_tot = N * J
        act = b_mask.reshape(M_tot)
        act_idx = torch.nonzero(act > 0.5, as_tuple=False).squeeze(-1)
        if act_idx.numel() == 0:
            return torch.zeros(N, J, device=F_rich.device, dtype=F_rich.dtype)

        A = act_idx.numel()
        sat_id = act_idx // J
        x = F_rich.reshape(M_tot, D)[act_idx, : self.node_dim]
        log_q = torch.log1p(15.0 * x[:, :4].clamp(min=0.0))
        rt_pkts = (x[:, 0] * 800.0).clamp(min=0.0)
        tot_pkts = rt_pkts + (x[:, 1] * 2000.0).clamp(min=0.0)
        dem = (x[:, 0] + x[:, 1] + 0.5 * x[:, 2] + 0.5 * x[:, 3]).clamp(min=0.0)

        E = E_mat[act_idx][:, act_idx]
        eye = torch.eye(A, device=E.device, dtype=E.dtype)
        E_nd = E * (1.0 - eye)
        S = (sat_id[:, None] == sat_id[None, :]).to(E.dtype)
        S_norm = S / S.sum(dim=-1, keepdim=True).clamp(min=1.0)

        rx_sum = torch.log1p(E_nd.T.sum(dim=-1, keepdim=True))
        tx_sum = torch.log1p(E_nd.sum(dim=-1, keepdim=True))
        W_in = torch.log1p(E_nd.T)
        W_out = torch.log1p(E_nd)

        p_cur = (1.0 / S.sum(dim=-1)).clamp(min=1e-4)
        snr_lin_est = torch.pow(10.0, (x[:, 4] * 15.0 + 20.0) / 10.0).clamp(min=1.0)
        inv_snr_full = 0.25 / snr_lin_est
        pkt_per_se = cfg.B_TOT * cfg.T_SLOT / cfg.L_P  # 80 packets per bps/Hz

        h = self.enc(torch.cat([x, log_q], dim=-1))
        mu_act = torch.zeros(A, device=F_rich.device, dtype=F_rich.dtype)
        for s in range(self.steps):
            i_rx = E_nd.T @ p_cur
            sinr_est = p_cur / (inv_snr_full + i_rx).clamp(min=1e-6)
            se_est = torch.log2(1.0 + sinr_est)
            cap_est = se_est * pkt_per_se
            sat_ratio = torch.log1p(tot_pkts / (cap_est + 1.0))
            rt_ratio = torch.log1p(rt_pkts / (cap_est + 1.0))
            cap_lim = (tot_pkts / (cap_est + 1.0)).clamp(max=1.0)
            victim_sens = (cap_lim * (0.3 + dem)) / (inv_snr_full + i_rx + p_cur).clamp(min=1e-4)
            live_harm = E_nd @ victim_sens

            dyn_feat = torch.stack(
                [
                    p_cur,
                    torch.log1p(i_rx * 10.0),
                    se_est / 10.0,
                    torch.log1p(live_harm),
                    sat_ratio,
                    rt_ratio,
                ],
                dim=-1,
            )
            h_p = torch.cat([h, p_cur.unsqueeze(-1), se_est.unsqueeze(-1) / 10.0, sat_ratio.unsqueeze(-1)], dim=-1)
            m_in = W_in @ self.msg_in[s](h_p)
            m_out = W_out @ self.msg_out[s](h_p)
            m_sat = S_norm @ self.sat_proj[s](h)
            h = h + self.upd[s](torch.cat([h, m_in, m_out, m_sat, rx_sum, tx_sum, dyn_feat], dim=-1))
            mu_act = 6.5 * torch.tanh(self.step_head[s](h).squeeze(-1))

            mu_full = torch.full((M_tot,), -1e9, device=F_rich.device, dtype=F_rich.dtype)
            mu_full[act_idx] = mu_act
            p_full = F.softmax(mu_full.view(N, J), dim=-1) * b_mask
            p_full = p_full / p_full.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            p_cur = p_full.reshape(M_tot)[act_idx]

        mu = torch.zeros(M_tot, device=F_rich.device, dtype=F_rich.dtype)
        mu[act_idx] = mu_act
        mu = mu.view(N, J)
        k_sel = b_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (mu - (mu * b_mask).sum(dim=-1, keepdim=True) / k_sel) * b_mask


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
        power_conditioned: bool = False,
        coop_power_gnn: bool = False,
    ):
        super().__init__()
        self.J = J
        self.d_g = d_g
        self.hidden_dim = hidden_dim
        self.xi_1 = xi_1
        self.xi_2 = xi_2
        self.power_conditioned = power_conditioned
        self.coop_power_gnn = coop_power_gnn
        self.log_sigma_min = math.log(0.10 if (power_conditioned or coop_power_gnn) else 0.12)
        self.log_sigma_max = math.log(1.20 if (power_conditioned or coop_power_gnn) else 0.85)

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
        if power_conditioned:
            # Extra local interference-demand features for beam selection, and beam-conditioned features for power
            self.node_disc_head = nn.Sequential(
                nn.Linear(in_head + 2, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1)
            )
            self.node_power_head = nn.Sequential(
                nn.Linear(in_head + 5 + d_g, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1)
            )
            self.cont_std_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.cont_std_head.weight)
            nn.init.constant_(self.cont_std_head.bias, 0.2)
        else:
            self.node_disc_head = nn.Sequential(
                nn.Linear(in_head, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            self.node_power_head = nn.Sequential(
                nn.Linear(in_head, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            self.cont_std_head = nn.Linear(hidden_dim, 1)
        if coop_power_gnn:
            self.coop_gnn = IterativeCoopPowerGNN(node_dim=cfg.RICH_NODE_DIM, hidden=48, steps=3)
        else:
            self.coop_gnn = None
        # Learnable sequential interference repulsion weights using exact I_mat, Omega, and C priors
        self.repulse_imat = nn.Parameter(torch.tensor(1.2))
        self.repulse_omega = nn.Parameter(torch.tensor(-0.5))
        self.repulse_cov = nn.Parameter(torch.tensor(-3.0))

    def power_parameters(self):
        if self.coop_gnn is not None:
            return list(self.coop_gnn.parameters())
        return list(self.node_power_head.parameters())

    def discrete_parameters(self):
        power_ids = {id(p) for p in self.power_parameters()}
        return [p for p in self.parameters() if id(p) not in power_ids]

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

    def _encode_nodes(self, F_in, I_mat, Omega, C, real_mask, h_in):
        B, J, _ = F_in.shape
        o_hat = self._encode_gat(F_in, I_mat, Omega, C, real_mask)  # (B, J, d_g)
        pool_ctx = (o_hat * real_mask.unsqueeze(-1)).sum(dim=1) / real_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sp = F.relu(self.spatial_fc(pool_ctx))  # (B, 128)
        gru_out, h_next = self.gru(sp.unsqueeze(1), h_in)
        feat = F.relu(self.feat_fc(gru_out.squeeze(1)))  # (B, 128)
        ctx_b = self.ctx_proj(feat).unsqueeze(1).expand(B, J, -1)
        log_q = torch.log1p(15.0 * F_in[..., :4].clamp(min=0.0))
        node_cat = torch.cat([F_in, log_q, o_hat, ctx_b], dim=-1)
        return node_cat, o_hat, feat, h_next

    def _disc_logits(self, node_cat, F_in, I_mat):
        if not self.power_conditioned:
            return self.node_disc_head(node_cat).squeeze(-1)
        eye = torch.eye(self.J, device=I_mat.device, dtype=I_mat.dtype).unsqueeze(0)
        I_nd = I_mat * (1.0 - eye)
        dem = (F_in[..., 0] + F_in[..., 1] + 0.5 * F_in[..., 2] + 0.5 * F_in[..., 3]).clamp(min=0.0)
        rx_dem = torch.log1p(torch.bmm(dem.unsqueeze(1), I_nd).squeeze(1).clamp(min=0.0))
        tx_dem = torch.log1p(torch.bmm(I_nd, dem.unsqueeze(2)).squeeze(2).clamp(min=0.0))
        disc_in = torch.cat([node_cat, rx_dem.unsqueeze(-1), tx_dem.unsqueeze(-1)], dim=-1)
        return self.node_disc_head(disc_in).squeeze(-1)

    def _power_params(self, node_cat, o_hat, F_in, I_mat, feat, b_mask=None):
        B, J, _ = F_in.shape
        if not self.power_conditioned:
            mu = 4.0 * torch.tanh(self.node_power_head(node_cat).squeeze(-1))
            log_std = self.log_sigma_min + 0.5 * (self.log_sigma_max - self.log_sigma_min) * (
                torch.tanh(self.cont_std_head(feat) - 0.8) + 1.0
            )
            return mu, log_std.exp().expand(B, J)

        if b_mask is None:
            b_mask = ((F_in[..., 0] + F_in[..., 1]) > 1e-6).to(F_in.dtype)
        eye = torch.eye(self.J, device=I_mat.device, dtype=I_mat.dtype).unsqueeze(0)
        I_nd = I_mat * (1.0 - eye)
        dem = (F_in[..., 0] + F_in[..., 1] + 0.5 * F_in[..., 2] + 0.5 * F_in[..., 3]).clamp(min=0.0)
        rt_urg = (F_in[..., 0] + F_in[..., 2] + F_in[..., 3]).clamp(min=0.0)
        k_sel = b_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        rx_sel = torch.log1p(torch.bmm(b_mask.unsqueeze(1), I_nd).squeeze(1).clamp(min=0.0))
        tx_sel = torch.log1p(torch.bmm(I_nd, b_mask.unsqueeze(2)).squeeze(2).clamp(min=0.0))
        tx_sel_dem = torch.log1p(torch.bmm(I_nd, (b_mask * dem).unsqueeze(2)).squeeze(2).clamp(min=0.0))
        sel_dem_frac = (dem * b_mask) / (dem * b_mask).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        sel_rt_frac = (rt_urg * b_mask) / (rt_urg * b_mask).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        sel_ctx = ((o_hat * b_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / k_sel.unsqueeze(-1)).expand(B, J, -1)
        pow_in = torch.cat(
            [
                node_cat,
                rx_sel.unsqueeze(-1),
                tx_sel.unsqueeze(-1),
                tx_sel_dem.unsqueeze(-1),
                sel_dem_frac.unsqueeze(-1),
                sel_rt_frac.unsqueeze(-1),
                sel_ctx,
            ],
            dim=-1,
        )
        mu_raw = 6.5 * torch.tanh(self.node_power_head(pow_in).squeeze(-1))
        mu_mean = (mu_raw * b_mask).sum(dim=-1, keepdim=True) / k_sel
        mu = (mu_raw - mu_mean) * b_mask
        log_std = self.log_sigma_min + 0.5 * (self.log_sigma_max - self.log_sigma_min) * (
            torch.tanh(self.cont_std_head(feat)) + 1.0
        )
        return mu, log_std.exp().expand(B, J)

    def compute_power_mu(self, F_in, I_mat, Omega, C, real_mask, h_in, b_mask, E_mat=None, detach_backbone=True):
        if self.coop_gnn is not None and E_mat is not None:
            return self.coop_gnn(F_in, b_mask, E_mat)
        if detach_backbone:
            with torch.no_grad():
                node_cat, o_hat, feat, _ = self._encode_nodes(F_in, I_mat, Omega, C, real_mask, h_in)
            node_cat, o_hat, feat = node_cat.detach(), o_hat.detach(), feat.detach()
        else:
            node_cat, o_hat, feat, _ = self._encode_nodes(F_in, I_mat, Omega, C, real_mask, h_in)
        mu, _ = self._power_params(node_cat, o_hat, F_in, I_mat, feat, b_mask=b_mask)
        return mu

    def forward_step(self, F_in, I_mat, Omega, C, real_mask, h_in, b_mask=None, E_mat=None):
        node_cat, o_hat, feat, h_next = self._encode_nodes(F_in, I_mat, Omega, C, real_mask, h_in)
        disc_logits = self._disc_logits(node_cat, F_in, I_mat)
        if self.coop_gnn is not None and E_mat is not None and b_mask is not None:
            mu = self.coop_gnn(F_in, b_mask, E_mat)
            std = torch.full_like(mu, 0.15)
        else:
            mu, std = self._power_params(node_cat, o_hat, F_in, I_mat, feat, b_mask=b_mask)
        return disc_logits, mu, std, h_next

    def _eff_mask(self, F_in: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        has_q = (F_in[..., 0] + F_in[..., 1]) > 1e-6
        return action_mask * has_q.to(action_mask.dtype)

    @torch.no_grad()
    def act(
        self,
        F_in,
        I_mat,
        Omega,
        C,
        real_mask,
        action_mask,
        h_in,
        deterministic: bool = False,
        E_mat=None,
        power_deterministic: bool = False,
    ):
        node_cat, o_hat, feat, h_next = self._encode_nodes(F_in, I_mat, Omega, C, real_mask, h_in)
        disc_logits = self._disc_logits(node_cat, F_in, I_mat)
        eff_mask = self._eff_mask(F_in, action_mask)
        pen = self._pair_penalty(I_mat, Omega, C)
        j_seq, b_mask, lp_d, _ = sample_without_replacement_batch(
            disc_logits, eff_mask, cfg.K_MAX, deterministic=deterministic, pair_penalty=pen
        )
        if self.coop_gnn is not None and E_mat is not None:
            mu = self.coop_gnn(F_in, b_mask, E_mat)
            std = torch.full_like(mu, 0.15)
        else:
            mu, std = self._power_params(node_cat, o_hat, F_in, I_mat, feat, b_mask=b_mask)
        if deterministic or power_deterministic or (self.coop_gnn is not None and E_mat is not None):
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

    def evaluate_discrete_actions(self, F_in, I_mat, Omega, C, real_mask, action_mask, h_in, j_seq, return_hidden=False):
        node_cat, _, _, h_next = self._encode_nodes(F_in, I_mat, Omega, C, real_mask, h_in)
        disc_logits = self._disc_logits(node_cat, F_in, I_mat)
        eff_mask = self._eff_mask(F_in, action_mask)
        pen = self._pair_penalty(I_mat, Omega, C)
        lp_d, ent_d = evaluate_without_replacement_batch(disc_logits, eff_mask, j_seq, pair_penalty=pen)
        return (lp_d, ent_d, h_next) if return_hidden else (lp_d, ent_d)

    def evaluate_actions(self, F_in, I_mat, Omega, C, real_mask, action_mask, h_in, j_seq, y, b_mask, return_hidden=False):
        disc_logits, mu, std, h_next = self.forward_step(F_in, I_mat, Omega, C, real_mask, h_in, b_mask=b_mask)
        eff_mask = self._eff_mask(F_in, action_mask)
        pen = self._pair_penalty(I_mat, Omega, C)
        lp_d, ent_d = evaluate_without_replacement_batch(disc_logits, eff_mask, j_seq, pair_penalty=pen)
        dist = torch.distributions.Normal(mu, std)
        lp_c = (dist.log_prob(y) * b_mask).sum(dim=-1)
        ent_c = (dist.entropy() * b_mask).sum(dim=-1)
        outputs = (lp_d + lp_c, ent_d, ent_c)
        return (*outputs, h_next) if return_hidden else outputs


class MAPPOCritic(nn.Module):
    """Per-satellite CTDE Centralized Critic taking global state s_t and per-satellite rich features."""

    def __init__(self, state_dim: int = 5 * cfg.M, local_dim: int = 2 * cfg.RICH_NODE_DIM, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.global_enc = nn.Sequential(
                nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
            )
            self.local_enc = nn.Sequential(nn.Linear(local_dim, 64), nn.LayerNorm(64), nn.ReLU())
            self.sat_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(64)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
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
        if self.use_attention:
            attn_out, _ = self.sat_attn(loc, loc, loc, need_weights=False)
            loc = self.attn_norm(loc + attn_out)
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
