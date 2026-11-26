# -*- coding: utf-8 -*-
"""Discrete Action-Masked Multi-Agent PPO (MAPPO-HO) baseline with trigger-step loss masking.

Cross-chapter on-policy RL baseline adapted from the Chapter 3 MAPPO CTDE paradigm:
  - Actor: parameter-shared permutation-equivariant discrete Categorical policy over
    {Stay} U {C_MAX candidate satellites} using the exact same local CellGraphObs
    flat-MLP encoder and dynamic Action Mask as Cell-RL.
  - Centralized Critic: takes per-cell local state features + global network satellite load
    statistics (rho, zeta, slot utilization, trigger ratios, RT urgency) to estimate V(s_t, m).
  - Trigger-Step Loss Mask: GAE is computed over the full 1 s macro trajectory (including
    non-triggered forced-Stay steps), whereas PPO-Clip policy loss and entropy regularization
    backpropagate ONLY through triggered decision steps (i_trig == 1) so non-triggered Stay
    steps do not drown policy gradients or distort entropy estimates.
"""

import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from handover.baselines import BaseHandoverPolicy
from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv, EnvStepOutput
from handover.graph import (
    SAT_FEAT_DIM,
    CELL_FEAT_DIM,
    COV_EDGE_DIM,
    SERV_EDGE_DIM,
    CellGraphObs,
    build_cell_graph_obs_batch,
    collate_cell_graphs,
)
from handover.reward import compute_step_rewards


GLOBAL_CRITIC_DIM = 12
CELL_CRITIC_DIM = 14
CRITIC_IN_DIM = CELL_CRITIC_DIM + GLOBAL_CRITIC_DIM


def extract_critic_features(out: EnvStepOutput, env: DynamicLEOHandoverEnv) -> np.ndarray:
    """Build vectorized (M, CRITIC_IN_DIM) centralized-critic features for all M cells in O(M)."""
    cfg = env.cfg
    snap = out.snap
    trig = out.trigger
    m_cnt = cfg.M
    n_act = len(snap.active_sat_ids)
    assert env.traffic is not None

    sat_counts = out.Z.sum(axis=1).astype(np.float32) if n_act > 0 else np.zeros(0, dtype=np.float32)
    j_cap = float(cfg.J)

    # Global network satellite load & trigger summary (12 dims)
    g = np.zeros(GLOBAL_CRITIC_DIM, dtype=np.float32)
    if n_act > 0:
        g[0] = float(np.clip(out.sat_rho.mean() / 2.0, 0.0, 2.5))
        g[1] = float(np.clip(out.sat_rho.std(), 0.0, 2.5))
        g[2] = float(np.clip(out.sat_rho.max() / 2.5, 0.0, 2.5))
        g[3] = float(np.clip(out.sat_zeta.mean() / 2.0, 0.0, 2.5))
        g[4] = float(np.clip(out.sat_zeta.max() / 2.5, 0.0, 2.5))
        g[5] = float(sat_counts.mean() / j_cap)
        g[6] = float(np.mean(sat_counts >= j_cap))
        g[7] = float(np.mean(sat_counts >= max(1.0, j_cap - 1.0)))
    g[8] = float(trig.i_trig.mean())
    g[9] = float(trig.i_force.mean())
    g[10] = float(env.cell_rt_urg.mean())
    g[11] = float(env.cell_recent_rt_drop.mean())

    # Per-cell local + current-serving-satellite state (14 dims)
    tot_bps, _, _ = env.traffic_model.get_offered_rates_bps(out.t)
    c_feat = np.zeros((m_cnt, CELL_CRITIC_DIM), dtype=np.float32)
    c_feat[:, 0] = np.minimum(env.traffic.cell_user_counts.astype(np.float32) / 8.0, 2.0)
    c_feat[:, 1] = np.minimum(tot_bps / 150e6, 2.5).astype(np.float32)
    c_feat[:, 2] = env.traffic.cell_rt_frac.astype(np.float32)
    c_feat[:, 3] = env.cell_rt_urg.astype(np.float32)
    c_feat[:, 4] = trig.i_trig.astype(np.float32)
    c_feat[:, 5] = trig.i_force.astype(np.float32)
    c_feat[:, 6] = np.minimum(trig.dwell_s.astype(np.float32) / 60.0, 2.0)
    c_feat[:, 7] = np.clip(env.cell_hist_tput_ratio, 0.0, 1.5).astype(np.float32)
    c_feat[:, 8] = np.clip(env.cell_recent_rt_drop, 0.0, 1.0).astype(np.float32)

    sat_id_to_local = {int(sid): i for i, sid in enumerate(snap.active_sat_ids)}
    cur_loc = np.array(
        [sat_id_to_local.get(int(sid), -1) if sid >= 0 else -1 for sid in out.cell_sat_id],
        dtype=np.int32,
    )
    valid = cur_loc >= 0
    if np.any(valid):
        vm = np.flatnonzero(valid)
        vs = cur_loc[vm]
        vis_ok = snap.V_geo[vs, vm] == 1
        vm = vm[vis_ok]
        vs = vs[vis_ok]
        if len(vm) > 0:
            c_feat[vm, 9] = np.clip(snap.rvt_s[vs, vm] / 150.0, 0.0, 1.5).astype(np.float32)
            c_feat[vm, 10] = np.clip(out.sat_rho[vs] / 2.5, 0.0, 2.0).astype(np.float32)
            c_feat[vm, 11] = np.clip(out.sat_zeta[vs] / 2.5, 0.0, 2.0).astype(np.float32)
            c_feat[vm, 12] = (sat_counts[vs] / j_cap).astype(np.float32)
            c_feat[vm, 13] = np.clip((out.links["nominal_sinr_db"][vs, vm] - 8.0) / 15.0, -2.0, 2.0).astype(np.float32)

    g_rep = np.broadcast_to(g[None, :], (m_cnt, GLOBAL_CRITIC_DIM))
    return np.concatenate([c_feat, g_rep], axis=-1).astype(np.float32)


class MAPPOActorNet(nn.Module):
    """Permutation-equivariant discrete action-masked Actor matching Cell-RL's local encoder."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sat_enc = nn.Sequential(nn.Linear(SAT_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.cell_enc = nn.Sequential(nn.Linear(CELL_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.flat_ctx = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        head_in = hidden_dim * 2 + COV_EDGE_DIM + SERV_EDGE_DIM + 1
        self.logit_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_sat = batch["x_sat"]               # (B, C, F_s)
        sat_mask = batch["sat_mask"]         # (B, C)
        x_cell = batch["x_cell"]             # (B, K, F_c)
        cell_mask = batch["cell_mask"]       # (B, K)
        e_cov = batch["e_cov"]               # (B, C, K, E_cov)
        e_serv = batch["e_serv"]             # (B, C, K, E_serv)
        cur_pos = batch["cur_sat_cand_pos"]  # (B,)

        B, C, _ = x_sat.shape
        h_sat = self.sat_enc(x_sat) * sat_mask.unsqueeze(-1)
        h_cell = self.cell_enc(x_cell) * cell_mask.unsqueeze(-1)
        sat_pool = h_sat.sum(dim=1) / sat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_self = self.flat_ctx(torch.cat([h_cell[:, 0, :], sat_pool], dim=-1))

        # Action 1..C: switch to candidate satellite c
        h_self_exp = h_self.unsqueeze(1).expand(B, C, self.hidden_dim)
        cov_self = e_cov[:, :, 0, :]
        serv_self = e_serv[:, :, 0, :]
        flag_ho = torch.zeros((B, C, 1), device=x_sat.device, dtype=x_sat.dtype)
        cand_in = torch.cat([h_self_exp, h_sat, cov_self, serv_self, flag_ho], dim=-1)

        # Action 0: Stay on current serving satellite
        has_cur = (cur_pos >= 0).unsqueeze(-1).to(x_sat.dtype)
        safe_pos = cur_pos.clamp(min=0)
        b_idx = torch.arange(B, device=x_sat.device)
        h_cur_sat = h_sat[b_idx, safe_pos] * has_cur
        cov_cur = cov_self[b_idx, safe_pos] * has_cur
        serv_cur = serv_self[b_idx, safe_pos] * has_cur
        flag_stay = torch.ones((B, 1), device=x_sat.device, dtype=x_sat.dtype)
        stay_in = torch.cat([h_self, h_cur_sat, cov_cur, serv_cur, flag_stay], dim=-1).unsqueeze(1)

        act_feat = torch.cat([stay_in, cand_in], dim=1)  # (B, 1 + C, head_in)
        return self.logit_head(act_feat).squeeze(-1)     # (B, 1 + C)


class MAPPOCriticNet(nn.Module):
    """Centralized value network V(s_global, o_cell) shared across cells."""

    def __init__(self, in_dim: int = CRITIC_IN_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_masked_ppo_actor_loss(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    trigger_mask: torch.Tensor,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trigger-masked PPO-Clip policy loss and entropy bonus.

    Non-triggered steps (trigger_mask == 0) contribute strictly zero gradient to both
    the clipped surrogate objective and the entropy regularizer.
    """
    masked_logits = logits.masked_fill(action_mask < 0.5, -1e9)
    dist = Categorical(logits=masked_logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    log_ratio = (new_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_elem = -torch.minimum(surr1, surr2)

    tm = trigger_mask.to(logits.dtype)
    denom = tm.sum().clamp(min=1.0)
    pg_loss = (pg_elem * tm).sum() / denom
    ent_mean = (entropy * tm).sum() / denom
    total_actor_loss = pg_loss - ent_coef * ent_mean
    return total_actor_loss, pg_loss.detach(), ent_mean.detach()


class MAPPOHOAgent(BaseHandoverPolicy):
    """Parameter-shared discrete Action-Masked MAPPO handover policy."""

    name = "MAPPO-HO"
    bypass_trigger_gate = False

    def __init__(
        self,
        cfg: HandoverConfig,
        name: str = "MAPPO-HO",
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.012,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.name = name
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.device = torch.device(device)
        self.epsilon = 0.0  # kept for interface parity with DRLHandoverAgent

        self.actor = MAPPOActorNet(hidden_dim=cfg.hgat_hidden_dim).to(self.device)
        self.critic = MAPPOCriticNet(in_dim=CRITIC_IN_DIM, hidden_dim=128).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def _to_tensors(self, batch_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).to(self.device) for k, v in batch_np.items()}

    @torch.no_grad()
    def select_actions_rollout(
        self,
        out: EnvStepOutput,
        env: DynamicLEOHandoverEnv,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, list[list[int]], list[tuple[int, CellGraphObs, int, float]]]:
        """Select actions for triggered cells, optionally sampling stochastically for PPO rollouts."""
        m_cnt = self.cfg.M
        prop = np.full(m_cnt, -1, dtype=np.int32)
        fallbacks: list[list[int]] = [[] for _ in range(m_cnt)]
        trig_records: list[tuple[int, CellGraphObs, int, float]] = []

        trig_cells = np.flatnonzero(out.trigger.i_trig > 0)
        if len(trig_cells) == 0:
            return prop, fallbacks, trig_records

        obs_list = build_cell_graph_obs_batch(trig_cells, out, env, traffic_aware=True)
        batch_t = self._to_tensors(collate_cell_graphs(obs_list))
        self.actor.eval()
        logits = self.actor(batch_t)
        mask = batch_t["action_mask"]
        masked_logits = logits.masked_fill(mask < 0.5, -1e9)
        dist = Categorical(logits=masked_logits)

        if deterministic:
            acts_t = masked_logits.argmax(dim=-1)
        else:
            acts_t = dist.sample()
        logp_t = dist.log_prob(acts_t)

        acts_np = acts_t.cpu().numpy()
        logp_np = logp_t.cpu().numpy()
        mlogits_np = masked_logits.cpu().numpy()

        for idx, raw_m in enumerate(trig_cells):
            m = int(raw_m)
            obs_m = obs_list[idx]
            act = int(acts_np[idx])
            trig_records.append((m, obs_m, act, float(logp_np[idx])))

            if act == 0:
                prop[m] = -1
            else:
                prop[m] = int(obs_m.cand_local_sats[act - 1])

            cand_scores = mlogits_np[idx, 1:]
            cand_order = np.argsort(-cand_scores)
            fb = [int(obs_m.cand_local_sats[c]) for c in cand_order if obs_m.cand_local_sats[c] >= 0]
            vis = np.flatnonzero(out.snap.V_geo[:, m] == 1)
            fb_set = set(fb)
            extra = [int(n) for n in vis if int(n) not in fb_set]
            extra.sort(key=lambda n: -float(out.snap.rvt_s[n, m]))
            fallbacks[m] = fb + extra

        return prop, fallbacks, trig_records

    def select_actions(
        self, out: EnvStepOutput, env: DynamicLEOHandoverEnv
    ) -> tuple[np.ndarray, list[list[int]] | None]:
        prop, fb, _ = self.select_actions_rollout(out, env, deterministic=True)
        return prop, fb

    @torch.no_grad()
    def predict_values(self, critic_feat_np: np.ndarray) -> np.ndarray:
        self.critic.eval()
        x = torch.from_numpy(critic_feat_np).to(self.device)
        return self.critic(x).cpu().numpy()

    def ppo_update(
        self,
        trig_obs_list: list[CellGraphObs],
        trig_actions: np.ndarray,
        trig_old_logp: np.ndarray,
        trig_adv: np.ndarray,
        critic_feats: np.ndarray,
        critic_returns: np.ndarray,
        critic_old_vals: np.ndarray,
        rng: np.random.Generator,
        ppo_epochs: int = 4,
        actor_minibatch: int = 256,
        critic_minibatch: int = 1024,
        ent_coef: float | None = None,
    ) -> tuple[float, float, float]:
        """Run multi-epoch PPO-Clip update on triggered Actor steps and full-trajectory Critic steps."""
        ec = self.ent_coef if ent_coef is None else float(ent_coef)

        # Normalize advantages over triggered decision steps only
        adv = trig_adv.astype(np.float32).copy()
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        n_trig = len(trig_obs_list)
        actor_losses = []
        entropies = []
        if n_trig > 0:
            self.actor.train()
            for _ in range(ppo_epochs):
                perm = rng.permutation(n_trig)
                for start in range(0, n_trig, actor_minibatch):
                    idx = perm[start : start + actor_minibatch]
                    sub_obs = [trig_obs_list[i] for i in idx]
                    batch_t = self._to_tensors(collate_cell_graphs(sub_obs))
                    act_t = torch.from_numpy(trig_actions[idx]).to(self.device)
                    old_lp_t = torch.from_numpy(trig_old_logp[idx]).to(self.device)
                    adv_t = torch.from_numpy(adv[idx]).to(self.device)
                    trig_mask_t = torch.ones_like(adv_t)

                    logits = self.actor(batch_t)
                    loss_a, pg_l, ent_m = compute_masked_ppo_actor_loss(
                        logits=logits,
                        action_mask=batch_t["action_mask"],
                        actions=act_t,
                        old_log_probs=old_lp_t,
                        advantages=adv_t,
                        trigger_mask=trig_mask_t,
                        clip_eps=self.clip_eps,
                        ent_coef=ec,
                    )
                    self.actor_opt.zero_grad()
                    loss_a.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_opt.step()
                    actor_losses.append(float(pg_l.item()))
                    entropies.append(float(ent_m.item()))

        # Critic update over all (T * M) trajectory steps with PPO value clipping
        n_crit = critic_feats.shape[0]
        critic_losses = []
        if n_crit > 0:
            self.critic.train()
            cf_t_all = torch.from_numpy(critic_feats).to(self.device)
            ret_t_all = torch.from_numpy(critic_returns.astype(np.float32)).to(self.device)
            ov_t_all = torch.from_numpy(critic_old_vals.astype(np.float32)).to(self.device)
            for _ in range(ppo_epochs):
                perm_c = rng.permutation(n_crit)
                for start in range(0, n_crit, critic_minibatch):
                    idx = perm_c[start : start + critic_minibatch]
                    v_pred = self.critic(cf_t_all[idx])
                    ret_b = ret_t_all[idx]
                    ov_b = ov_t_all[idx]
                    v_clipped = ov_b + (v_pred - ov_b).clamp(-self.clip_eps, self.clip_eps)
                    loss_v = 0.5 * torch.maximum((v_pred - ret_b) ** 2, (v_clipped - ret_b) ** 2).mean()
                    self.critic_opt.zero_grad()
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_opt.step()
                    critic_losses.append(float(loss_v.item()))

        return (
            float(np.mean(actor_losses)) if actor_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
        )

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "name": self.name,
            "algo_type": "mappo_ho",
            "clip_eps": self.clip_eps,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        if "critic_state_dict" in ckpt:
            self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor.eval()
        self.critic.eval()
        return ckpt


def evaluate_mappo_ho(
    agent: MAPPOHOAgent,
    cfg: HandoverConfig,
    seeds: tuple[int, ...] = (9001, 9002),
    duration_s: int = 300,
) -> tuple[float, dict[str, float]]:
    """Run deterministic validation episodes for MAPPO-HO matching train.py's validation protocol."""
    env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate", force_only_trigger=False, traffic_aware_trigger=True)
    tot_rews = []
    metric_list = []
    for sd in seeds:
        out = env.reset(seed=sd, duration_s=duration_s)
        ep_r = 0.0
        while not out.done:
            prev_out = out
            prop, fb = agent.select_actions(out, env)
            out = env.step(prop, fallback_rankings=fb)
            r_tot, _, _ = compute_step_rewards(prev_out, out, env)
            ep_r += float(r_tot.mean())
        tot_rews.append(ep_r)
        metric_list.append(env.get_episode_summary())
    avg_m = {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}
    return float(np.mean(tot_rews)), avg_m


def train_mappo_ho(
    cfg: HandoverConfig,
    episodes: int = 120,
    train_duration_s: int = 240,
    val_duration_s: int = 300,
    ppo_epochs: int = 4,
    save_dir: str = "handover/checkpoints_paper",
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train MAPPO-HO on-policy with full-trajectory GAE and trigger-step Actor loss masking."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    agent = MAPPOHOAgent(cfg=cfg, device=device)
    env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate", force_only_trigger=False, traffic_aware_trigger=True)
    actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.actor_opt, T_max=max(1, episodes), eta_min=6e-5)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.critic_opt, T_max=max(1, episodes), eta_min=1e-4)

    best_val_ret = -1e18
    best_actor_state = copy.deepcopy(agent.actor.state_dict())
    best_critic_state = copy.deepcopy(agent.critic.state_dict())
    history = []
    t0 = time.perf_counter()

    # Exact same randomized load/user distribution as train.py for fair comparison
    train_rates = [22.0, 26.0, 28.0, 32.0, 35.0]
    train_users = [180, 200, 220, 240]

    for ep in range(1, episodes + 1):
        frac = (ep - 1) / max(episodes - 1, 1)
        ent_coef = float(0.015 * (1.0 - frac) + 0.004 * frac)

        ep_rate = float(rng.choice(train_rates))
        ep_users = int(rng.choice(train_users))
        out = env.reset(seed=seed * 100 + ep, duration_s=train_duration_s, num_users=ep_users, input_rate_mbps=ep_rate)

        c_feats_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []
        rews_list: list[np.ndarray] = []
        trig_step_records: list[tuple[int, int, CellGraphObs, int, float]] = []

        step_idx = 0
        ep_rew_sum = 0.0
        while not out.done:
            prev_out = out
            cf = extract_critic_features(out, env)  # (M, 26)
            v_est = agent.predict_values(cf)        # (M,)
            prop, fb, t_recs = agent.select_actions_rollout(out, env, deterministic=False)

            out = env.step(prop, fallback_rankings=fb)
            r_tot, _, _ = compute_step_rewards(prev_out, out, env)
            ep_rew_sum += float(r_tot.mean())

            c_feats_list.append(cf)
            vals_list.append(v_est)
            rews_list.append(r_tot)
            for m, obs_m, act_m, logp_m in t_recs:
                trig_step_records.append((step_idx, m, obs_m, act_m, logp_m))
            step_idx += 1

        # Compute full-trajectory GAE(gamma, lambda) across all T steps and all M=144 cells
        T_ep = len(rews_list)
        rews_arr = np.stack(rews_list, axis=0)   # (T, M)
        vals_arr = np.stack(vals_list, axis=0)   # (T, M)
        adv_arr = np.zeros((T_ep, cfg.M), dtype=np.float32)
        gae = np.zeros(cfg.M, dtype=np.float32)
        next_v = np.zeros(cfg.M, dtype=np.float32)
        gamma = float(cfg.gamma)
        lam = float(agent.gae_lambda)
        for t in reversed(range(T_ep)):
            delta = rews_arr[t] + gamma * next_v - vals_arr[t]
            gae = delta + gamma * lam * gae
            adv_arr[t] = gae
            next_v = vals_arr[t]
        ret_arr = adv_arr + vals_arr

        # Slice out strictly triggered steps (i_trig == 1) for Actor PPO-Clip & entropy updates
        trig_obs = [rec[2] for rec in trig_step_records]
        trig_acts = np.array([rec[3] for rec in trig_step_records], dtype=np.int64)
        trig_old_lp = np.array([rec[4] for rec in trig_step_records], dtype=np.float32)
        trig_adv = np.array([adv_arr[rec[0], rec[1]] for rec in trig_step_records], dtype=np.float32)

        cf_flat = np.concatenate(c_feats_list, axis=0)      # (T*M, 26)
        ret_flat = ret_arr.reshape(-1)                      # (T*M,)
        val_flat = vals_arr.reshape(-1)                     # (T*M,)

        pg_loss, vf_loss, ent_mean = agent.ppo_update(
            trig_obs_list=trig_obs,
            trig_actions=trig_acts,
            trig_old_logp=trig_old_lp,
            trig_adv=trig_adv,
            critic_feats=cf_flat,
            critic_returns=ret_flat,
            critic_old_vals=val_flat,
            rng=rng,
            ppo_epochs=ppo_epochs,
            ent_coef=ent_coef,
        )
        actor_sched.step()
        critic_sched.step()

        ep_sum = env.get_episode_summary()
        rec = {
            "episode": ep,
            "train_ret": ep_rew_sum,
            "loss": vf_loss,
            "pg_loss": pg_loss,
            "entropy": ent_mean,
            "n_trig_samples": len(trig_obs),
            "ho": ep_sum["total_handovers"],
            "outage": ep_sum["outage_rate"],
            "rt_drop": ep_sum["rt_drop_rate"],
            "tput": ep_sum["effective_tput_mbps"],
            "load_var": ep_sum["load_variance"],
        }

        val_interval = episodes if episodes <= 5 else (10 if episodes >= 20 else max(1, episodes // 5))
        if ep % val_interval == 0 or ep == episodes:
            val_ret, val_m = evaluate_mappo_ho(agent, cfg, seeds=(9001, 9002), duration_s=val_duration_s)
            rec["val_ret"] = val_ret
            rec["val_metrics"] = val_m
            if val_ret > best_val_ret:
                best_val_ret = val_ret
                best_actor_state = copy.deepcopy(agent.actor.state_dict())
                best_critic_state = copy.deepcopy(agent.critic.state_dict())
            if verbose:
                print(
                    f"[MAPPO-HO   ] ep {ep:3d}/{episodes} | vf={vf_loss:5.3f} ent={ent_mean:4.2f} | "
                    f"val_ret={val_ret:6.1f} | HO={val_m['total_handovers']:4.0f} "
                    f"PP={val_m['ping_pong_rate']*100:4.1f}% Out={val_m['outage_rate']*100:4.2f}% "
                    f"Drop={val_m['rt_drop_rate']*100:4.1f}% Tput={val_m['effective_tput_mbps']:5.2f} "
                    f"Var={val_m['load_variance']:.3f} | {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
        history.append(rec)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "mappo_ho.pt")
    agent.actor.load_state_dict(best_actor_state)
    agent.critic.load_state_dict(best_critic_state)
    agent.save(ckpt_path, extra={"algo_key": "mappo_ho", "best_val_ret": best_val_ret, "history": history})
    hist_path = os.path.join(save_dir, "mappo_ho_history.json")
    with open(hist_path, "w") as hf:
        json.dump(
            {"algo_key": "mappo_ho", "display": "MAPPO-HO", "best_val_ret": best_val_ret, "history": history},
            hf,
            indent=2,
        )
    if verbose:
        print(f"Saved best MAPPO-HO checkpoint -> {ckpt_path} (best_val_ret={best_val_ret:.2f})", flush=True)
    return {"algo_key": "mappo_ho", "ckpt_path": ckpt_path, "best_val_ret": best_val_ret, "history": history}
