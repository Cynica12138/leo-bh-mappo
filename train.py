# -*- coding: utf-8 -*-
"""Train MAPPO-BH, PPO-BH, QMIX-BH, and MADDPG-BH (BC warm-start + genuine RL fine-tuning)."""

import argparse
import math
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from env import LEOBHEnv
from coordination import FEATURE_COUNT
from schedule_search import SlotService, best_global_swap
from models import (
    MAPPOActor,
    MAPPOCritic,
    PPOBHModel,
    QMIXAgent,
    QMIXMixer,
    MADDPGActor,
    MADDPGCritic,
    evaluate_without_replacement_batch,
)


def compute_gae(rewards, values, next_value, gamma=cfg.GAMMA, lam=cfg.LAMBDA_GAE):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


def compute_gae_vec(rewards, values, next_value, gamma=cfg.GAMMA, lam=cfg.LAMBDA_GAE):
    """Vectorized per-satellite GAE for mixed local+global credit assignment."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        vn = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * vn - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def compute_rich_teacher_targets(obs: dict):
    """CSI + RT/BULK + interference-aware initial targets for MAPPO-BH BC warm-start."""
    F = obs["F_rich"]   # [rt_q, bulk_q, avg_d, hol_w, snr_norm, hist_tput, v_local, user_cnt]
    I_mat = obs["I_mat"]
    am = obs["action_mask"]
    N, J = am.shape
    sc = np.zeros((N, J), dtype=np.float32)
    pw = np.zeros((N, J), dtype=np.float32)

    for n in range(N):
        avail = np.where((am[n] > 0.5) & ((F[n, :, 0] + F[n, :, 1]) > 1e-6))[0]
        if len(avail) == 0:
            continue
        rt_q = F[n, :, 0]
        bulk_q = F[n, :, 1]
        d = F[n, :, 2]
        hol = F[n, :, 3]
        snr = F[n, :, 4]
        base = 2.6 * rt_q + 1.4 * bulk_q + 4.0 * d + 2.5 * hol + cfg.SNR_PRIOR_W * snr
        demand = rt_q + 0.6 * bulk_q + d
        sym_I = 0.5 * (I_mat[n] + I_mat[n].T)
        s = base - 0.6 * (sym_I @ demand)
        if F.shape[-1] > cfg.RICH_NODE_DIM:
            s += 0.75 * F[n, :, 10] - 1.25 * F[n, :, 9] - 0.5 * F[n, :, 8]
        sc[n, avail] = np.clip(s[avail], -3.0, 12.0)

        w = np.maximum(0.8 * rt_q[avail] + 0.6 * bulk_q[avail] + 2.5 * d[avail] + 1.5 * hol[avail], 0.015)
        p_log = 0.75 * np.log(w) + cfg.SNR_POW_W * snr[avail] + 0.4
        pw[n, avail] = np.clip(p_log, -3.5, 3.5)
    return sc, pw


def extract_slot_physics_snapshot(
    env: LEOBHEnv, beam_sel: np.ndarray, power_ratios: np.ndarray, rt_weight: float = 0.5, compute_swaps: bool = False
):
    """Extract differentiable channel/PF snapshot and optional COMA per-beam swap advantages for CTDE training."""
    model = SlotService(env)
    watts = power_ratios.astype(np.float64).ravel() * cfg.P_TOT_W
    rx0 = watts @ model.h
    sig0 = watts[model.user_beam] * model.direct_gain
    se0 = np.log2(1.0 + sig0 / (model.noise + np.maximum(rx0 - sig0, 0.0)))

    # Exact PF RB bandwidth scale per user at current schedule: cap[u] = se[u] * rb_scale[u]
    rb_scale = np.zeros(env.U, dtype=np.float64)
    rb_bw = cfg.B_TOT / cfg.N_RB
    for size, users, hist in model.pf_groups:
        if size == 1:
            rb_scale[users.ravel()] = cfg.B_TOT * (cfg.T_SLOT / cfg.L_P)
            continue
        eff = se0[users]
        lit_cell = eff.sum(axis=-1) > 1e-9
        assigned = np.full_like(eff, cfg.N_RB / float(size))
        if np.any(lit_cell):
            lit_eff = eff[lit_cell]
            est = hist[lit_cell].copy()
            lit_assigned = np.zeros_like(lit_eff)
            cell_idx = np.arange(lit_eff.shape[0])
            for _ in range(cfg.N_RB // 4):
                winner = np.argmax(lit_eff / est, axis=-1)
                est[cell_idx, winner] += lit_eff[cell_idx, winner] * (4 * rb_bw / cfg.PF_WINDOW)
                lit_assigned[cell_idx, winner] += 4
            assigned[lit_cell] = lit_assigned
        rb_scale[users] = assigned * rb_bw * (cfg.T_SLOT / cfg.L_P)

    ages = np.arange(1, cfg.T_TH_R + 1, dtype=np.float64)
    rt_q = model.rt
    rt_age_norm = np.where(rt_q > 0, (env.rt_buckets @ ages) / (np.maximum(rt_q, 1e-9) * cfg.T_TH_R), 0.0)

    # Optional exact per-satellite COMA counterfactual swap advantages (old -> new) holding other 15 satellites fixed
    swap_old = np.full(cfg.N, -1, dtype=np.int64)
    swap_new = np.full(cfg.N, -1, dtype=np.int64)
    swap_adv = np.zeros(cfg.N, dtype=np.float32)
    if compute_swaps:
        cell_backlog = np.bincount(env.user_cell, weights=model.backlog, minlength=cfg.M)
        available = (env.action_mask > 0.5) & (cell_backlog[env.cand_cells.clip(min=0)] > 1e-9)
        lit = (beam_sel > 0.5) & available
        n_idx, old_idx, new_idx = np.where(lit[:, :, None] & (available & ~lit)[:, None, :])
    else:
        n_idx = np.empty(0, dtype=np.int64)
    if n_idx.size > 0:
        remove = n_idx * cfg.J + old_idx
        add = n_idx * cfg.J + new_idx
        # Strictly conserve each satellite's total power P_TOT_W by transferring exact watts[remove] to add
        delta = watts[remove, None]
        rx = rx0[None, :] + delta * (model.h[add] - model.h[remove])
        sig = np.where(model.user_beam[None, :] == remove[:, None], 0.0, sig0[None, :])
        sig = sig + (model.user_beam[None, :] == add[:, None]) * delta * model.direct_gain[None, :]
        se = np.log2(1.0 + sig / (model.noise + np.maximum(rx - sig, 0.0)))

        cap_sw = se * rb_scale[None, :]
        rt_w_vec = 1.0 + 3.0 * rt_age_norm
        cap0 = se0 * rb_scale
        base_score = float(
            np.minimum(cap0, model.backlog).sum()
            + rt_weight * (
                (np.minimum(cap0, rt_q) * rt_w_vec).sum()
                + 3.0 * np.minimum(cap0, model.urgent).sum()
            )
        )
        sw_scores = (
            np.minimum(cap_sw, model.backlog[None, :]).sum(axis=1)
            + rt_weight * (
                (np.minimum(cap_sw, rt_q[None, :]) * rt_w_vec[None, :]).sum(axis=1)
                + 3.0 * np.minimum(cap_sw, model.urgent[None, :]).sum(axis=1)
            )
        )
        deltas = sw_scores - base_score
        for sat_n in range(cfg.N):
            mask_n = n_idx == sat_n
            if not np.any(mask_n):
                continue
            loc_ids = np.flatnonzero(mask_n)
            best_i = loc_ids[int(np.argmax(deltas[loc_ids]))]
            if deltas[best_i] > 15.0:
                swap_old[sat_n] = int(old_idx[best_i])
                swap_new[sat_n] = int(new_idx[best_i])
                swap_adv[sat_n] = float(deltas[best_i])

    return {
        "rb_scale": rb_scale.astype(np.float32),
        "backlog": model.backlog.astype(np.float32),
        "rt": rt_q.astype(np.float32),
        "urgent": model.urgent.astype(np.float32),
        "rt_age_norm": rt_age_norm.astype(np.float32),
        "swap_old": swap_old,
        "swap_new": swap_new,
        "swap_adv": swap_adv,
    }


def compute_counterfactual_rewards(env: LEOBHEnv, power_ratios: np.ndarray, rt_diff_weight: float = 0.45) -> np.ndarray:
    """Exact per-satellite counterfactual difference reward D_n = Score(a_n, a_-n) - Score(0, a_-n)."""
    model = SlotService(env)
    watts = power_ratios.astype(np.float64).ravel() * cfg.P_TOT_W
    h_3d = model.h.reshape(cfg.N, cfg.J, env.U)
    w_2d = watts.reshape(cfg.N, cfg.J)
    sat_rx = np.einsum("nj,nju->nu", w_2d, h_3d)
    rx0 = sat_rx.sum(axis=0)
    rx_all = np.vstack([rx0[None, :], np.maximum(rx0[None, :] - sat_rx, 0.0)])

    cands = np.broadcast_to(watts, (cfg.N + 1, watts.size)).copy()
    for n in range(cfg.N):
        cands[n + 1, n * cfg.J : (n + 1) * cfg.J] = 0.0

    cap = model.capacity(cands, rx_all)  # (N+1, U)
    deliv_tot = np.minimum(cap, model.backlog).sum(axis=1)

    ages = np.arange(1, cfg.T_TH_R + 1, dtype=np.float64)
    rt_q = model.rt
    rt_age_norm = np.where(rt_q > 0, (env.rt_buckets @ ages) / (np.maximum(rt_q, 1e-9) * cfg.T_TH_R), 0.0)
    rt_val = (np.minimum(cap, rt_q) * (1.0 + 2.0 * rt_age_norm) + 2.0 * np.minimum(cap, model.urgent)).sum(axis=1)

    d_zeta_bits = (deliv_tot[0] - deliv_tot[1:]) * cfg.L_P
    d_rt_pkts = rt_val[0] - rt_val[1:]
    sat_zeta_norm = cfg.ZETA_NORM / cfg.N
    sat_pkt_norm = sat_zeta_norm / cfg.L_P

    tau_u_start = np.where(rt_q > 0, (env.rt_buckets @ ages) / np.maximum(rt_q, 1e-9), 0.0)
    sat_delay = np.zeros(cfg.N, dtype=np.float64)
    for n in range(cfg.N):
        us = env.sat_users_arr[n]
        if len(us) > 0:
            sat_delay[n] = float(tau_u_start[us].mean())

    r_diff = (
        env.alpha * (d_zeta_bits / sat_zeta_norm)
        + rt_diff_weight * (d_rt_pkts / sat_pkt_norm)
        - (1.0 - env.alpha) * (sat_delay / cfg.GAMMA_NORM)
    )
    return r_diff.astype(np.float32)


def collect_rich_bc_dataset(num_episodes: int = 20, slots: int = 40, seed: int = 777, cross_satellite: bool = False, search_teacher: bool = False):
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed, cross_satellite=cross_satellite)
    data = []
    for ep in range(num_episodes):
        rate = float(cfg.EVAL_RATES_MBPS[ep % len(cfg.EVAL_RATES_MBPS)])
        num_u = int(cfg.EVAL_USERS + ((ep * 13) % 41) - 20)
        obs = env.reset(input_rate_mbps=rate, num_users=num_u)
        for _ in range(slots):
            sc, pw = compute_rich_teacher_targets(obs)
            am = obs["action_mask"]
            F_r = obs["F_rich"]
            sym_I = 0.5 * (obs["I_mat"] + np.transpose(obs["I_mat"], (0, 2, 1)))
            b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            p = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            for n in range(cfg.N):
                occ = np.where((am[n] > 0.5) & ((F_r[n, :, 0] + F_r[n, :, 1]) > 1e-6))[0]
                k = min(cfg.K_MAX, len(occ))
                if k == 0:
                    continue
                cur = sc[n].copy()
                chosen, rem = [], list(occ)
                for _ in range(k):
                    bj = rem[int(np.argmax([cur[j] for j in rem]))]
                    chosen.append(bj); rem.remove(bj)
                    cur -= 1.5 * sym_I[n, bj]
                b[n, chosen] = 1.0
                ex = np.exp(pw[n, chosen])
                p[n, chosen] = ex / ex.sum()
            if search_teacher:
                b_ref, p_ref = best_global_swap(env, b, p)
                # Regress a ranking correction only where the feasible search improved the schedule.
                sc += 2.0 * (b_ref - b)
                b, p = b_ref, p_ref
            data.append((obs["F_rich"], obs["I_mat"], obs["Omega"], obs["C"], obs["real_mask"], obs["action_mask"], sc, pw, b))
            obs, _, _ = env.step(b, p)
    return data


def equal_power_np(b_mask: np.ndarray) -> np.ndarray:
    c = b_mask.sum(axis=1, keepdims=True)
    return np.where(c > 0, b_mask / np.maximum(c, 1.0), 0.0)


def compute_teacher_targets(obs: dict, interference_aware: bool = True):
    """Smooth deterministic teacher targets for BC warm-start (no per-satellite z-score ambiguity)."""
    F_in = obs["F"]        # (N, J, 5): [q, avg_d, hol_w, hist_tput, v_cov]
    Om = obs["Omega"]      # (N, J, J)
    C = obs["C"]           # (N, J, J)
    am = obs["action_mask"]
    N, J = am.shape
    scores = np.zeros((N, J), dtype=np.float32)
    power_logits = np.zeros((N, J), dtype=np.float32)

    for n in range(N):
        avail = np.where(am[n] > 0.5)[0]
        if len(avail) == 0:
            continue
        q = F_in[n, :, 0]
        d = F_in[n, :, 1]
        hol = F_in[n, :, 2]
        occ = avail[q[avail] > 1e-6]
        if len(occ) == 0:
            continue

        if interference_aware:
            # Smooth linear demand + RT urgency with neighbor interference awareness (no high-load compression)
            base = 2.4 * q + 3.6 * d + 2.4 * hol
            demand_w = q + d
            int_pen = 0.12 * (Om[n] @ demand_w) + 0.05 * (C[n] @ demand_w)
            s = base - int_pen
        else:
            # Myopic queue + delay target (for equal-power / centralized baselines)
            s = 2.0 * q + 3.0 * d + 2.0 * hol

        scores[n, occ] = np.clip(s[occ], -3.0, 12.0)
        # Unclipped shift-invariant continuous power logits: softmax(pw) == w^0.8 / sum(w^0.8) across 10-50 Mbps
        pw = 0.8 * np.log(np.maximum(q[occ] + 2.5 * d[occ] + 1.5 * hol[occ], 0.015)) + 0.5
        power_logits[n, occ] = np.clip(pw, -3.5, 3.5)

    return scores, power_logits


def collect_bc_dataset(num_episodes: int = 24, slots: int = 40, interference_aware: bool = True, seed: int = 777):
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed)
    data = []
    for ep in range(num_episodes):
        rate = float(cfg.EVAL_RATES_MBPS[ep % len(cfg.EVAL_RATES_MBPS)])
        num_u = int(cfg.EVAL_USERS + ((ep * 13) % 51) - 25)
        obs = env.reset(input_rate_mbps=rate, num_users=num_u)
        for _ in range(slots):
            sc, pw = compute_teacher_targets(obs, interference_aware=interference_aware)
            data.append((obs["s"], obs["F"], obs["Omega"], obs["C"], obs["real_mask"], obs["action_mask"], sc, pw))
            am = obs["action_mask"]
            Om = obs["Omega"]
            q_all = obs["F"][:, :, 0]
            b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            p = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            for n in range(cfg.N):
                avail = np.where(am[n] > 0.5)[0]
                occ = avail[q_all[n, avail] > 1e-6]
                cand = occ if len(occ) > 0 else avail
                k = min(cfg.K_MAX, len(cand))
                if k > 0:
                    if interference_aware:
                        # Sequential interference-avoiding rollout matching MAPPOActor decoding
                        cur_sc = sc[n].copy()
                        chosen = []
                        rem = list(cand)
                        for _ in range(k):
                            bj = rem[int(np.argmax([cur_sc[j] for j in rem]))]
                            chosen.append(bj)
                            rem.remove(bj)
                            cur_sc -= 0.9 * Om[n, bj]
                        b[n, chosen] = 1.0
                        ex = np.exp(pw[n, chosen])
                        p[n, chosen] = ex / ex.sum()
                    else:
                        chosen = cand[np.argsort(-sc[n, cand])[:k]]
                        b[n, chosen] = 1.0
                        p[n, chosen] = 1.0 / k
            obs, _, _ = env.step(b, p)
    return data


def validate_policy(eval_fn, env: LEOBHEnv, slots: int = 80, rate: float = 50.0):
    tputs, delays, rews = [], [], []
    for tc in range(5):
        obs = env.reset(input_rate_mbps=rate, num_users=cfg.EVAL_USERS, seed=5000 + tc)
        state = eval_fn("reset", None, None)
        tot_bits, d_sum, r_sum = 0.0, 0.0, 0.0
        for step_i in range(slots + 20):
            b, p, state = eval_fn("act", obs, state)
            obs, r, info = env.step(b, p)
            if step_i >= 20:
                tot_bits += info["zeta_t"]
                d_sum += info["gamma_t"]
                r_sum += r
        tputs.append((tot_bits / (slots * cfg.T_SLOT)) / (cfg.EVAL_USERS * 1e6))
        delays.append(d_sum / slots)
        rews.append(r_sum)
    return float(np.mean(rews)), float(np.mean(tputs)), float(np.mean(delays))


def train_mappo(
    episodes: int,
    slots: int,
    save_path: str,
    bc_data: list,
    seed: int = 42,
    cross_satellite: bool = False,
    sequence_length: int = 1,
    expiry_weight: float = 0.0,
    power_conditioned: bool = False,
    coop_power_gnn: bool = False,
    difference_reward: bool = False,
    analytic_ctde: bool = False,
    diff_weight: float = 0.75,
    rt_diff_weight: float = 0.65,
    apg_weight: float = 1.0,
    coma_weight: float = 0.0,
    apg_epochs: int = 3,
    bc_epochs: int = 18,
    critic_warmup: int = 16,
    lr_actor: float = 2.5e-4,
    lr_power: float = 2e-3,
    kl_target: float = 0.025,
    ent_d_coef: float = 0.003,
    ent_c_coef: float = 0.001,
    val_rates: tuple[float, ...] = (50.0,),
):
    if coop_power_gnn:
        cross_satellite = True
    print(f"\n[MAPPO-BH] BC warm-start ({bc_epochs} ep) + {episodes} episodes RL fine-tuning"
          f" (power_cond={power_conditioned}, coop_gnn={coop_power_gnn}, diff_rew={difference_reward},"
          f" analytic_ctde={analytic_ctde}, cross_sat={cross_satellite})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_dim = cfg.RICH_NODE_DIM + FEATURE_COUNT if cross_satellite else cfg.RICH_NODE_DIM
    actor = MAPPOActor(node_dim=node_dim, power_conditioned=power_conditioned, coop_power_gnn=coop_power_gnn)
    critic = MAPPOCritic(local_dim=2 * node_dim, use_attention=power_conditioned or coop_power_gnn or difference_reward)

    # Phase 1: Recurrent sequence BC warm-start (unrolled over real episodes to match eval GRU state)
    if bc_epochs > 0 and len(bc_data) >= 40:
        bc_params = actor.discrete_parameters() if coop_power_gnn else list(actor.parameters())
        opt_bc = torch.optim.Adam(bc_params, lr=2e-3)
        seq_len = 40
        n_eps_bc = len(bc_data) // seq_len
        t_F_seq = torch.tensor(np.array([d[0] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, node_dim).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, node_dim)
        t_I_seq = torch.tensor(np.array([d[1] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.J).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.J)
        t_Om_seq = torch.tensor(np.array([d[2] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.J).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.J)
        t_C_seq = torch.tensor(np.array([d[3] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.J).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.J)
        t_rm_seq = torch.tensor(np.array([d[4] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
        t_am_seq = torch.tensor(np.array([d[5] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
        sc_scale = 0.55 if analytic_ctde else 1.0
        t_sc_seq = torch.tensor(np.array([d[6] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J) * sc_scale
        t_pw_seq = torch.tensor(np.array([d[7] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
        t_bm_seq = torch.tensor(np.array([d[8] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
        B_seq = t_F_seq.shape[0]
        for _ in range(bc_epochs):
            perm = torch.randperm(B_seq)
            for st in range(0, B_seq, 96):
                idx = perm[st : st + 96]
                h = torch.zeros(1, len(idx), cfg.HIDDEN_DIM)
                loss_sum = 0.0
                for t in range(seq_len):
                    bm_t = t_bm_seq[idx, t] if power_conditioned else None
                    logits, mu, _, h = actor.forward_step(
                        t_F_seq[idx, t], t_I_seq[idx, t], t_Om_seq[idx, t], t_C_seq[idx, t], t_rm_seq[idx, t], h, b_mask=bm_t
                    )
                    am = t_am_seq[idx, t] * ((t_F_seq[idx, t][..., 0] + t_F_seq[idx, t][..., 1]) > 1e-6).float()
                    denom = am.sum().clamp(min=1.0)
                    loss_d = ((logits - t_sc_seq[idx, t]) ** 2 * am).sum() / denom
                    if coop_power_gnn:
                        loss_sum = loss_sum + loss_d
                    elif power_conditioned:
                        bm_act = t_bm_seq[idx, t]
                        k_b = bm_act.sum(dim=-1, keepdim=True).clamp(min=1.0)
                        pw_target = (t_pw_seq[idx, t] - (t_pw_seq[idx, t] * bm_act).sum(dim=-1, keepdim=True) / k_b) * bm_act
                        loss_p = ((mu - pw_target) ** 2 * bm_act).sum() / bm_act.sum().clamp(min=1.0)
                        loss_sum = loss_sum + (loss_d + 1.2 * loss_p)
                    else:
                        loss_p = ((mu - t_pw_seq[idx, t]) ** 2 * am).sum() / denom
                        loss_sum = loss_sum + (loss_d + 1.2 * loss_p)
                loss = loss_sum / seq_len
                opt_bc.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(bc_params, 1.0)
                opt_bc.step()

    # Phase 2: Genuine MAPPO RL fine-tuning with CTDE credit assignment
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed, cross_satellite=cross_satellite)
    val_env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=999, cross_satellite=cross_satellite)
    if analytic_ctde:
        opt_a = torch.optim.Adam(actor.discrete_parameters(), lr=lr_actor)
        opt_pow = torch.optim.Adam(actor.power_parameters(), lr=lr_power)
    else:
        opt_a = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        opt_pow = None
    opt_c = torch.optim.Adam(critic.parameters(), lr=5e-4)

    def eval_fn(cmd, obs, state):
        if cmd == "reset":
            return torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)
        e_mat = torch.from_numpy(obs["E_mat"]) if (coop_power_gnn and "E_mat" in obs) else None
        b, p, _, _, _, h_n = actor.act(
            torch.from_numpy(obs["F_rich"]),
            torch.from_numpy(obs["I_mat"]),
            torch.from_numpy(obs["Omega"]),
            torch.from_numpy(obs["C"]),
            torch.from_numpy(obs["real_mask"]),
            torch.from_numpy(obs["action_mask"]),
            state,
            deterministic=True,
            E_mat=e_mat,
        )
        return b.numpy(), p.numpy(), h_n

    def validate_multi():
        rets, tputs, delays = [], [], []
        for rate in val_rates:
            r, tp, dl = validate_policy(eval_fn, val_env, rate=rate)
            rets.append(r); tputs.append(tp); delays.append(dl)
        return float(np.mean(rets)), tputs[-1], delays[-1]

    t0 = time.time()
    vr0, vt0, vd0 = validate_multi()
    best_val = vr0
    best_state = copy.deepcopy(actor.state_dict())
    bc_state = copy.deepcopy(actor.state_dict())
    print(f"  [MAPPO-BH] post-BC | val@50Mbps: tput={vt0:5.2f} Mbps, delay={vd0:4.2f} slots, score={vr0:6.2f}", flush=True)

    history = []
    batch_eps = 4
    acc = {k: [] for k in ["s", "F", "I", "Om", "C", "rm", "am", "h", "j", "y", "b", "lp", "adv", "ret"]}
    acc_ep_phys = []
    acc_slot_phys = []
    noise_full_bw = float(cfg.K_BOLTZMANN * cfg.T_NOISE * (10.0 ** (cfg.UE_LOSS_NF_DB / 10.0)) * cfg.B_TOT)
    train_rates = [10, 15, 20, 25, 30, 35, 40, 45, 50, 35, 40, 45, 50] if analytic_ctde else [20, 25, 30, 35, 40, 45, 50]

    for ep in range(1, episodes + 1):
        lr_cur = (lr_actor * 0.15) + 0.5 * (lr_actor - lr_actor * 0.15) * (1.0 + math.cos(math.pi * ep / episodes))
        for pg in opt_a.param_groups:
            pg["lr"] = lr_cur
        if opt_pow is not None:
            lr_p_cur = (lr_power * 0.15) + 0.5 * (lr_power - lr_power * 0.15) * (1.0 + math.cos(math.pi * ep / episodes))
            for pg in opt_pow.param_groups:
                pg["lr"] = lr_p_cur

        rate = float(np.random.choice(train_rates))
        num_u = int(cfg.EVAL_USERS + np.random.randint(-25, 26))
        obs = env.reset(input_rate_mbps=rate, num_users=num_u)
        h = torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)
        e_mat_ep = torch.from_numpy(obs["E_mat"]) if (coop_power_gnn and "E_mat" in obs) else None

        if analytic_ctde:
            ep_idx_in_batch = len(acc_ep_phys)
            ep_model = SlotService(env)
            acc_ep_phys.append({
                "H_sq": torch.from_numpy(ep_model.h.astype(np.float32)),
                "user_beam": torch.from_numpy(ep_model.user_beam.astype(np.int64)),
                "direct_gain": torch.from_numpy(ep_model.direct_gain.astype(np.float32)),
                "E_mat": e_mat_ep,
            })

        buf_s, buf_F, buf_I, buf_Om, buf_C, buf_rm, buf_am, buf_h = [], [], [], [], [], [], [], []
        buf_j, buf_y, buf_b, buf_lp, buf_r, buf_v = [], [], [], [], [], []
        ep_ret = 0.0

        for _ in range(slots):
            s_t = torch.from_numpy(obs["s"]).unsqueeze(0)
            F_t = torch.from_numpy(obs["F_rich"])
            I_t = torch.from_numpy(obs["I_mat"])
            Om_t = torch.from_numpy(obs["Omega"])
            C_t = torch.from_numpy(obs["C"])
            rm_t = torch.from_numpy(obs["real_mask"])
            am_t = torch.from_numpy(obs["action_mask"])
            with torch.no_grad():
                v_n = critic(s_t, F_t.unsqueeze(0), am_t.unsqueeze(0)).squeeze(0).numpy()
                b_m, p_m, j_seq, y_s, lp_n, h_next = actor.act(
                    F_t, I_t, Om_t, C_t, rm_t, am_t, h,
                    deterministic=False,
                    E_mat=e_mat_ep,
                    power_deterministic=analytic_ctde,
                )
            b_np, p_np = b_m.numpy(), p_m.numpy()
            if difference_reward:
                r_diff = compute_counterfactual_rewards(env, p_np, rt_diff_weight=rt_diff_weight)
            if analytic_ctde:
                snap = extract_slot_physics_snapshot(env, b_np, p_np, rt_weight=rt_diff_weight, compute_swaps=(coma_weight > 0.0))
                snap["ep_id"] = ep_idx_in_batch
                acc_slot_phys.append(snap)
            next_obs, rew, info = env.step(b_np, p_np)
            ep_ret += rew
            team_reward = float(rew) - expiry_weight * info["expired_rt"] / max(env.U * env.base_lam_rt, 1e-9)
            if difference_reward:
                r_mix = diff_weight * r_diff + (1.0 - diff_weight) * team_reward
            else:
                local_weight = cfg.LOCAL_REWARD_WEIGHT if expiry_weight == 0.0 else 0.5
                r_mix = local_weight * info["r_local"] + (1.0 - local_weight) * team_reward

            buf_s.append(obs["s"]); buf_F.append(obs["F_rich"]); buf_I.append(obs["I_mat"])
            buf_Om.append(obs["Omega"]); buf_C.append(obs["C"]); buf_rm.append(obs["real_mask"]); buf_am.append(obs["action_mask"])
            buf_h.append(h.squeeze(0).numpy()); buf_j.append(j_seq.numpy()); buf_y.append(y_s.numpy())
            buf_b.append(b_np); buf_lp.append(lp_n.numpy()); buf_r.append(r_mix); buf_v.append(v_n)
            obs, h = next_obs, h_next

        with torch.no_grad():
            next_v = critic(
                torch.from_numpy(obs["s"]).unsqueeze(0),
                torch.from_numpy(obs["F_rich"]).unsqueeze(0),
                torch.from_numpy(obs["action_mask"]).unsqueeze(0),
            ).squeeze(0).numpy()
        adv, ret = compute_gae_vec(np.array(buf_r), np.array(buf_v), next_v)
        for k, v in [("s", buf_s), ("F", buf_F), ("I", buf_I), ("Om", buf_Om), ("C", buf_C), ("rm", buf_rm), ("am", buf_am),
                     ("h", buf_h), ("j", buf_j), ("y", buf_y), ("b", buf_b), ("lp", buf_lp)]:
            acc[k].append(np.array(v))
        acc["adv"].append(adv); acc["ret"].append(ret)
        history.append(ep_ret)

        if ep % batch_eps == 0 or ep == episodes:
            t_s = torch.tensor(np.concatenate(acc["s"], axis=0), dtype=torch.float32)
            T = t_s.shape[0]
            t_F_3d = torch.tensor(np.concatenate(acc["F"], axis=0), dtype=torch.float32)
            t_am_3d = torch.tensor(np.concatenate(acc["am"], axis=0), dtype=torch.float32)
            t_ret = torch.tensor(np.concatenate(acc["ret"], axis=0), dtype=torch.float32)

            adv_all = np.concatenate(acc["adv"], axis=0)
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
            t_adv = torch.tensor(adv_all, dtype=torch.float32).reshape(-1)
            t_F = t_F_3d.reshape(T * cfg.N, cfg.J, node_dim)
            t_I = torch.tensor(np.concatenate(acc["I"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.J)
            t_Om = torch.tensor(np.concatenate(acc["Om"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.J)
            t_C = torch.tensor(np.concatenate(acc["C"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.J)
            t_rm = torch.tensor(np.concatenate(acc["rm"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_am = t_am_3d.reshape(T * cfg.N, cfg.J)
            t_h = torch.tensor(np.concatenate(acc["h"], axis=0), dtype=torch.float32).reshape(1, T * cfg.N, cfg.HIDDEN_DIM)
            t_j = torch.tensor(np.concatenate(acc["j"], axis=0), dtype=torch.long).reshape(T * cfg.N, cfg.K_MAX)
            t_y = torch.tensor(np.concatenate(acc["y"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_b = torch.tensor(np.concatenate(acc["b"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_old_lp = torch.tensor(np.concatenate(acc["lp"], axis=0), dtype=torch.float32).reshape(T * cfg.N)

            total_samples = T * cfg.N
            stop_early = False
            for _ in range(cfg.N_EPO):
                if ep > critic_warmup:
                    sequences = torch.arange(total_samples)
                    if sequence_length > 1:
                        sequences = sequences.reshape(-1, slots, cfg.N).transpose(1, 2)
                    sequences = sequences.reshape(-1, sequence_length)
                    order = torch.randperm(len(sequences))
                    sequences_per_batch = max(1, 512 // sequence_length)
                    for chunk_ids in order.split(sequences_per_batch):
                        batch = sequences[chunk_ids].transpose(0, 1)
                        hidden = t_h[:, batch[0], :]
                        step_stats = []
                        for idx in batch:
                            if analytic_ctde:
                                new_lp, ent_d, hidden = actor.evaluate_discrete_actions(
                                    t_F[idx], t_I[idx], t_Om[idx], t_C[idx], t_rm[idx], t_am[idx],
                                    hidden, t_j[idx], return_hidden=True,
                                )
                                ent_c = torch.zeros_like(ent_d)
                            else:
                                new_lp, ent_d, ent_c, hidden = actor.evaluate_actions(
                                    t_F[idx], t_I[idx], t_Om[idx], t_C[idx], t_rm[idx], t_am[idx],
                                    hidden, t_j[idx], t_y[idx], t_b[idx], return_hidden=True,
                                )
                            step_stats.append((new_lp, ent_d, ent_c))
                        new_lp, ent_d, ent_c = map(torch.cat, zip(*step_stats))
                        idx = batch.reshape(-1)
                        log_ratio = (new_lp - t_old_lp[idx]).clamp(-10, 10)
                        if ((log_ratio.exp() - 1.0) - log_ratio).mean().item() > kl_target:
                            stop_early = True
                            break
                        ratio = log_ratio.exp()
                        surr = torch.min(ratio * t_adv[idx], ratio.clamp(1.0 - cfg.EPS_CLIP, 1.0 + cfg.EPS_CLIP) * t_adv[idx])
                        loss_a = -surr.mean() - ent_d_coef * ent_d.mean() - (0.0 if analytic_ctde else ent_c_coef * ent_c.mean())
                        opt_a.zero_grad()
                        loss_a.backward()
                        nn.utils.clip_grad_norm_(actor.discrete_parameters() if analytic_ctde else actor.parameters(), 0.5)
                        opt_a.step()

                perm_c = torch.randperm(T)
                for start in range(0, T, min(cfg.N_MB, T)):
                    idx = perm_c[start : start + min(cfg.N_MB, T)]
                    v_pred = critic(t_s[idx], t_F_3d[idx], t_am_3d[idx])
                    loss_c = F.smooth_l1_loss(v_pred, t_ret[idx], beta=2.0) if difference_reward else F.mse_loss(v_pred, t_ret[idx])
                    opt_c.zero_grad()
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_c.step()
                if stop_early:
                    break

            # Decoupled Exact Differentiable-Channel Power APG (CTDE only, starts from ep 1!)
            if analytic_ctde and opt_pow is not None:
                b_slots = 16
                for _ in range(apg_epochs):
                    slot_perm = torch.randperm(T)
                    for st_s in range(0, T, b_slots):
                        s_ids = slot_perm[st_s : st_s + b_slots]
                        apg_scores = []
                        for slot_i in s_ids.tolist():
                            snap = acc_slot_phys[slot_i]
                            ep_p = acc_ep_phys[snap["ep_id"]]
                            s0, s1 = slot_i * cfg.N, (slot_i + 1) * cfg.N
                            b_k = t_b[s0:s1]
                            mu_k = actor.compute_power_mu(
                                t_F[s0:s1], t_I[s0:s1], t_Om[s0:s1], t_C[s0:s1], t_rm[s0:s1],
                                t_h[:, s0:s1, :], b_k, E_mat=ep_p["E_mat"], detach_backbone=True,
                            )
                            p_det = F.softmax(mu_k.masked_fill(b_k < 0.5, -1e9), dim=-1) * b_k
                            p_det = p_det / p_det.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                            P_k = (p_det * cfg.P_TOT_W).reshape(-1)
                            total_rx = P_k @ ep_p["H_sq"]
                            sig = P_k[ep_p["user_beam"]] * ep_p["direct_gain"]
                            sinr = sig / (noise_full_bw + (total_rx - sig).clamp(min=0.0))
                            cap = torch.log2(1.0 + sinr) * torch.from_numpy(snap["rb_scale"])
                            bl_t = torch.from_numpy(snap["backlog"])
                            rt_t = torch.from_numpy(snap["rt"])
                            urg_t = torch.from_numpy(snap["urgent"])
                            age_t = torch.from_numpy(snap["rt_age_norm"])
                            u_k = (
                                torch.minimum(cap, bl_t)
                                + rt_diff_weight * (torch.minimum(cap, rt_t) * (1.0 + 3.0 * age_t) + 3.0 * torch.minimum(cap, urg_t))
                            ).sum()
                            apg_scores.append(u_k)
                        loss_apg = -apg_weight * torch.stack(apg_scores).mean() / (cfg.ZETA_NORM / cfg.L_P)
                        opt_pow.zero_grad()
                        loss_apg.backward()
                        nn.utils.clip_grad_norm_(actor.power_parameters(), 1.0)
                        opt_pow.step()

            for k in acc:
                acc[k].clear()
            acc_ep_phys.clear()
            acc_slot_phys.clear()

        if ep % max(1, episodes // 10) == 0 or ep == episodes:
            vr, vt, vd = validate_multi()
            if vr > best_val:
                best_val = vr
                best_state = copy.deepcopy(actor.state_dict())
            print(f"  [MAPPO-BH] ep {ep:4d}/{episodes} | val@50Mbps: tput={vt:5.2f} Mbps, delay={vd:4.2f} slots, score={vr:6.2f} (best={best_val:6.2f}) | {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "algorithm": "MAPPO-BH",
            "actor_state_dict": actor.state_dict(),
            "best_actor_state_dict": best_state,
            "bc_actor_state_dict": bc_state,
            "cross_satellite": cross_satellite,
            "power_conditioned": power_conditioned,
            "coop_power_gnn": coop_power_gnn,
            "difference_reward": difference_reward,
            "sequence_length": sequence_length,
            "expiry_weight": expiry_weight,
            "critic_state_dict": critic.state_dict(),
            "config": {"N": cfg.N, "M": cfg.M, "J": cfg.J, "K_MAX": cfg.K_MAX, "D_G": cfg.D_G_RICH, "HIDDEN_DIM": cfg.HIDDEN_DIM, "RICH_NODE_DIM": cfg.RICH_NODE_DIM},
            "train_returns": history,
        },
        save_path,
    )
    print(f"  [MAPPO-BH] Saved checkpoint -> {save_path}", flush=True)


def train_ppo(episodes: int, slots: int, save_path: str, bc_data: list, seed: int = 43):
    print(f"\n[PPO-BH] BC warm-start + {episodes} episodes RL fine-tuning...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PPOBHModel()
    opt_bc = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    t_s_all = torch.tensor(np.array([d[0] for d in bc_data]), dtype=torch.float32)
    t_F_all = torch.tensor(np.array([d[1] for d in bc_data]), dtype=torch.float32)
    t_am_all = torch.tensor(np.array([d[5] for d in bc_data]), dtype=torch.float32)
    t_sc_all = torch.tensor(np.array([d[6] for d in bc_data]), dtype=torch.float32)
    for _ in range(16):
        perm = torch.randperm(len(bc_data))
        for st in range(0, len(bc_data), 64):
            idx = perm[st : st + 64]
            logits, mu, _, _ = model(t_s_all[idx], t_F_all[idx])
            am = t_am_all[idx] * (t_F_all[idx][..., 0] > 1e-6).float()
            loss = ((logits - t_sc_all[idx]) ** 2 * am).sum() / am.sum().clamp(min=1.0)
            opt_bc.zero_grad()
            loss.backward()
            opt_bc.step()

    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed)
    val_env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=999)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)

    def eval_fn(cmd, obs, state):
        if cmd == "reset":
            return None
        b, p, _, _, _, _ = model.act(
            torch.from_numpy(obs["s"]).unsqueeze(0),
            torch.from_numpy(obs["F"]).unsqueeze(0),
            torch.from_numpy(obs["action_mask"]),
            deterministic=True,
        )
        return b.numpy(), p.numpy(), None

    t0 = time.time()
    vr0, vt0, vd0 = validate_policy(eval_fn, val_env)
    best_val, best_state = vr0, copy.deepcopy(model.state_dict())
    print(f"  [PPO-BH]   post-BC | val@50Mbps: tput={vt0:5.2f} Mbps, delay={vd0:4.2f} slots, ret={vr0:6.2f}")

    history = []
    for ep in range(1, episodes + 1):
        rate = float(np.random.choice(cfg.EVAL_RATES_MBPS))
        num_u = int(cfg.EVAL_USERS + np.random.randint(-25, 26))
        obs = env.reset(input_rate_mbps=rate, num_users=num_u)
        buf_s, buf_F, buf_am, buf_j, buf_y, buf_b, buf_lp, buf_r, buf_v = [], [], [], [], [], [], [], [], []
        ep_ret = 0.0
        for _ in range(slots):
            b_m, p_m, j_seq, y_s, lp_tot, val = model.act(
                torch.from_numpy(obs["s"]).unsqueeze(0),
                torch.from_numpy(obs["F"]).unsqueeze(0),
                torch.from_numpy(obs["action_mask"]),
                deterministic=False,
            )
            next_obs, rew, _ = env.step(b_m.numpy(), p_m.numpy())
            ep_ret += rew
            buf_s.append(obs["s"]); buf_F.append(obs["F"]); buf_am.append(obs["action_mask"])
            buf_j.append(j_seq.numpy()); buf_y.append(y_s.numpy()); buf_b.append(b_m.numpy())
            buf_lp.append(lp_tot.item()); buf_r.append(rew); buf_v.append(val.item())
            obs = next_obs

        with torch.no_grad():
            _, _, _, nv = model(torch.from_numpy(obs["s"]).unsqueeze(0), torch.from_numpy(obs["F"]).unsqueeze(0))
        adv, ret = compute_gae(buf_r, buf_v, nv.item())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T = slots
        t_s = torch.tensor(np.array(buf_s), dtype=torch.float32)
        t_F = torch.tensor(np.array(buf_F), dtype=torch.float32)
        t_am = torch.tensor(np.array(buf_am), dtype=torch.float32)
        t_j = torch.tensor(np.array(buf_j), dtype=torch.long)
        t_y = torch.tensor(np.array(buf_y), dtype=torch.float32)
        t_b = torch.tensor(np.array(buf_b), dtype=torch.float32)
        t_old_lp = torch.tensor(np.array(buf_lp), dtype=torch.float32)
        t_adv = torch.tensor(adv, dtype=torch.float32)
        t_ret = torch.tensor(ret, dtype=torch.float32)

        for _ in range(2):
            perm = torch.randperm(T)
            for start in range(0, T, 32):
                idx = perm[start : start + 32]
                B = len(idx)
                logits, mu, std, val = model(t_s[idx], t_F[idx])
                eff_am = t_am[idx] * (t_F[idx][..., 0] > 1e-6).float()
                lp_d, ent_d = evaluate_without_replacement_batch(
                    logits.reshape(B * cfg.N, cfg.J), eff_am.reshape(B * cfg.N, cfg.J), t_j[idx].reshape(B * cfg.N, cfg.K_MAX)
                )
                dist = torch.distributions.Normal(mu, std)
                new_lp = lp_d.view(B, cfg.N).sum(dim=1) + (dist.log_prob(t_y[idx]) * t_b[idx]).sum(dim=(1, 2))
                ratio = (new_lp - t_old_lp[idx]).clamp(-10, 10).exp()
                surr = torch.min(ratio * t_adv[idx], ratio.clamp(0.8, 1.2) * t_adv[idx])
                loss = -surr.mean() + 0.5 * F.mse_loss(val, t_ret[idx]) - 0.002 * ent_d.mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()

        history.append(ep_ret)
        if ep % max(1, episodes // 4) == 0:
            vr, vt, vd = validate_policy(eval_fn, val_env)
            if vr > best_val:
                best_val, best_state = vr, copy.deepcopy(model.state_dict())
            print(f"  [PPO-BH]   ep {ep:4d}/{episodes} | val@50Mbps: tput={vt:5.2f} Mbps, delay={vd:4.2f} slots, ret={vr:6.2f} | {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"algorithm": "PPO-BH", "state_dict": best_state, "train_returns": history}, save_path)
    print(f"  [PPO-BH]   Saved checkpoint -> {save_path}")


def train_qmix(episodes: int, slots: int, save_path: str, bc_data: list, seed: int = 44):
    print(f"\n[QMIX-BH] BC warm-start + {episodes} episodes RL fine-tuning...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = QMIXAgent()
    mixer = QMIXMixer()

    # Recurrent sequence BC warm-start for Q-values
    opt_bc = torch.optim.Adam(agent.parameters(), lr=2e-3)
    seq_len = 40
    n_eps_bc = len(bc_data) // seq_len
    t_F_seq = torch.tensor(np.array([d[1] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.NODE_FEAT_DIM).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.NODE_FEAT_DIM)
    t_am_seq = torch.tensor(np.array([d[5] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    t_sc_seq = torch.tensor(np.array([d[6] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    B_seq = t_F_seq.shape[0]
    for _ in range(20):
        perm = torch.randperm(B_seq)
        for st in range(0, B_seq, 108):
            idx = perm[st : st + 108]
            h = torch.zeros(1, len(idx), 128)
            loss_sum = 0.0
            for t in range(seq_len):
                q, h = agent(t_F_seq[idx, t], h)
                am = t_am_seq[idx, t] * (t_F_seq[idx, t][..., 0] > 1e-6).float()
                loss_sum = loss_sum + ((q - t_sc_seq[idx, t]) ** 2 * am).sum() / am.sum().clamp(min=1.0)
            loss = loss_sum / seq_len
            opt_bc.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt_bc.step()

    t_agent, t_mixer = copy.deepcopy(agent), copy.deepcopy(mixer)
    opt = torch.optim.Adam(list(agent.parameters()) + list(mixer.parameters()), lr=1e-4)
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed)
    val_env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=999)

    def eval_fn(cmd, obs, state):
        if cmd == "reset":
            return torch.zeros(1, cfg.N, 128)
        with torch.no_grad():
            q_vals, h_n = agent(torch.from_numpy(obs["F"]), state)
            q_np = q_vals.numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                b[n, avail[np.argsort(-q_np[n, avail])[:k]]] = 1.0
        return b, equal_power_np(b), h_n

    t0 = time.time()
    vr0, vt0, vd0 = validate_policy(eval_fn, val_env)
    best_val, best_state = vr0, copy.deepcopy(agent.state_dict())
    print(f"  [QMIX-BH]  post-BC | val@50Mbps: tput={vt0:5.2f} Mbps, delay={vd0:4.2f} slots, ret={vr0:6.2f}")

    replay, history = [], []
    for ep in range(1, episodes + 1):
        eps = max(0.05, 0.25 * (1.0 - ep / episodes))
        obs = env.reset(input_rate_mbps=float(np.random.choice(cfg.EVAL_RATES_MBPS)), num_users=int(cfg.EVAL_USERS + np.random.randint(-25, 26)))
        h = torch.zeros(1, cfg.N, 128)
        ep_ret = 0.0
        for _ in range(slots):
            am = obs["action_mask"]
            with torch.no_grad():
                q_vals, h_next = agent(torch.from_numpy(obs["F"]), h)
                q_np = q_vals.numpy()
            b_sel = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            for n in range(cfg.N):
                avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
                k = min(cfg.K_MAX, len(avail))
                if k == 0: continue
                chosen = np.random.choice(avail, size=k, replace=False) if np.random.random() < eps else avail[np.argsort(-q_np[n, avail])[:k]]
                b_sel[n, chosen] = 1.0
            next_obs, rew, _ = env.step(b_sel, equal_power_np(b_sel))
            ep_ret += rew
            replay.append((obs["s"], obs["F"], b_sel, rew, next_obs["s"], next_obs["F"], next_obs["action_mask"]))
            if len(replay) > 4000: replay.pop(0)
            obs, h = next_obs, h_next

        if len(replay) >= 64:
            for _ in range(3):
                idxs = np.random.choice(len(replay), size=64, replace=False)
                bs = torch.tensor(np.array([replay[i][0] for i in idxs]), dtype=torch.float32)
                bF = torch.tensor(np.array([replay[i][1] for i in idxs]), dtype=torch.float32)
                bb = torch.tensor(np.array([replay[i][2] for i in idxs]), dtype=torch.float32)
                br = torch.tensor(np.array([replay[i][3] for i in idxs]), dtype=torch.float32)
                bns = torch.tensor(np.array([replay[i][4] for i in idxs]), dtype=torch.float32)
                bnF = torch.tensor(np.array([replay[i][5] for i in idxs]), dtype=torch.float32)
                bnam = torch.tensor(np.array([replay[i][6] for i in idxs]), dtype=torch.float32)
                B = len(idxs)
                h0 = torch.zeros(1, B * cfg.N, 128)
                q_all, _ = agent(bF.reshape(B * cfg.N, cfg.J, cfg.NODE_FEAT_DIM), h0)
                q_chosen = (q_all.view(B, cfg.N, cfg.J) * bb).sum(dim=-1) / bb.sum(dim=-1).clamp(min=1.0)
                q_tot = mixer(q_chosen, bs)
                with torch.no_grad():
                    nq_all, _ = t_agent(bnF.reshape(B * cfg.N, cfg.J, cfg.NODE_FEAT_DIM), h0)
                    nq_masked = nq_all.view(B, cfg.N, cfg.J).masked_fill((bnam < 0.5) | (bnF[..., 0] <= 1e-6), -1e4)
                    topk_vals = torch.topk(nq_masked, k=cfg.K_MAX, dim=-1).values
                    valid_k = (topk_vals > -1e3).float()
                    nq_chosen = (topk_vals * valid_k).sum(dim=-1) / valid_k.sum(dim=-1).clamp(min=1.0)
                    target = br + cfg.GAMMA * t_mixer(nq_chosen, bns)
                loss = F.smooth_l1_loss(q_tot, target)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(agent.parameters()) + list(mixer.parameters()), 1.0); opt.step()

        if ep % 10 == 0:
            t_agent.load_state_dict(agent.state_dict()); t_mixer.load_state_dict(mixer.state_dict())
        history.append(ep_ret)
        if ep % max(1, episodes // 4) == 0:
            vr, vt, vd = validate_policy(eval_fn, val_env)
            if vr > best_val:
                best_val, best_state = vr, copy.deepcopy(agent.state_dict())
            print(f"  [QMIX-BH]  ep {ep:4d}/{episodes} | val@50Mbps: tput={vt:5.2f} Mbps, delay={vd:4.2f} slots, ret={vr:6.2f} | {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"algorithm": "QMIX-BH", "agent_state_dict": best_state, "mixer_state_dict": mixer.state_dict(), "train_returns": history}, save_path)
    print(f"  [QMIX-BH]  Saved checkpoint -> {save_path}")


def train_maddpg(episodes: int, slots: int, save_path: str, bc_data: list, seed: int = 45):
    print(f"\n[MADDPG-BH] BC warm-start + {episodes} episodes RL fine-tuning...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = MADDPGActor()
    critic = MADDPGCritic()

    opt_bc = torch.optim.Adam(actor.parameters(), lr=1.5e-3)
    t_F_all = torch.tensor(np.array([d[1] for d in bc_data]), dtype=torch.float32).reshape(-1, cfg.J, cfg.NODE_FEAT_DIM)
    t_am_all = torch.tensor(np.array([d[5] for d in bc_data]), dtype=torch.float32).reshape(-1, cfg.J)
    t_sc_all = torch.tensor(np.array([d[6] for d in bc_data]), dtype=torch.float32).reshape(-1, cfg.J)
    for _ in range(16):
        perm = torch.randperm(t_F_all.shape[0])
        for st in range(0, t_F_all.shape[0], 256):
            idx = perm[st : st + 256]
            logits = actor(t_F_all[idx])
            am = t_am_all[idx] * (t_F_all[idx][..., 0] > 1e-6).float()
            loss = ((logits - t_sc_all[idx]) ** 2 * am).sum() / am.sum().clamp(min=1.0)
            opt_bc.zero_grad(); loss.backward(); opt_bc.step()

    t_actor, t_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed)
    val_env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=999)

    def eval_fn(cmd, obs, state):
        if cmd == "reset":
            return None
        with torch.no_grad():
            logits = actor(torch.from_numpy(obs["F"])).numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                b[n, avail[np.argsort(-logits[n, avail])[:k]]] = 1.0
        return b, equal_power_np(b), None

    t0 = time.time()
    vr0, vt0, vd0 = validate_policy(eval_fn, val_env)
    best_val, best_state = vr0, copy.deepcopy(actor.state_dict())
    print(f"  [MADDPG-BH] post-BC | val@50Mbps: tput={vt0:5.2f} Mbps, delay={vd0:4.2f} slots, ret={vr0:6.2f}")

    replay, history = [], []
    for ep in range(1, episodes + 1):
        noise_scale = max(0.05, 0.25 * (1.0 - ep / episodes))
        obs = env.reset(input_rate_mbps=float(np.random.choice(cfg.EVAL_RATES_MBPS)), num_users=int(cfg.EVAL_USERS + np.random.randint(-25, 26)))
        ep_ret = 0.0
        for _ in range(slots):
            am = obs["action_mask"]
            with torch.no_grad():
                logits = actor(torch.from_numpy(obs["F"])).numpy() + np.random.normal(0.0, noise_scale, size=(cfg.N, cfg.J))
            b_sel = np.zeros((cfg.N, cfg.J), dtype=np.float32)
            for n in range(cfg.N):
                avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
                k = min(cfg.K_MAX, len(avail))
                if k > 0:
                    b_sel[n, avail[np.argsort(-logits[n, avail])[:k]]] = 1.0
            next_obs, rew, _ = env.step(b_sel, equal_power_np(b_sel))
            ep_ret += rew
            replay.append((obs["s"], obs["F"], am, b_sel, rew, next_obs["s"], next_obs["F"], next_obs["action_mask"]))
            if len(replay) > 4000: replay.pop(0)
            obs = next_obs

        if len(replay) >= 64:
            for _ in range(3):
                idxs = np.random.choice(len(replay), size=64, replace=False)
                bs = torch.tensor(np.array([replay[i][0] for i in idxs]), dtype=torch.float32)
                bF = torch.tensor(np.array([replay[i][1] for i in idxs]), dtype=torch.float32)
                bam = torch.tensor(np.array([replay[i][2] for i in idxs]), dtype=torch.float32)
                bb = torch.tensor(np.array([replay[i][3] for i in idxs]), dtype=torch.float32)
                br = torch.tensor(np.array([replay[i][4] for i in idxs]), dtype=torch.float32)
                bns = torch.tensor(np.array([replay[i][5] for i in idxs]), dtype=torch.float32)
                bnF = torch.tensor(np.array([replay[i][6] for i in idxs]), dtype=torch.float32)
                bnam = torch.tensor(np.array([replay[i][7] for i in idxs]), dtype=torch.float32)
                B = len(idxs)
                with torch.no_grad():
                    n_logits = t_actor(bnF.reshape(B * cfg.N, cfg.J, cfg.NODE_FEAT_DIM)).view(B, cfg.N, cfg.J)
                    target_q = br + cfg.GAMMA * t_critic(bns, (torch.sigmoid(n_logits) * bnam).reshape(B, -1))
                loss_c = F.smooth_l1_loss(critic(bs, bb.reshape(B, -1)), target_q)
                opt_c.zero_grad(); loss_c.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt_c.step()

                cur_logits = actor(bF.reshape(B * cfg.N, cfg.J, cfg.NODE_FEAT_DIM)).view(B, cfg.N, cfg.J)
                loss_a = -critic(bs, (torch.sigmoid(cur_logits) * bam).reshape(B, -1)).mean() + 0.01 * (cur_logits ** 2).mean()
                opt_a.zero_grad(); loss_a.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()

                for p, tp in zip(actor.parameters(), t_actor.parameters()): tp.data.mul_(0.98).add_(0.02 * p.data)
                for p, tp in zip(critic.parameters(), t_critic.parameters()): tp.data.mul_(0.98).add_(0.02 * p.data)

        history.append(ep_ret)
        if ep % max(1, episodes // 4) == 0:
            vr, vt, vd = validate_policy(eval_fn, val_env)
            if vr > best_val:
                best_val, best_state = vr, copy.deepcopy(actor.state_dict())
            print(f"  [MADDPG-BH] ep {ep:4d}/{episodes} | val@50Mbps: tput={vt:5.2f} Mbps, delay={vd:4.2f} slots, ret={vr:6.2f} | {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"algorithm": "MADDPG-BH", "actor_state_dict": best_state, "critic_state_dict": critic.state_dict(), "train_returns": history}, save_path)
    print(f"  [MADDPG-BH] Saved checkpoint -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80, help="RL fine-tuning episodes per algorithm")
    parser.add_argument("--slots", type=int, default=64, help="Slots per rollout episode")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--algos", nargs="+", default=["mappo", "ppo", "qmix", "maddpg"])
    parser.add_argument("--cross-satellite", action="store_true", help="Use neighbor interference features in MAPPO")
    parser.add_argument("--power-conditioned", action="store_true", help="Condition power head on selected beams and intra-set interference")
    parser.add_argument("--coop-power-gnn", action="store_true", help="Use 3-step physics-informed iterative interference GNN for cooperative power control")
    parser.add_argument("--difference-reward", action="store_true", help="Use exact per-satellite counterfactual difference rewards in CTDE")
    parser.add_argument("--analytic-ctde", action="store_true", help="Decoupled discrete PPO + differentiable-channel power APG during CTDE training")
    parser.add_argument("--apg-weight", type=float, default=1.0, help="Weight on differentiable-channel power APG loss")
    parser.add_argument("--apg-epochs", type=int, default=3, help="Differentiable power APG epochs per rollout batch")
    parser.add_argument("--coma-weight", type=float, default=0.0, help="Weight on COMA counterfactual beam-swap advantage loss")
    parser.add_argument("--diff-weight", type=float, default=0.75, help="Weight on counterfactual difference reward vs team reward")
    parser.add_argument("--rt-diff-weight", type=float, default=0.65, help="Weight on RT age-weighted delivery in difference reward & APG")
    parser.add_argument("--bc-epochs", type=int, default=18, help="BC warm-start epochs for MAPPO (0 for RL from scratch)")
    parser.add_argument("--critic-warmup", type=int, default=16, help="Episodes of critic-only warm-up before actor updates")
    parser.add_argument("--lr-actor", type=float, default=2.5e-4, help="Initial discrete actor learning rate")
    parser.add_argument("--lr-power", type=float, default=2e-3, help="Initial power head/GNN APG learning rate")
    parser.add_argument("--kl-target", type=float, default=None, help="KL early-stop threshold per update")
    parser.add_argument("--ent-d", type=float, default=0.003, help="Discrete entropy coefficient")
    parser.add_argument("--ent-c", type=float, default=0.001, help="Continuous entropy coefficient")
    parser.add_argument("--val-rates", nargs="+", type=float, default=[50.0], help="Validation rates for best checkpoint selection")
    parser.add_argument("--sequence-length", type=int, default=1, help="MAPPO recurrent update length")
    parser.add_argument("--expiry-weight", type=float, default=0.0, help="MAPPO team penalty per normalized RT expiry")
    parser.add_argument("--search-teacher", action="store_true", help="Refine MAPPO demonstration schedules using a beam swap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.slots < 1 or args.slots % args.sequence_length:
        parser.error("The MAPPO sequence length must divide the rollout slot count")

    cross_sat = args.cross_satellite or args.coop_power_gnn
    kl_target = args.kl_target if args.kl_target is not None else (0.04 if args.analytic_ctde else (0.10 if (args.power_conditioned or args.difference_reward) else 0.025))
    torch.set_num_threads(args.threads)
    if "mappo" in args.algos:
        bc_mappo = (
            collect_rich_bc_dataset(num_episodes=20, slots=40, seed=777, cross_satellite=cross_sat, search_teacher=args.search_teacher)
            if args.bc_epochs > 0
            else []
        )
        train_mappo(
            args.episodes,
            args.slots,
            os.path.join(args.ckpt_dir, "mappo_bh.pt"),
            bc_mappo,
            seed=args.seed,
            cross_satellite=cross_sat,
            sequence_length=args.sequence_length,
            expiry_weight=args.expiry_weight,
            power_conditioned=args.power_conditioned,
            coop_power_gnn=args.coop_power_gnn,
            difference_reward=args.difference_reward,
            analytic_ctde=args.analytic_ctde,
            diff_weight=args.diff_weight,
            rt_diff_weight=args.rt_diff_weight,
            apg_weight=args.apg_weight,
            coma_weight=args.coma_weight,
            apg_epochs=args.apg_epochs,
            bc_epochs=args.bc_epochs,
            critic_warmup=args.critic_warmup,
            lr_actor=args.lr_actor,
            lr_power=args.lr_power,
            kl_target=kl_target,
            ent_d_coef=args.ent_d,
            ent_c_coef=args.ent_c,
            val_rates=tuple(args.val_rates),
        )
    if any(a in args.algos for a in ["ppo", "qmix", "maddpg"]):
        bc_base = collect_bc_dataset(num_episodes=27, slots=40, interference_aware=False, seed=888)
        if "ppo" in args.algos:
            train_ppo(args.episodes, args.slots, os.path.join(args.ckpt_dir, "ppo_bh.pt"), bc_base)
        if "qmix" in args.algos:
            train_qmix(args.episodes, args.slots, os.path.join(args.ckpt_dir, "qmix_bh.pt"), bc_base)
        if "maddpg" in args.algos:
            train_maddpg(args.episodes, args.slots, os.path.join(args.ckpt_dir, "maddpg_bh.pt"), bc_base)


if __name__ == "__main__":
    main()
