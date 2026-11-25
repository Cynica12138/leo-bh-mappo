# -*- coding: utf-8 -*-
"""Train MAPPO-BH, PPO-BH, QMIX-BH, and MADDPG-BH (BC warm-start + genuine RL fine-tuning)."""

import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from env import LEOBHEnv
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
        num_u = int(170 + (ep * 13) % 65)
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
        obs = env.reset(input_rate_mbps=rate, num_users=200, seed=5000 + tc)
        state = eval_fn("reset", None, None)
        tot_bits, d_sum, r_sum = 0.0, 0.0, 0.0
        for step_i in range(slots + 20):
            b, p, state = eval_fn("act", obs, state)
            obs, r, info = env.step(b, p)
            if step_i >= 20:
                tot_bits += info["zeta_t"]
                d_sum += info["gamma_t"]
                r_sum += r
        tputs.append((tot_bits / (slots * cfg.T_SLOT)) / (200 * 1e6))
        delays.append(d_sum / slots)
        rews.append(r_sum)
    return float(np.mean(rews)), float(np.mean(tputs)), float(np.mean(delays))


def train_mappo(episodes: int, slots: int, save_path: str, bc_data: list, seed: int = 42):
    print(f"\n[MAPPO-BH] BC warm-start + {episodes} episodes RL fine-tuning...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = MAPPOActor()
    critic = MAPPOCritic()

    # Phase 1: Recurrent sequence BC warm-start (unrolled over real episodes to match eval GRU state)
    opt_bc = torch.optim.Adam(actor.parameters(), lr=2e-3)
    seq_len = 40
    n_eps_bc = len(bc_data) // seq_len
    t_F_seq = torch.tensor(np.array([d[1] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.NODE_FEAT_DIM).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.NODE_FEAT_DIM)
    t_Om_seq = torch.tensor(np.array([d[2] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.J).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.J)
    t_C_seq = torch.tensor(np.array([d[3] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J, cfg.J).permute(0, 2, 1, 3, 4).reshape(n_eps_bc * cfg.N, seq_len, cfg.J, cfg.J)
    t_rm_seq = torch.tensor(np.array([d[4] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    t_am_seq = torch.tensor(np.array([d[5] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    t_sc_seq = torch.tensor(np.array([d[6] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    t_pw_seq = torch.tensor(np.array([d[7] for d in bc_data[: n_eps_bc * seq_len]]), dtype=torch.float32).reshape(n_eps_bc, seq_len, cfg.N, cfg.J).permute(0, 2, 1, 3).reshape(n_eps_bc * cfg.N, seq_len, cfg.J)
    B_seq = t_F_seq.shape[0]
    for _ in range(25):
        perm = torch.randperm(B_seq)
        for st in range(0, B_seq, 108):
            idx = perm[st : st + 108]
            h = torch.zeros(1, len(idx), cfg.HIDDEN_DIM)
            loss_sum = 0.0
            for t in range(seq_len):
                logits, mu, _, h = actor.forward_step(t_F_seq[idx, t], t_Om_seq[idx, t], t_C_seq[idx, t], t_rm_seq[idx, t], h)
                am = t_am_seq[idx, t] * (t_F_seq[idx, t][..., 0] > 1e-6).float()
                denom = am.sum().clamp(min=1.0)
                loss_d = ((logits - t_sc_seq[idx, t]) ** 2 * am).sum() / denom
                loss_p = ((mu - t_pw_seq[idx, t]) ** 2 * am).sum() / denom
                loss_sum = loss_sum + (loss_d + 1.5 * loss_p)
            loss = loss_sum / seq_len
            opt_bc.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_bc.step()

    # Phase 2: Genuine MAPPO RL fine-tuning
    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=seed)
    val_env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=999)
    opt_a = torch.optim.Adam(actor.parameters(), lr=5e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.LR_CRITIC)

    def eval_fn(cmd, obs, state):
        if cmd == "reset":
            return torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)
        b, p, _, _, _, h_n = actor.act(
            torch.from_numpy(obs["F"]),
            torch.from_numpy(obs["Omega"]),
            torch.from_numpy(obs["C"]),
            torch.from_numpy(obs["real_mask"]),
            torch.from_numpy(obs["action_mask"]),
            state,
            deterministic=True,
        )
        return b.numpy(), p.numpy(), h_n

    t0 = time.time()
    vr0, vt0, vd0 = validate_policy(eval_fn, val_env)
    best_val = vr0
    best_state = copy.deepcopy(actor.state_dict())
    print(f"  [MAPPO-BH] post-BC | val@50Mbps: tput={vt0:5.2f} Mbps, delay={vd0:4.2f} slots, ret={vr0:6.2f}")

    history = []
    batch_eps = 4
    acc = {k: [] for k in ["s", "F", "Om", "C", "rm", "am", "h", "j", "y", "b", "lp", "adv", "ret"]}

    for ep in range(1, episodes + 1):
        rate = float(np.random.choice(cfg.EVAL_RATES_MBPS))
        num_u = int(np.random.randint(170, 235))
        obs = env.reset(input_rate_mbps=rate, num_users=num_u)
        h = torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)

        buf_s, buf_F, buf_Om, buf_C, buf_rm, buf_am, buf_h = [], [], [], [], [], [], []
        buf_j, buf_y, buf_b, buf_lp, buf_r, buf_v = [], [], [], [], [], []
        ep_ret = 0.0

        for _ in range(slots):
            s_t = torch.from_numpy(obs["s"]).unsqueeze(0)
            F_t = torch.from_numpy(obs["F"])
            Om_t = torch.from_numpy(obs["Omega"])
            C_t = torch.from_numpy(obs["C"])
            rm_t = torch.from_numpy(obs["real_mask"])
            am_t = torch.from_numpy(obs["action_mask"])
            with torch.no_grad():
                v_t = critic(s_t).item()
                b_m, p_m, j_seq, y_s, lp_n, h_next = actor.act(F_t, Om_t, C_t, rm_t, am_t, h, deterministic=False)
            next_obs, rew, _ = env.step(b_m.numpy(), p_m.numpy())
            ep_ret += rew

            buf_s.append(obs["s"]); buf_F.append(obs["F"]); buf_Om.append(obs["Omega"])
            buf_C.append(obs["C"]); buf_rm.append(obs["real_mask"]); buf_am.append(obs["action_mask"])
            buf_h.append(h.squeeze(0).numpy()); buf_j.append(j_seq.numpy()); buf_y.append(y_s.numpy())
            buf_b.append(b_m.numpy()); buf_lp.append(lp_n.numpy()); buf_r.append(rew); buf_v.append(v_t)
            obs, h = next_obs, h_next

        with torch.no_grad():
            next_v = critic(torch.from_numpy(obs["s"]).unsqueeze(0)).item()
        adv, ret = compute_gae(buf_r, buf_v, next_v)
        for k, v in [("s", buf_s), ("F", buf_F), ("Om", buf_Om), ("C", buf_C), ("rm", buf_rm), ("am", buf_am),
                     ("h", buf_h), ("j", buf_j), ("y", buf_y), ("b", buf_b), ("lp", buf_lp)]:
            acc[k].append(np.array(v))
        acc["adv"].append(adv); acc["ret"].append(ret)
        history.append(ep_ret)

        if ep % batch_eps == 0 or ep == episodes:
            t_s = torch.tensor(np.concatenate(acc["s"], axis=0), dtype=torch.float32)
            T = t_s.shape[0]
            adv_all = np.concatenate(acc["adv"], axis=0)
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
            t_ret = torch.tensor(np.concatenate(acc["ret"], axis=0), dtype=torch.float32)
            t_adv = torch.tensor(adv_all, dtype=torch.float32).unsqueeze(1).expand(T, cfg.N).reshape(-1)
            t_F = torch.tensor(np.concatenate(acc["F"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.NODE_FEAT_DIM)
            t_Om = torch.tensor(np.concatenate(acc["Om"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.J)
            t_C = torch.tensor(np.concatenate(acc["C"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J, cfg.J)
            t_rm = torch.tensor(np.concatenate(acc["rm"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_am = torch.tensor(np.concatenate(acc["am"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_h = torch.tensor(np.concatenate(acc["h"], axis=0), dtype=torch.float32).reshape(1, T * cfg.N, cfg.HIDDEN_DIM)
            t_j = torch.tensor(np.concatenate(acc["j"], axis=0), dtype=torch.long).reshape(T * cfg.N, cfg.K_MAX)
            t_y = torch.tensor(np.concatenate(acc["y"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_b = torch.tensor(np.concatenate(acc["b"], axis=0), dtype=torch.float32).reshape(T * cfg.N, cfg.J)
            t_old_lp = torch.tensor(np.concatenate(acc["lp"], axis=0), dtype=torch.float32).reshape(T * cfg.N)

            total_samples = T * cfg.N
            stop_early = False
            for _ in range(cfg.N_EPO):
                if ep > 16:
                    perm_a = torch.randperm(total_samples)
                    for start in range(0, total_samples, 512):
                        idx = perm_a[start : start + 512]
                        new_lp, ent_d, ent_c = actor.evaluate_actions(
                            t_F[idx], t_Om[idx], t_C[idx], t_rm[idx], t_am[idx], t_h[:, idx, :], t_j[idx], t_y[idx], t_b[idx]
                        )
                        log_ratio = (new_lp - t_old_lp[idx]).clamp(-10, 10)
                        if ((log_ratio.exp() - 1.0) - log_ratio).mean().item() > 0.015:
                            stop_early = True
                            break
                        ratio = log_ratio.exp()
                        surr = torch.min(ratio * t_adv[idx], ratio.clamp(1.0 - cfg.EPS_CLIP, 1.0 + cfg.EPS_CLIP) * t_adv[idx])
                        loss_a = -surr.mean() - 0.002 * ent_d.mean() - 0.0005 * ent_c.mean()
                        opt_a.zero_grad()
                        loss_a.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        opt_a.step()

                perm_c = torch.randperm(T)
                for start in range(0, T, min(cfg.N_MB, T)):
                    idx = perm_c[start : start + min(cfg.N_MB, T)]
                    loss_c = F.mse_loss(critic(t_s[idx]), t_ret[idx])
                    opt_c.zero_grad()
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_c.step()
                if stop_early:
                    break
            for k in acc:
                acc[k].clear()

        if ep % max(1, episodes // 8) == 0:
            vr, vt, vd = validate_policy(eval_fn, val_env)
            if vr > best_val:
                best_val = vr
                best_state = copy.deepcopy(actor.state_dict())
            print(f"  [MAPPO-BH] ep {ep:4d}/{episodes} | val@50Mbps: tput={vt:5.2f} Mbps, delay={vd:4.2f} slots, ret={vr:6.2f} | {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "algorithm": "MAPPO-BH",
            "actor_state_dict": actor.state_dict(),
            "best_actor_state_dict": best_state,
            "critic_state_dict": critic.state_dict(),
            "config": {"N": cfg.N, "M": cfg.M, "J": cfg.J, "K_MAX": cfg.K_MAX, "D_G": cfg.D_G, "HIDDEN_DIM": cfg.HIDDEN_DIM},
            "train_returns": history,
        },
        save_path,
    )
    print(f"  [MAPPO-BH] Saved checkpoint -> {save_path}")


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
        num_u = int(np.random.randint(170, 235))
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
        obs = env.reset(input_rate_mbps=float(np.random.choice(cfg.EVAL_RATES_MBPS)), num_users=int(np.random.randint(170, 235)))
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
                    nq_masked = nq_all.view(B, cfg.N, cfg.J).masked_fill(bnam < 0.5, -1e4)
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
        obs = env.reset(input_rate_mbps=float(np.random.choice(cfg.EVAL_RATES_MBPS)), num_users=int(np.random.randint(170, 235)))
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
    args = parser.parse_args()

    torch.set_num_threads(2)
    if "mappo" in args.algos:
        bc_mappo = collect_bc_dataset(num_episodes=27, slots=40, interference_aware=True, seed=777)
        train_mappo(args.episodes, args.slots, os.path.join(args.ckpt_dir, "mappo_bh.pt"), bc_mappo)
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
