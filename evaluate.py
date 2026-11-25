# -*- coding: utf-8 -*-
"""Evaluate trained .pt checkpoints and heuristic baselines across 10-50 Mbps."""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from env import LEOBHEnv
from heuristics import PeriodicBH, QLFBeamHopping, USWGPBeamHopping
from models import MAPPOActor, PPOBHModel, QMIXAgent, MADDPGActor


def equal_power_np(b_mask: np.ndarray) -> np.ndarray:
    c = b_mask.sum(axis=1, keepdims=True)
    return np.where(c > 0, b_mask / np.maximum(c, 1.0), 0.0)


class MAPPOPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.actor = MAPPOActor()
        state = ckpt.get("best_actor_state_dict") or ckpt["actor_state_dict"]
        self.actor.load_state_dict(state)
        self.actor.eval()
        self.h = None

    def reset(self):
        self.h = torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)

    @torch.no_grad()
    def act(self, obs: dict):
        b, p, _, _, _, self.h = self.actor.act(
            torch.from_numpy(obs["F"]),
            torch.from_numpy(obs["Omega"]),
            torch.from_numpy(obs["C"]),
            torch.from_numpy(obs["real_mask"]),
            torch.from_numpy(obs["action_mask"]),
            self.h,
            deterministic=True,
        )
        return b.numpy(), p.numpy()


class PPOPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = PPOBHModel()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        b, p, _, _, _, _ = self.model.act(
            torch.from_numpy(obs["s"]).unsqueeze(0),
            torch.from_numpy(obs["F"]).unsqueeze(0),
            torch.from_numpy(obs["action_mask"]),
            deterministic=True,
        )
        return b.numpy(), p.numpy()


class QMIXPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.agent = QMIXAgent()
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.agent.eval()
        self.h = None

    def reset(self):
        self.h = torch.zeros(1, cfg.N, 128)

    @torch.no_grad()
    def act(self, obs: dict):
        q_vals, self.h = self.agent(torch.from_numpy(obs["F"]), self.h)
        q_np = q_vals.numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                chosen = avail[np.argsort(-q_np[n, avail])[:k]]
                b[n, chosen] = 1.0
        return b, equal_power_np(b)


class MADDPGPolicyWrapper:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.actor = MADDPGActor()
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        logits = self.actor(torch.from_numpy(obs["F"])).numpy()
        am = obs["action_mask"]
        b = np.zeros((cfg.N, cfg.J), dtype=np.float32)
        for n in range(cfg.N):
            avail = np.where((am[n] > 0.5) & (obs["F"][n, :, 0] > 1e-6))[0]
            k = min(cfg.K_MAX, len(avail))
            if k > 0:
                chosen = avail[np.argsort(-logits[n, avail])[:k]]
                b[n, chosen] = 1.0
        return b, equal_power_np(b)


def benchmark_mappo_latency(policy: MAPPOPolicyWrapper, env: LEOBHEnv):
    torch.set_num_threads(1)
    obs = env.reset(input_rate_mbps=40.0, num_users=300, seed=999)
    policy.reset()
    for _ in range(30):
        policy.act(obs)
    t0 = time.perf_counter()
    for _ in range(300):
        policy.act(obs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / 300.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--test-cases", type=int, default=20)
    parser.add_argument("--slots", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    policies = {
        "P-BH": PeriodicBH(),
        "QLF-BH": QLFBeamHopping(),
        "USWGP-BH": USWGPBeamHopping(),
        "PPO-BH": PPOPolicyWrapper(os.path.join(args.ckpt_dir, "ppo_bh.pt")),
        "QMIX-BH": QMIXPolicyWrapper(os.path.join(args.ckpt_dir, "qmix_bh.pt")),
        "MADDPG-BH": MADDPGPolicyWrapper(os.path.join(args.ckpt_dir, "maddpg_bh.pt")),
        "MAPPO-BH": MAPPOPolicyWrapper(os.path.join(args.ckpt_dir, "mappo_bh.pt")),
    }

    env = LEOBHEnv("isa", alpha=cfg.ALPHA, seed=0)
    lat_ms = benchmark_mappo_latency(policies["MAPPO-BH"], env)
    print(f"[Latency] MAPPO-BH single-thread 16-sat inference: {lat_ms:.3f} ms/slot (300-slot avg)")

    results = {name: {"rates": cfg.EVAL_RATES_MBPS, "tput_mean": [], "tput_ci": [], "delay_mean": [], "delay_ci": []} for name in policies}

    for rate in cfg.EVAL_RATES_MBPS:
        print(f"\n=== Evaluating @ {rate} Mbps ({args.test_cases} test cases x {args.slots} slots) ===")
        for name, pol in policies.items():
            tputs, delays = [], []
            for tc in range(args.test_cases):
                obs = env.reset(input_rate_mbps=rate, num_users=200, seed=1000 + tc)
                pol.reset()
                tot_bits = 0.0
                d_sum = 0.0
                warmup = 20
                for step_i in range(args.slots + warmup):
                    b, p = pol.act(obs)
                    obs, _, info = env.step(b, p)
                    if step_i >= warmup:
                        tot_bits += info["zeta_t"]
                        d_sum += info["gamma_t"]
                tputs.append((tot_bits / (args.slots * cfg.T_SLOT)) / (env.U * 1e6))
                delays.append(d_sum / args.slots)
            tm, ts = float(np.mean(tputs)), float(1.96 * np.std(tputs, ddof=1) / np.sqrt(len(tputs)))
            dm, ds = float(np.mean(delays)), float(1.96 * np.std(delays, ddof=1) / np.sqrt(len(delays)))
            results[name]["tput_mean"].append(tm)
            results[name]["tput_ci"].append(ts)
            results[name]["delay_mean"].append(dm)
            results[name]["delay_ci"].append(ds)
            print(f"  {name:10s} | Throughput: {tm:6.2f} +/- {ts:4.2f} Mbps | RT Delay: {dm:5.2f} +/- {ds:4.2f} slots")

    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump({"mappo_single_thread_ms": lat_ms, "results": results}, f, indent=2)

    # Plot throughput & delay comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    styles = {
        "P-BH": ("--", "x"),
        "QLF-BH": (":", "v"),
        "USWGP-BH": ("-.", "*"),
        "PPO-BH": ("--", "o"),
        "QMIX-BH": ("-", "D"),
        "MADDPG-BH": (":", "^"),
        "MAPPO-BH": ("-", "s"),
    }
    for name, res in results.items():
        ls, mk = styles[name]
        axes[0].errorbar(res["rates"], res["tput_mean"], yerr=res["tput_ci"], linestyle=ls, marker=mk, capsize=3, label=name)
        axes[1].errorbar(res["rates"], res["delay_mean"], yerr=res["delay_ci"], linestyle=ls, marker=mk, capsize=3, label=name)
    axes[0].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[0].set_ylabel("Throughput per user (Mbps)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Data rate of input traffic per user (Mbps)")
    axes[1].set_ylabel("Average RT queueing delay (time slots)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "bh_comparison.png"))
    print(f"\nSaved results to {args.out_dir}/eval_results.json and {args.out_dir}/bh_comparison.png")


if __name__ == "__main__":
    main()
