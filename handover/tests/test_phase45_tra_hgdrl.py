# -*- coding: utf-8 -*-
"""Phase 4 & 5 verification: HGAT permutation equivariance, CVaR interpolation, O(1) reward, SMDP buffer, and training smoke test."""

import os
import tempfile
import unittest
import numpy as np
import torch

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv
from handover.graph import build_cell_graph_obs, collate_cell_graphs
from handover.reward import compute_step_rewards
from handover.replay_buffer import SMDPTracker, StratifiedSMDPReplayBuffer
from handover.hgat_qrdqn import (
    TRAHGDRLNet,
    compute_approx_cvar,
    quantile_huber_loss,
)
from handover.train import train_algo, build_agent


class Phase45TRAHGDRLTests(unittest.TestCase):
    def test_approx_cvar_exact_continuous_interpolation(self):
        """Verify continuous linear-interpolation CVaR for integer & non-integer alpha * N_q."""
        # Single batch, 2 actions, N_q = 4 unsorted quantiles
        q = torch.tensor([[[4.0, 1.0, 3.0, 2.0], [10.0, -5.0, 0.0, 5.0]]], dtype=torch.float32)
        # alpha = 1.0 -> exact mean
        c_mean = compute_approx_cvar(q, torch.tensor([1.0]))
        torch.testing.assert_close(c_mean, torch.tensor([[2.5, 2.5]]))
        # alpha = 0.5 -> bottom 2 quantiles: [1, 2] -> 1.5; [-5, 0] -> -2.5
        c_half = compute_approx_cvar(q, torch.tensor([0.5]))
        torch.testing.assert_close(c_half, torch.tensor([[1.5, -2.5]]))
        # Non-integer alpha * N_q = 0.6 * 4 = 2.4 -> (q_(1) + q_(2) + 0.4 * q_(3)) / 2.4
        c_frac = compute_approx_cvar(q, torch.tensor([0.6]))
        expected_0 = (1.0 + 2.0 + 0.4 * 3.0) / 2.4
        expected_1 = (-5.0 + 0.0 + 0.4 * 5.0) / 2.4
        torch.testing.assert_close(c_frac, torch.tensor([[expected_0, expected_1]]), atol=1e-5, rtol=1e-5)

    def test_hgat_qrdqn_candidate_permutation_equivariance(self):
        """Shuffling candidate satellite slots must yield identically permuted quantile distributions."""
        torch.manual_seed(0)
        cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=20)
        env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
        out = env.reset(seed=11, t0_sec=100.0)
        obs = build_cell_graph_obs(0, out, env, traffic_aware=True)

        net = TRAHGDRLNet(hidden_dim=32, num_layers=2, num_quantiles=16, use_hgat=True, distributional=True)
        net.eval()

        batch1 = {k: torch.from_numpy(v) for k, v in collate_cell_graphs([obs]).items()}
        with torch.no_grad():
            q1, _ = net(batch1)  # (1, 1+C, 16)

        # Permute candidate satellite order 0..C_MAX-1
        perm = np.array([3, 0, 5, 1, 4, 2], dtype=np.int64)
        inv_perm = np.argsort(perm)
        obs_p = obs.copy()
        obs_p.cand_local_sats = obs.cand_local_sats[perm]
        obs_p.x_sat = obs.x_sat[perm]
        obs_p.sat_mask = obs.sat_mask[perm]
        obs_p.e_cov = obs.e_cov[perm]
        obs_p.m_cov = obs.m_cov[perm]
        obs_p.e_serv = obs.e_serv[perm]
        obs_p.m_serv = obs.m_serv[perm]
        obs_p.action_mask = np.concatenate([[obs.action_mask[0]], obs.action_mask[1:][perm]])
        obs_p.cur_sat_cand_pos = int(inv_perm[obs.cur_sat_cand_pos]) if obs.cur_sat_cand_pos >= 0 else -1

        batch2 = {k: torch.from_numpy(v) for k, v in collate_cell_graphs([obs_p]).items()}
        with torch.no_grad():
            q2, _ = net(batch2)

        # Action 0 (Stay) must be invariant; actions 1..C must match permutation
        torch.testing.assert_close(q2[:, 0, :], q1[:, 0, :], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(q2[:, 1:, :], q1[:, 1:, :][:, perm, :], atol=1e-5, rtol=1e-5)

    def test_smdp_replay_deep_copy_and_counterfactual_reward(self):
        """Verify SMDP buffer isolates array mutations and O(1) counterfactual reward is finite & zero for Stay."""
        cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=15)
        env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
        out = env.reset(seed=5, t0_sec=50.0)
        obs0 = build_cell_graph_obs(0, out, env)
        saved_x = obs0.x_sat.copy()

        tracker = SMDPTracker(cfg.M, gamma=0.99, max_step=5)
        tracker.record_decision(0, obs0, action=0)
        # Mutate original obs0 in place -> tracker copy must remain intact
        obs0.x_sat.fill(-999.0)

        prev_out = out
        out = env.step(np.full(cfg.M, -1, dtype=np.int32))
        r_tot, r_loc, r_coop = compute_step_rewards(prev_out, out, env)
        self.assertTrue(np.all(np.isfinite(r_tot)))
        # Cells that Stayed on their visible satellite must have r_coop == 0
        stayed = (out.admission.handover_flags == 0) & (out.admission.outage_flags == 0)
        np.testing.assert_allclose(r_coop[stayed], 0.0, atol=1e-6)

        tracker.accumulate_step_rewards(r_tot)
        tr = tracker.finalize(0, build_cell_graph_obs(0, out, env), done=False)
        self.assertIsNotNone(tr)
        np.testing.assert_array_equal(tr.obs.x_sat, saved_x)

        buf = StratifiedSMDPReplayBuffer(capacity=100)
        buf.push(tr)
        sample = buf.sample(4, np.random.default_rng(0))
        self.assertEqual(sample["action"].shape, (4,))
        self.assertEqual(sample["obs"]["action_mask"].shape, (4, 1 + cfg.C_MAX))

    def test_training_smoke_cell_rl_mahgdqn_and_trahgdrl(self):
        """Smoke-test training + checkpoint save/load for Cell-RL, MA-HGDQN, and TRA-HGDRL (two-stage CVaR)."""
        cfg = HandoverConfig(
            constellation_name="tight_45n",
            num_users=180,
            input_rate_mbps=25.0,
            hgat_hidden_dim=32,
            num_quantiles=16,
            batch_size=32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for algo in ("cell_rl", "ma_hgdqn", "tra_hgdrl"):
                res = train_algo(
                    algo_key=algo,
                    cfg=cfg,
                    episodes=3,
                    train_duration_s=40,
                    val_duration_s=40,
                    updates_per_ep=5,
                    save_dir=tmpdir,
                    seed=123,
                    verbose=False,
                )
                self.assertTrue(os.path.exists(res["ckpt_path"]))
                self.assertTrue(np.isfinite(res["best_val_ret"]))
                # Verify checkpoint reload & deterministic inference
                agent, _ = build_agent(algo, cfg)
                agent.load(res["ckpt_path"])
                env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
                out = env.reset(seed=77, duration_s=15)
                prop, fb = agent.select_actions(out, env)
                self.assertEqual(prop.shape, (cfg.M,))
                self.assertEqual(len(fb), cfg.M)
            print("[Phase 4 & 5 Smoke] Cell-RL, MA-HGDQN, and TRA-HGDRL (2-stage CVaR) all trained & reloaded cleanly!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
