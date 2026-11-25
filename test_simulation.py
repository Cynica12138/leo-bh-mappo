"""Regression tests for saved observations and coordinated scheduling."""

import copy
import unittest

import numpy as np
import torch

import config as cfg
from env import LEOBHEnv
from heuristics import QLFBeamHopping, USWGPBeamHopping
from models import MAPPOActor
from schedule_search import SlotService, best_global_swap
from train import compute_counterfactual_rewards


class SimulationTests(unittest.TestCase):
    def test_reset_preserves_observation(self):
        env = LEOBHEnv()
        previous = env.reset(50, 200, 777)
        saved = {key: previous[key].copy() for key in ("I_mat", "Omega", "action_mask")}
        current = env.reset(50, 200, 778)
        for key, values in saved.items():
            np.testing.assert_array_equal(previous[key], values)
            self.assertFalse(np.shares_memory(previous[key], current[key]))

    def test_seed_replays_arrivals_with_different_actions(self):
        first = LEOBHEnv()
        second = LEOBHEnv(cross_satellite=True)
        a, b = [env.reset(50, 200, 1234) for env in (first, second)]
        for _ in range(16):
            a, _, info_a = first.step(*QLFBeamHopping().act(a))
            b, _, info_b = second.step(*USWGPBeamHopping().act(b))
            self.assertEqual(info_a["arrived_rt"], info_b["arrived_rt"])
            np.testing.assert_array_equal(first.rt_buckets[:, 0], second.rt_buckets[:, 0])
            np.testing.assert_array_equal(first.cell_on, second.cell_on)
            self.assertEqual(first.rng.bit_generator.state, second.rng.bit_generator.state)

    def test_swap_preserves_action_budget_and_matches_service(self):
        env = LEOBHEnv()
        obs = env.reset(50, 200, 1234)
        policy = USWGPBeamHopping()
        for _ in range(8):
            obs, _, _ = env.step(*policy.act(obs))
        b, p = policy.act(obs)
        model = SlotService(env)
        initial_watts = p.reshape(1, -1).astype(float) * cfg.P_TOT_W
        initial_score = model.scores(initial_watts, None, 0.5)[0]
        b, p = best_global_swap(env, b, p)
        np.testing.assert_array_less(b.sum(axis=1), cfg.K_MAX + 1)
        self.assertTrue(np.all(b <= obs["action_mask"]))
        np.testing.assert_allclose(p.sum(axis=1), (b.sum(axis=1) > 0).astype(float), atol=1e-7)
        watts = p.reshape(1, -1).astype(float) * cfg.P_TOT_W
        self.assertGreaterEqual(model.scores(watts, None, 0.5)[0] + 1e-6, initial_score)
        expected = model.scores(watts, None, 0)[0] * cfg.L_P
        _, _, info = copy.deepcopy(env).step(b, p)
        np.testing.assert_allclose(expected, info["zeta_t"], rtol=1e-7)

    def test_power_conditioned_and_coop_gnn_actor_feasibility(self):
        for coop_gnn in (False, True):
            env = LEOBHEnv(cross_satellite=coop_gnn)
            obs = env.reset(50, 200, 1234)
            node_dim = obs["F_rich"].shape[-1]
            actor = MAPPOActor(node_dim=node_dim, power_conditioned=True, coop_power_gnn=coop_gnn)
            h = torch.zeros(1, cfg.N, cfg.HIDDEN_DIM)
            e_mat = torch.from_numpy(obs["E_mat"]) if coop_gnn else None
            b, p, j_seq, y, lp, h_next = actor.act(
                torch.from_numpy(obs["F_rich"]),
                torch.from_numpy(obs["I_mat"]),
                torch.from_numpy(obs["Omega"]),
                torch.from_numpy(obs["C"]),
                torch.from_numpy(obs["real_mask"]),
                torch.from_numpy(obs["action_mask"]),
                h,
                deterministic=False,
                E_mat=e_mat,
                power_deterministic=True,
            )
            b_np, p_np = b.numpy(), p.numpy()
            np.testing.assert_array_less(b_np.sum(axis=1), cfg.K_MAX + 1)
            self.assertTrue(np.all(b_np <= obs["action_mask"] + 1e-6))
            np.testing.assert_allclose(p_np.sum(axis=1), (b_np.sum(axis=1) > 0).astype(float), atol=1e-5)
            # Discrete evaluation must match rollout log-prob when power_deterministic=True
            lp_eval, ent_d = actor.evaluate_discrete_actions(
                torch.from_numpy(obs["F_rich"]),
                torch.from_numpy(obs["I_mat"]),
                torch.from_numpy(obs["Omega"]),
                torch.from_numpy(obs["C"]),
                torch.from_numpy(obs["real_mask"]),
                torch.from_numpy(obs["action_mask"]),
                h,
                j_seq,
            )
            np.testing.assert_allclose(lp.numpy(), lp_eval.detach().numpy(), atol=1e-5)
            self.assertTrue(np.all(np.isfinite(compute_counterfactual_rewards(env, p_np))))


if __name__ == "__main__":
    unittest.main()
