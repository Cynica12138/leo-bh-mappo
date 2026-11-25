# -*- coding: utf-8 -*-
"""Phase 0 verification: frozen paper-simulation MAPPOActor dynamic topology & equivariance stress test."""

import os
import time
import unittest
import numpy as np
import torch

import config as base_cfg
from env import LEOBHEnv
from handover.mappo_adapter import DynamicMAPPOAdapter


def _build_synthetic_batch(rng: np.random.Generator, n_act: int, j_dim: int, node_dim: int, assoc_counts: list[int]):
    """Construct realistic synthetic dynamic-N_act observations for stress testing."""
    f_rich = np.zeros((n_act, j_dim, node_dim), dtype=np.float32)
    i_mat = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)
    omega = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)
    c_mat = np.zeros((n_act, j_dim, j_dim), dtype=np.float32)
    real_mask = np.zeros((n_act, j_dim), dtype=np.float32)
    action_mask = np.zeros((n_act, j_dim), dtype=np.float32)

    for n in range(n_act):
        k_assoc = assoc_counts[n]
        k_real = min(j_dim, max(k_assoc, rng.integers(k_assoc, j_dim + 1)))
        real_mask[n, :k_real] = 1.0
        action_mask[n, :k_assoc] = 1.0
        if k_real > 0:
            # Cols 0,1 = RT & BULK queue; col 2,3 = RT delay & HOL; col 4 = SNR; col 5 = hist tput; col 6 = v_local; col 7 = num_users
            f_rich[n, :k_real, 0] = rng.uniform(0.05, 1.5, size=k_real)
            f_rich[n, :k_real, 1] = rng.uniform(0.05, 2.0, size=k_real)
            f_rich[n, :k_real, 2] = rng.uniform(0.05, 0.7, size=k_real)
            f_rich[n, :k_real, 3] = rng.uniform(0.10, 0.9, size=k_real)
            f_rich[n, :k_real, 4] = rng.uniform(-0.5, 1.2, size=k_real)
            f_rich[n, :k_real, 5] = rng.uniform(0.1, 1.0, size=k_real)
            f_rich[n, :k_real, 6] = rng.uniform(0.1, 0.5, size=k_real)
            f_rich[n, :k_real, 7] = rng.uniform(0.2, 1.4, size=k_real)
            if node_dim > 8:
                f_rich[n, :k_real, 8:] = rng.uniform(0.0, 0.8, size=(k_real, node_dim - 8))
            raw_i = rng.uniform(0.0, 0.3, size=(k_real, k_real)).astype(np.float32)
            np.fill_diagonal(raw_i, 0.0)
            i_mat[n, :k_real, :k_real] = raw_i
            om = rng.uniform(0.0, 0.8, size=(k_real, k_real)).astype(np.float32)
            om = 0.5 * (om + om.T)
            np.fill_diagonal(om, 1.0)
            omega[n, :k_real, :k_real] = om
            cm = rng.uniform(0.0, 0.4, size=(k_real, k_real)).astype(np.float32)
            cm = 0.5 * (cm + cm.T)
            c_mat[n, :k_real, :k_real] = cm

    e_mat = np.zeros((n_act * j_dim, n_act * j_dim), dtype=np.float32)
    if n_act > 0:
        raw_e = rng.uniform(0.0, 0.15, size=(n_act * j_dim, n_act * j_dim)).astype(np.float32)
        np.fill_diagonal(raw_e, 1.0)
        e_mat = raw_e
    return f_rich, i_mat, omega, c_mat, real_mask, action_mask, e_mat


class Phase0MAPPOAdapterTests(unittest.TestCase):
    CKPTS = [
        ("mappo-apg-local", "checkpoints/mappo-apg-local/mappo_bh.pt"),
        ("mappo-coop-gnn", "checkpoints/mappo-coop-gnn/mappo_bh.pt"),
    ]

    def test_dynamic_satellite_churn_and_edge_counts(self):
        """Verify frozen checkpoints handle N_act varying (10->18->12->0), entering/departing sat_ids, and 0..14 cells."""
        rng = np.random.default_rng(2026)
        for label, path in self.CKPTS:
            self.assertTrue(os.path.exists(path), f"Missing checkpoint {path}")
            adapter = DynamicMAPPOAdapter(path)
            adapter.reset()

            # Sequence of changing active satellite sets with edge association counts (0, 1, 2, 3, 4, 14)
            scenarios = [
                ([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [0, 1, 2, 3, 4, 6, 9, 12, 14, 5]),
                ([102, 103, 105, 107, 109, 110, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],
                 [0, 1, 2, 3, 4, 8, 14, 10, 7, 6, 5, 4, 3, 2, 1, 0, 11, 13]),
                ([201, 202, 205, 210, 301, 302, 303, 304, 305, 306, 307, 308],
                 [0, 0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 6]),
                ([], []),
            ]
            for sat_ids, assoc_counts in scenarios:
                n_act = len(sat_ids)
                f_r, i_m, om, c_m, rm, am, e_m = _build_synthetic_batch(
                    rng, n_act, base_cfg.J, adapter.node_dim, assoc_counts
                )
                b, p = adapter.act(
                    sat_ids, f_r, i_m, om, c_m, rm, am,
                    E_mat=e_m if adapter.coop_power_gnn else None,
                    deterministic=True,
                )
                self.assertEqual(b.shape, (n_act, base_cfg.J), label)
                self.assertEqual(p.shape, (n_act, base_cfg.J), label)
                self.assertTrue(np.all(np.isfinite(b)) and np.all(np.isfinite(p)), label)
                if n_act > 0:
                    self.assertTrue(np.all(b <= am + 1e-6), f"{label}: beam outside action_mask")
                    self.assertTrue(np.all(b.sum(axis=1) <= base_cfg.K_MAX + 1e-6), f"{label}: exceeded K_MAX")
                    for n, cnt in enumerate(assoc_counts):
                        if cnt == 0:
                            self.assertEqual(b[n].sum(), 0.0, f"{label}: zero-assoc sat activated beams")
                            self.assertAlmostEqual(float(p[n].sum()), 0.0, places=5)
                        else:
                            self.assertEqual(int(round(b[n].sum())), min(cnt, base_cfg.K_MAX))
                            self.assertAlmostEqual(float(p[n].sum()), 1.0, places=5)
                            self.assertTrue(np.all(p[n][b[n] < 0.5] == 0.0))
                # Verify GRU hidden-state dict tracks exact active satellite set
                if n_act > 0:
                    self.assertEqual(set(adapter.hidden_states.keys()), set(sat_ids))

    def test_strict_permutation_equivariance(self):
        """Shuffling candidate cell slots must yield identically permuted beams and power allocations."""
        for label, path in self.CKPTS:
            adapter_a = DynamicMAPPOAdapter(path)
            adapter_b = DynamicMAPPOAdapter(path)
            env = LEOBHEnv(cross_satellite=adapter_a.cross_satellite, seed=123)
            obs = env.reset(input_rate_mbps=50.0, num_users=200, seed=1234)
            sat_ids = list(range(base_cfg.N))

            b1, p1 = adapter_a.act(
                sat_ids,
                obs["F_rich"],
                obs["I_mat"],
                obs["Omega"],
                obs["C"],
                obs["real_mask"],
                obs["action_mask"],
                E_mat=obs.get("E_mat"),
                deterministic=True,
            )

            # Apply a random permutation to candidate slots for every satellite
            rng = np.random.default_rng(77)
            perms = np.stack([rng.permutation(base_cfg.J) for _ in range(base_cfg.N)], axis=0)
            f_p = np.zeros_like(obs["F_rich"])
            i_p = np.zeros_like(obs["I_mat"])
            om_p = np.zeros_like(obs["Omega"])
            c_p = np.zeros_like(obs["C"])
            rm_p = np.zeros_like(obs["real_mask"])
            am_p = np.zeros_like(obs["action_mask"])
            for n in range(base_cfg.N):
                pr = perms[n]
                f_p[n] = obs["F_rich"][n, pr]
                i_p[n] = obs["I_mat"][n][np.ix_(pr, pr)]
                om_p[n] = obs["Omega"][n][np.ix_(pr, pr)]
                c_p[n] = obs["C"][n][np.ix_(pr, pr)]
                rm_p[n] = obs["real_mask"][n, pr]
                am_p[n] = obs["action_mask"][n, pr]

            e_p = None
            if adapter_b.coop_power_gnn:
                flat_perm = np.concatenate([n * base_cfg.J + perms[n] for n in range(base_cfg.N)])
                e_p = obs["E_mat"][np.ix_(flat_perm, flat_perm)]

            b2, p2 = adapter_b.act(
                sat_ids, f_p, i_p, om_p, c_p, rm_p, am_p, E_mat=e_p, deterministic=True
            )
            for n in range(base_cfg.N):
                pr = perms[n]
                np.testing.assert_allclose(b2[n], b1[n, pr], atol=1e-5, err_msg=f"{label} beam perm mismatch sat {n}")
                np.testing.assert_allclose(p2[n], p1[n, pr], atol=1e-4, err_msg=f"{label} power perm mismatch sat {n}")

    def test_single_thread_latency(self):
        """Measure single-thread batched inference latency for both frozen MAPPO adapters."""
        torch.set_num_threads(1)
        for label, path in self.CKPTS:
            adapter = DynamicMAPPOAdapter(path)
            env = LEOBHEnv(cross_satellite=adapter.cross_satellite, seed=0)
            obs = env.reset(input_rate_mbps=40.0, num_users=200, seed=999)
            sat_ids = list(range(base_cfg.N))
            adapter.reset()
            for _ in range(15):
                adapter.act(
                    sat_ids, obs["F_rich"], obs["I_mat"], obs["Omega"], obs["C"],
                    obs["real_mask"], obs["action_mask"], E_mat=obs.get("E_mat"), deterministic=True
                )
            t0 = time.perf_counter()
            iters = 100
            for _ in range(iters):
                adapter.act(
                    sat_ids, obs["F_rich"], obs["I_mat"], obs["Omega"], obs["C"],
                    obs["real_mask"], obs["action_mask"], E_mat=obs.get("E_mat"), deterministic=True
                )
            ms = (time.perf_counter() - t0) * 1000.0 / iters
            print(f"[Phase 0 Latency] {label:16s}: {ms:.3f} ms/slot (N=16, 1-thread)")
            self.assertLess(ms, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
