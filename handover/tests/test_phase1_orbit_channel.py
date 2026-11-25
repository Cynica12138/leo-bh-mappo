# -*- coding: utf-8 -*-
"""Phase 1 verification: Walker orbit propagation, spherical geometry, exact RVT, and link-level AR(1) channel."""

import math
import time
import unittest
import numpy as np

from handover.config import HandoverConfig
from handover.geometry import DynamicGeometry, place_users_on_sphere
from handover.channel import DynamicChannelModel


class Phase1OrbitChannelTests(unittest.TestCase):
    def test_all_three_constellation_presets_600s(self):
        """Verify 600 s coverage, zero holes, capacity redundancy, and exact RVT monotonicity across all 3 presets."""
        for preset in ("tight_45n", "standard_47n", "dense_35n"):
            cfg = HandoverConfig(constellation_name=preset, episode_duration_s=600)
            geom = DynamicGeometry(cfg)
            t0 = time.perf_counter()
            geom.precompute_episode(t0_sec=1234.0, duration_s=600)
            dt_s = time.perf_counter() - t0
            summary = geom.summarize_episode()
            print(
                f"[Phase 1 {preset:12s}] precompute={dt_s:.2f}s | "
                f"N_act={summary['active_sats_min']}~{summary['active_sats_max']} (mean {summary['active_sats_mean']:.1f}) | "
                f"cap={summary['cap_ratio_min']:.2f}x~{summary['cap_ratio_max']:.2f}x | "
                f"vis/cell={summary['vis_per_cell_min']}~{summary['vis_per_cell_max']} (mean {summary['vis_per_cell_mean']:.2f}) | "
                f"RVT mean={summary['rvt_mean_s']:.1f}s max={summary['rvt_max_s']:.1f}s"
            )
            # Zero coverage holes & feasible capacity at every single second
            self.assertGreaterEqual(summary["vis_per_cell_min"], 2, f"{preset} has <2 visible sats")
            self.assertGreaterEqual(summary["cap_ratio_min"], 1.05, f"{preset} capacity ratio too tight")
            self.assertLessEqual(summary["rvt_max_s"], 170.0, f"{preset} RVT unexpectedly large")

            # Verify strict V_geo == (elev >= 40 deg) and exact 1-second RVT countdown
            snap0 = geom.get_snapshot(100)
            snap1 = geom.get_snapshot(101)
            min_el = math.radians(cfg.THETA_MIN_DEG) - 1e-6
            self.assertTrue(np.all(snap0.elev_rad[snap0.V_geo == 1] >= min_el))
            self.assertTrue(np.all(snap0.elev_rad[snap0.V_geo == 0] < min_el + 2e-6))

            id_to_idx1 = {int(s): i for i, s in enumerate(snap1.active_sat_ids)}
            checked = 0
            for i0, sid in enumerate(snap0.active_sat_ids):
                i1 = id_to_idx1.get(int(sid), -1)
                if i1 < 0:
                    continue
                both_vis = (snap0.V_geo[i0] == 1) & (snap1.V_geo[i1] == 1)
                if np.any(both_vis):
                    np.testing.assert_allclose(snap1.rvt_s[i1, both_vis], snap0.rvt_s[i0, both_vis] - 1.0, atol=1e-5)
                    checked += int(both_vis.sum())
            self.assertGreater(checked, 100)

    def test_link_level_ar1_shadowing_autocorrelation_and_independence(self):
        """Verify per-link (sat_id, cell_id) AR(1) lag-1 autocorrelation matches exp(-1/tau) and differs across sats."""
        cfg = HandoverConfig(
            constellation_name="tight_45n",
            episode_duration_s=300,
            shadow_corr_time_s=10.0,
            shadow_sigma_low_elev_db=3.0,
            shadow_sigma_high_elev_db=3.0,
        )
        geom = DynamicGeometry(cfg)
        geom.precompute_episode(t0_sec=0.0, duration_s=300)
        chan = DynamicChannelModel(cfg, geom.layout)
        rng = np.random.default_rng(42)
        chan.reset(num_users=200, rng=rng)

        # Track consecutive pairs (x_{t-1}, x_t) for links present across steps
        prev_map: dict[int, np.ndarray] = {}
        xs, ys = [], []
        cross_corr_a, cross_corr_b = [], []
        for t in range(301):
            snap = geom.get_snapshot(t)
            links = chan.update_nominal_links(snap, rng)
            sh = links["shadow_db"]
            cur_map = {int(sid): sh[i].copy() for i, sid in enumerate(snap.active_sat_ids)}
            for sid, vec in cur_map.items():
                if sid in prev_map:
                    xs.append(prev_map[sid])
                    ys.append(vec)
            # Check independence across two distinct active satellites on the same cells
            if len(snap.active_sat_ids) >= 2:
                cross_corr_a.append(sh[0])
                cross_corr_b.append(sh[1])
            prev_map = cur_map

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        emp_rho = float(np.corrcoef(x_all, y_all)[0, 1])
        expected_rho = math.exp(-1.0 / 10.0)
        emp_cross = float(np.corrcoef(np.concatenate(cross_corr_a), np.concatenate(cross_corr_b))[0, 1])
        print(f"[Phase 1 AR(1)] emp_rho={emp_rho:.4f} (expected {expected_rho:.4f}), cross-sat corr={emp_cross:.4f}")
        self.assertAlmostEqual(emp_rho, expected_rho, delta=0.02)
        self.assertLess(abs(emp_cross), 0.05)

    def test_3d_bessel_channel_and_imat_sanity(self):
        """Verify 3D Bessel H_sq, nominal rates, and intra-sat coupling I_mat physical values."""
        cfg = HandoverConfig(constellation_name="tight_45n", episode_duration_s=30)
        geom = DynamicGeometry(cfg)
        geom.precompute_episode(t0_sec=500.0, duration_s=30)
        chan = DynamicChannelModel(cfg, geom.layout)
        rng = np.random.default_rng(7)
        chan.reset(num_users=200, rng=rng)

        user_cell = rng.integers(0, cfg.M, size=200, dtype=np.int32)
        _, user_ecef = place_users_on_sphere(geom.layout, user_cell, cfg.CELL_RADIUS_M, rng)
        cell_users_arr = [np.flatnonzero(user_cell == m).astype(np.int32) for m in range(cfg.M)]

        snap = geom.get_snapshot(10)
        links = chan.update_nominal_links(snap, rng)
        vis_rates_mbps = links["nominal_rate_bps"][snap.V_geo == 1] / 1e6
        vis_sinr_db = links["nominal_sinr_db"][snap.V_geo == 1]
        print(
            f"[Phase 1 Link Budget] visible nominal SINR: mean={vis_sinr_db.mean():.1f} dB "
            f"[{vis_sinr_db.min():.1f}, {vis_sinr_db.max():.1f}] | "
            f"nominal rate: mean={vis_rates_mbps.mean():.1f} Mbps"
        )
        # D2C S-band 40 MHz nominal link rate should be realistic (~100-450 Mbps/beam)
        self.assertGreater(float(vis_rates_mbps.mean()), 80.0)
        self.assertLess(float(vis_rates_mbps.mean()), 600.0)

        # Build candidate cell table (up to J=14 closest visible cells per active sat) and compute H_sq / I_mat
        n_act = len(snap.active_sat_ids)
        cand_cells = np.full((n_act, cfg.J), -1, dtype=np.int32)
        for n in range(n_act):
            vis_m = np.flatnonzero(snap.V_geo[n] == 1)
            order = vis_m[np.argsort(snap.slant_dist_m[n, vis_m])][: cfg.J]
            cand_cells[n, : len(order)] = order

        H_sq, I_mat = chan.build_user_channel_tensors(
            snap, links["shadow_db"], cand_cells, user_ecef, user_cell, cell_users_arr
        )
        self.assertEqual(H_sq.shape, (n_act, cfg.J, 200))
        self.assertEqual(I_mat.shape, (n_act, cfg.J, cfg.J))
        self.assertTrue(np.all(np.isfinite(H_sq)) and np.all(H_sq >= 0.0))
        self.assertTrue(np.all(np.isfinite(I_mat)) and np.all(I_mat >= 0.0))
        # Own-cell users must have strictly larger average H_sq than distant non-cell users
        own_gains, far_gains = [], []
        for n in range(n_act):
            for j in range(cfg.J):
                m = int(cand_cells[n, j])
                if m >= 0 and len(cell_users_arr[m]) > 0:
                    us = cell_users_arr[m]
                    other = np.setdiff1d(np.arange(200), us)
                    own_gains.append(float(H_sq[n, j, us].mean()))
                    far_gains.append(float(H_sq[n, j, other].mean()))
        self.assertGreater(np.mean(own_gains), 10.0 * np.mean(far_gains))


if __name__ == "__main__":
    unittest.main(verbosity=2)
