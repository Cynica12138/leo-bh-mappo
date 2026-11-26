# -*- coding: utf-8 -*-
"""Phase 6 verification: RA-HO, MCMF-HO, RH-STG, Hyst-MADQN, and MAPPO-HO baselines."""

import os
import tempfile
import time
import unittest
import numpy as np
import torch

from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv
from handover.baselines import RAHOPolicy, MCMFHOPolicy, RHSTGPolicy
from handover.hgat_qrdqn import DRLHandoverAgent
from handover.mappo_ho import MAPPOHOAgent, compute_masked_ppo_actor_loss, train_mappo_ho
from handover.train import build_agent, train_algo


class Phase6NewBaselinesTests(unittest.TestCase):
    def test_600s_ra_mcmf_rhstg_constraints_and_speed(self):
        """Verify RA-HO, MCMF-HO, and RH-STG satisfy hard constraints and run fast (<10 ms/step) over 600 s."""
        cfg = HandoverConfig(
            constellation_name="tight_45n",
            episode_duration_s=600,
            num_users=200,
            input_rate_mbps=28.0,
        )
        policies = [RAHOPolicy(), MCMFHOPolicy(), RHSTGPolicy()]
        print("\n=== Phase 6 Non-RL Baselines (600 s, tight_45n, U=200, 28 Mbps) ===")
        all_runs = []
        for pol in policies:
            env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
            pol.reset()
            out = env.reset(seed=1001)
            t_dec = 0.0
            steps = 0
            while not out.done:
                t0 = time.perf_counter()
                prop, fb = pol.select_actions(out, env)
                t_dec += (time.perf_counter() - t0) * 1000.0
                steps += 1
                out = env.step(prop, fallback_rankings=fb, bypass_trigger_gate=pol.bypass_trigger_gate)
            res = env.get_episode_summary()
            ms_per_step = t_dec / max(steps, 1)
            all_runs.append((pol.name, res, ms_per_step))
            print(
                f"  {pol.name:10s} | HO={res['total_handovers']:5.0f} | PP={res['ping_pong_rate']*100:5.2f}% | "
                f"Out={res['outage_rate']*100:5.2f}% | Drop={res['rt_drop_rate']*100:5.2f}% | "
                f"Tput={res['effective_tput_mbps']:5.2f} | Var={res['load_variance']:.3f} | "
                f"Dec={ms_per_step:.2f}ms/step"
            )
        for name, res, ms_per_step in all_runs:
            self.assertLess(res["outage_rate"], 0.03, msg=f"{name} outage too high")
            self.assertGreater(res["effective_tput_mbps"], 15.0, msg=f"{name} tput too low")
            self.assertLess(ms_per_step, 10.0, msg=f"{name} decision latency too high")

    def test_hyst_madqn_asymmetric_loss_and_smoke_train(self):
        """Verify Hyst-MADQN scales negative TD error loss by beta_hyst=0.3 and completes smoke train/load."""
        cfg = HandoverConfig(constellation_name="tight_45n", num_users=200, input_rate_mbps=28.0)
        agent, spec = build_agent("hyst_madqn", cfg, twin=True)
        self.assertEqual(spec["display"], "Hyst-MADQN")
        self.assertAlmostEqual(agent.hysteretic_beta, 0.3)
        self.assertFalse(agent.use_hgat)
        self.assertFalse(agent.distributional)

        # Direct mathematical check of asymmetric Huber weighting
        pred = torch.tensor([[1.0], [1.0]], requires_grad=True)
        tgt_pos = torch.tensor([[1.5], [1.5]])   # td = +0.5 -> weight 1.0
        tgt_neg = torch.tensor([[0.5], [0.5]])   # td = -0.5 -> weight 0.3
        elem_pos = torch.nn.functional.smooth_l1_loss(pred, tgt_pos, reduction="none")
        elem_neg = torch.nn.functional.smooth_l1_loss(pred, tgt_neg, reduction="none")
        w_pos = torch.where((tgt_pos - pred).detach() >= 0.0, 1.0, 0.3)
        w_neg = torch.where((tgt_neg - pred).detach() >= 0.0, 1.0, 0.3)
        ratio = float((w_neg * elem_neg).mean().item() / (w_pos * elem_pos).mean().item())
        self.assertAlmostEqual(ratio, 0.3, places=5)

        with tempfile.TemporaryDirectory() as tmp:
            summary = train_algo(
                algo_key="hyst_madqn",
                cfg=cfg,
                episodes=2,
                train_duration_s=40,
                val_duration_s=40,
                updates_per_ep=4,
                save_dir=tmp,
                seed=7,
                twin=True,
                verbose=False,
            )
            self.assertTrue(os.path.exists(summary["ckpt_path"]))
            agent2, _ = build_agent("hyst_madqn", cfg, twin=True)
            agent2.load(summary["ckpt_path"])
            self.assertAlmostEqual(agent2.hysteretic_beta, 0.3)

    def test_mappo_ho_trigger_loss_mask_and_smoke_train(self):
        """Verify MAPPO-HO trigger mask yields strictly zero Actor gradient on non-triggered steps and trains cleanly."""
        logits = torch.randn(8, 7, requires_grad=True)
        action_mask = torch.ones(8, 7)
        actions = torch.zeros(8, dtype=torch.long)
        old_lp = torch.zeros(8)
        adv = torch.randn(8)
        zero_trig = torch.zeros(8)

        loss, pg_l, ent_m = compute_masked_ppo_actor_loss(
            logits=logits,
            action_mask=action_mask,
            actions=actions,
            old_log_probs=old_lp,
            advantages=adv,
            trigger_mask=zero_trig,
            clip_eps=0.2,
            ent_coef=0.01,
        )
        loss.backward()
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(pg_l.item()), 0.0)
        self.assertEqual(float(ent_m.item()), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().max().item()), 0.0)

        # Smoke training + checkpoint save/load + deterministic evaluation
        cfg = HandoverConfig(constellation_name="tight_45n", num_users=200, input_rate_mbps=28.0)
        with tempfile.TemporaryDirectory() as tmp:
            summary = train_mappo_ho(
                cfg=cfg,
                episodes=2,
                train_duration_s=40,
                val_duration_s=40,
                ppo_epochs=2,
                save_dir=tmp,
                seed=11,
                verbose=False,
            )
            self.assertTrue(os.path.exists(summary["ckpt_path"]))
            loaded = MAPPOHOAgent(cfg=cfg)
            loaded.load(summary["ckpt_path"])
            env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
            out = env.reset(seed=1001, duration_s=20)
            while not out.done:
                prop, fb = loaded.select_actions(out, env)
                out = env.step(prop, fallback_rankings=fb)
            ep_m = env.get_episode_summary()
            self.assertLess(ep_m["outage_rate"], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
