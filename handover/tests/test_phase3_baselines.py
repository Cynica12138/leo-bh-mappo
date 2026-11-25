# -*- coding: utf-8 -*-
"""Phase 3 verification: 600 s comparison of Max-RVT, ISA-HO, and Swap-HO under static & hotspot-drift scenarios."""

import unittest
from handover.config import HandoverConfig
from handover.dynamic_env import DynamicLEOHandoverEnv
from handover.baselines import MaxRVTPolicy, ISAHOPolicy, SwapHOPolicy


def _run_policy(cfg: HandoverConfig, policy, seed: int = 100, t0_sec: float = 300.0) -> dict[str, float]:
    env = DynamicLEOHandoverEnv(cfg, service_mode="surrogate")
    policy.reset()
    out = env.reset(seed=seed, t0_sec=t0_sec)
    while not out.done:
        prop, fallbacks = policy.select_actions(out, env)
        out = env.step(prop, fallback_rankings=fallbacks, bypass_trigger_gate=policy.bypass_trigger_gate)
    return env.get_episode_summary()


class Phase3TraditionalBaselinesTests(unittest.TestCase):
    def test_600s_traditional_baselines_comparison(self):
        """Compare Max-RVT, ISA-HO, and Swap-HO over 600 s on tight_45n and hotspot-drift scenarios."""
        for label, drift, u_cnt, rate in [
            ("tight_45n_default", False, 200, 28.0),
            ("tight_45n_hotspot_drift", True, 220, 30.0),
        ]:
            cfg = HandoverConfig(
                constellation_name="tight_45n",
                episode_duration_s=600,
                num_users=u_cnt,
                input_rate_mbps=rate,
                enable_hotspot_drift=drift,
            )
            policies = [MaxRVTPolicy(), ISAHOPolicy(reassoc_period_s=15), SwapHOPolicy()]
            results = {}
            print(f"\n=== Phase 3 Scenario: {label} (600 s, U={u_cnt}, {rate} Mbps) ===")
            for pol in policies:
                res = _run_policy(cfg, pol, seed=2026, t0_sec=600.0)
                results[pol.name] = res
                print(
                    f"  {pol.name:10s} | HO={res['total_handovers']:5.0f} | PP={res['ping_pong_rate']*100:5.2f}% | "
                    f"Out={res['outage_rate']*100:5.2f}% | RT_drop={res['rt_drop_rate']*100:5.2f}% | "
                    f"RT_del={res['rt_delivered_delay_ms']:4.1f}ms | Tput={res['effective_tput_mbps']:5.2f}Mbps | "
                    f"LoadVar={res['load_variance']:.4f} | Jain={res['jain_fairness']:.3f}"
                )

            # Communication intuition sanity checks:
            # 1. ISA-HO globally re-balances -> lower outage, lower load variance, higher throughput, but MORE handovers
            self.assertLess(results["ISA-HO"]["outage_rate"], results["Max-RVT"]["outage_rate"])
            self.assertLess(results["ISA-HO"]["load_variance"], results["Max-RVT"]["load_variance"])
            self.assertGreater(results["ISA-HO"]["effective_tput_mbps"], results["Max-RVT"]["effective_tput_mbps"])
            self.assertGreater(results["ISA-HO"]["total_handovers"], results["Max-RVT"]["total_handovers"])
            # 2. Swap-HO balances beam load & reduces handovers vs ISA-HO while outperforming Max-RVT on throughput/load/RT delay
            self.assertLess(results["Swap-HO"]["total_handovers"], results["ISA-HO"]["total_handovers"])
            self.assertLess(results["Swap-HO"]["load_variance"], results["Max-RVT"]["load_variance"])
            self.assertGreater(results["Swap-HO"]["effective_tput_mbps"], results["Max-RVT"]["effective_tput_mbps"])
            self.assertLess(results["Swap-HO"]["rt_delivered_delay_ms"], results["Max-RVT"]["rt_delivered_delay_ms"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
