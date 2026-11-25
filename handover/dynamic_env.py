# -*- coding: utf-8 -*-
"""Unified 1 s macro handover environment integrating orbit, channel, traffic, trigger, admission, and service."""

from dataclasses import dataclass
import numpy as np

from handover.config import HandoverConfig
from handover.geometry import DynamicGeometry, TopologySnapshot
from handover.channel import DynamicChannelModel
from handover.traffic import DynamicTrafficModel, TrafficSnapshot
from handover.trigger import DualStageTrigger, TriggerOutput
from handover.admission import UnifiedAdmissionController, AdmissionResult, solve_isa_feasible
from handover.service_model import SurrogateServiceModel, MicroSlotServiceModel, MacroServiceMetrics


@dataclass
class EnvStepOutput:
    """Structured state & telemetry returned at every 1 s macro handover step."""

    t: int
    done: bool
    snap: TopologySnapshot
    links: dict[str, np.ndarray]
    trigger: TriggerOutput
    admission: AdmissionResult | None
    service: MacroServiceMetrics | None
    d_nm: np.ndarray            # (N_act, M) beam-time demand ratio
    sat_rho: np.ndarray         # (N_act,) satellite beam resource load ratio
    sat_zeta: np.ndarray        # (N_act,) satellite max single-cell overload ratio
    cell_sat_id: np.ndarray     # (M,) current serving global sat_id (-1 if outage)
    Z: np.ndarray               # (N_act, M) current binary association
    ping_pong_flags: np.ndarray # (M,) int32 {0, 1}
    cell_ho_cost: np.ndarray    # (M,) float64 normalized signaling cost


class DynamicLEOHandoverEnv:
    """Two-timescale dynamic LEO handover environment (1 s macro + 20 ms surrogate or 2 ms micro slots).

    Exposes the three core extension interfaces:
      - update_topology(t)
      - update_association(Z_new, active_sat_ids)
      - execute_handover(cell_id, target_sat_id)
    """

    def __init__(
        self,
        cfg: HandoverConfig | None = None,
        service_mode: str = "surrogate",
        force_only_trigger: bool = False,
        traffic_aware_trigger: bool = True,
        ckpt_path: str | None = None,
        cci_sig_ratio: float = 0.24,
        beam_eff_scale: float = 0.78,
    ):
        self.cfg = cfg if cfg is not None else HandoverConfig()
        self.service_mode = service_mode
        self.geom = DynamicGeometry(self.cfg)
        self.channel = DynamicChannelModel(self.cfg, self.geom.layout)
        self.traffic_model = DynamicTrafficModel(self.cfg, self.geom.layout)
        self.trigger = DualStageTrigger(
            self.cfg, force_only=force_only_trigger, traffic_aware=traffic_aware_trigger
        )
        self.admission = UnifiedAdmissionController(self.cfg)

        if service_mode == "surrogate":
            self.service = SurrogateServiceModel(
                self.cfg, self.geom.layout, cci_sig_ratio=cci_sig_ratio, beam_eff_scale=beam_eff_scale
            )
        elif service_mode in ("mappo_local", "mappo_coop", "uswgp_micro"):
            self.service = MicroSlotServiceModel(
                self.cfg, self.geom.layout, self.channel, scheduler_mode=service_mode, ckpt_path=ckpt_path
            )
        else:
            raise ValueError(f"Unknown service_mode: {service_mode}")

        self.rng = np.random.default_rng(0)
        self.t = 0
        self.duration_s = self.cfg.episode_duration_s
        self.traffic: TrafficSnapshot | None = None
        self.snap: TopologySnapshot | None = None
        self.links: dict[str, np.ndarray] | None = None
        self.Z: np.ndarray | None = None
        self.cell_sat_id = np.full(self.cfg.M, -1, dtype=np.int32)
        self.d_nm: np.ndarray | None = None
        self.sat_rho: np.ndarray | None = None
        self.sat_zeta: np.ndarray | None = None
        self.current_trigger: TriggerOutput | None = None

        # Per-cell rolling service stats & ping-pong history
        self.cell_recent_rt_drop = np.zeros(self.cfg.M, dtype=np.float64)
        self.cell_hist_tput_ratio = np.ones(self.cfg.M, dtype=np.float64)
        self.cell_rt_urg = np.zeros(self.cfg.M, dtype=np.float64)
        self.cell_queue_bits = np.zeros(self.cfg.M, dtype=np.float64)
        self._last_ho_from_sat = np.full(self.cfg.M, -1, dtype=np.int32)
        self._last_ho_time = np.full(self.cfg.M, -1000, dtype=np.int32)

        # Episode accumulators
        self.ep_stats: dict = {}

    def _compute_demand_and_loads(
        self,
        t: int,
        Z: np.ndarray,
        nominal_rate_bps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute d_{n,m}(t) (unclipped, bounded queue drain horizon), sat_rho, and sat_zeta."""
        tot_bps, _, _ = self.traffic_model.get_offered_rates_bps(t)
        # Bound instantaneous backlog drain contribution to a finite 2 s horizon so rho stays physically well-conditioned
        eff_q_bits = np.minimum(self.cell_queue_bits, 2.0 * np.maximum(tot_bps, 2.0e6) * self.cfg.T_H)
        req_bps = tot_bps + eff_q_bits / self.cfg.T_H            # (M,)
        d_nm = req_bps[None, :] / (nominal_rate_bps + 1e3)       # (N_act, M)

        n_act = Z.shape[0]
        sat_rho = np.zeros(n_act, dtype=np.float64)
        sat_zeta = np.zeros(n_act, dtype=np.float64)
        for n in range(n_act):
            assoc_m = np.flatnonzero(Z[n] == 1)
            if len(assoc_m) > 0:
                ds = d_nm[n, assoc_m]
                sat_rho[n] = float(ds.sum()) / float(self.cfg.K_MAX)
                sat_zeta[n] = float(ds.max())
        return d_nm, sat_rho, sat_zeta

    def update_topology(self, t: int) -> tuple[TopologySnapshot, dict[str, np.ndarray]]:
        """Core Interface 1: advance orbital geometry & link-level AR(1) shadowing to macro step t."""
        self.t = int(t)
        self.snap = self.geom.get_snapshot(self.t, current_cell_sat_id=self.cell_sat_id)
        self.links = self.channel.update_nominal_links(self.snap, self.rng)
        return self.snap, self.links

    def update_association(self, Z_new: np.ndarray, active_sat_ids: np.ndarray) -> None:
        """Core Interface 2: validate hard constraints and commit new association matrix Z."""
        assert self.snap is not None
        n_act, m_cnt = Z_new.shape
        assert n_act == len(active_sat_ids) and m_cnt == self.cfg.M
        assert np.all(Z_new <= self.snap.V_geo), "Association violates V_geo visibility!"
        assert np.all(Z_new.sum(axis=0) <= 1), "Association violates single-satellite connectivity!"
        assert np.all(Z_new.sum(axis=1) <= self.cfg.J), f"Association violates per-sat cap J={self.cfg.J}!"

        self.Z = Z_new.astype(np.int32).copy()
        new_ids = np.full(m_cnt, -1, dtype=np.int32)
        for m in range(m_cnt):
            s = np.flatnonzero(self.Z[:, m] == 1)
            if len(s) == 1:
                new_ids[m] = int(active_sat_ids[s[0]])
        self.cell_sat_id = new_ids

    def execute_handover(self, cell_id: int, target_sat_id: int, tau_ho_ms: float | None = None) -> None:
        """Core Interface 3: pause cell service for tau_HO interruption WITHOUT resetting queues or HOL."""
        pause_ms = self.cfg.tau_ho_ms if tau_ho_ms is None else float(tau_ho_ms)
        self.service.trigger_handover_pause(int(cell_id), pause_ms)

    def _normalized_ho_cost(self) -> np.ndarray:
        assert self.traffic is not None
        raw = self.cfg.ho_cost_common + self.traffic.cell_user_counts.astype(np.float64) * self.cfg.ho_cost_per_user
        scale = max(float(self.cfg.ho_cost_common + 3.0 * self.cfg.ho_cost_per_user), 1e-6)
        return raw / scale

    def reset(
        self,
        seed: int = 0,
        t0_sec: float | None = None,
        duration_s: int | None = None,
        num_users: int | None = None,
        input_rate_mbps: float | None = None,
    ) -> EnvStepOutput:
        """Reset episode with random orbit phase, hotspot traffic, and P0-safe ISA initial association."""
        self.rng = np.random.default_rng(seed)
        self.duration_s = int(self.cfg.episode_duration_s if duration_s is None else duration_s)
        if t0_sec is None:
            t0_sec = float(self.rng.uniform(0.0, self.cfg.constellation.orbit_period_s))

        self.geom.precompute_episode(t0_sec=t0_sec, duration_s=self.duration_s)
        self.traffic = self.traffic_model.reset(self.rng, num_users=num_users, input_rate_mbps=input_rate_mbps)
        self.channel.reset(self.traffic.U, self.rng)
        self.trigger.reset()
        self.service.reset(self.rng, self.traffic)

        self.cell_sat_id.fill(-1)
        self.cell_recent_rt_drop.fill(0.0)
        self.cell_hist_tput_ratio.fill(1.0)
        self.cell_rt_urg.fill(0.0)
        self.cell_queue_bits.fill(0.0)
        self._last_ho_from_sat.fill(-1)
        self._last_ho_time.fill(-1000)

        self.ep_stats = {
            "steps": 0,
            "total_sim_time_s": 0.0,
            "total_handovers": 0,
            "total_ping_pongs": 0,
            "total_outage_cell_steps": 0,
            "total_delivered_bits": 0.0,
            "total_served_rt_pkts": 0.0,
            "total_expired_rt_pkts": 0.0,
            "total_rt_deliv_delay_weighted_ms": 0.0,
            "load_var_sum": 0.0,
            "jain_sum": 0.0,
            "rho_mean_sum": 0.0,
            "zeta_max_sum": 0.0,
            "rejected_proactive": 0,
            "rejected_mandatory": 0,
        }

        # Step t = 0: topology + P0-safe ISA initial association (shared fair starting point!)
        snap, links = self.update_topology(0)
        tot_bps, _, _ = self.traffic_model.get_offered_rates_bps(0)
        Z0 = solve_isa_feasible(tot_bps / 1e6, snap.V_geo, J=self.cfg.J, elev_rad=snap.elev_rad, rng=self.rng)
        self.update_association(Z0, snap.active_sat_ids)
        self.trigger.update_dwell(np.full(self.cfg.M, -1, dtype=np.int32), self.cell_sat_id)

        # Refresh candidate list so currently serving sat is guaranteed included
        self.snap = self.geom.get_snapshot(0, current_cell_sat_id=self.cell_sat_id)
        self.d_nm, self.sat_rho, self.sat_zeta = self._compute_demand_and_loads(
            0, self.Z, links["nominal_rate_bps"]
        )

        # Short 1-step warmup (no handover) to populate initial queues realistically
        self._run_service(0)
        self.d_nm, self.sat_rho, self.sat_zeta = self._compute_demand_and_loads(
            0, self.Z, links["nominal_rate_bps"]
        )

        trig0 = self.trigger.evaluate(
            self.snap,
            self.cell_sat_id,
            links["nominal_sinr_db"],
            self.sat_rho,
            self.sat_zeta,
            self.cell_rt_urg,
            self.cell_recent_rt_drop,
            sat_counts=self.Z.sum(axis=1),
        )
        self.current_trigger = trig0
        return EnvStepOutput(
            t=0,
            done=False,
            snap=self.snap,
            links=links,
            trigger=trig0,
            admission=None,
            service=None,
            d_nm=self.d_nm.copy(),
            sat_rho=self.sat_rho.copy(),
            sat_zeta=self.sat_zeta.copy(),
            cell_sat_id=self.cell_sat_id.copy(),
            Z=self.Z.copy(),
            ping_pong_flags=np.zeros(self.cfg.M, dtype=np.int32),
            cell_ho_cost=self._normalized_ho_cost(),
        )

    def _run_service(self, t: int, micro_slots_override: int | None = None) -> MacroServiceMetrics:
        assert self.snap is not None and self.links is not None and self.Z is not None and self.traffic is not None
        if isinstance(self.service, SurrogateServiceModel):
            _, rt_bps, bk_bps = self.traffic_model.get_offered_rates_bps(t)
            metrics = self.service.step_macro(
                self.Z,
                self.links["cell_path_gain"],
                rt_bps,
                bk_bps,
                self.traffic.U,
                self.rng,
            )
        else:
            mult = self.traffic_model.cell_rate_multiplier(t)
            metrics = self.service.step_macro(
                self.snap,
                self.Z,
                self.links["shadow_db"],
                self.traffic,
                mult,
                self.rng,
                num_slots=micro_slots_override,
            )

        self.channel.update_ema_interference(metrics.cell_measured_interf_w)
        self.cell_queue_bits = metrics.cell_total_queue_bits
        self.cell_rt_urg = metrics.cell_rt_urg
        self.cell_recent_rt_drop = 0.5 * self.cell_recent_rt_drop + 0.5 * metrics.cell_rt_drop_rate

        tot_bps, _, _ = self.traffic_model.get_offered_rates_bps(t)
        sat_ratio = np.where(tot_bps > 1e3, np.minimum(metrics.cell_delivered_bits / np.maximum(tot_bps, 1e3), 1.5), 1.0)
        self.cell_hist_tput_ratio = 0.6 * self.cell_hist_tput_ratio + 0.4 * sat_ratio
        return metrics

    def step(
        self,
        proposed_target_local: np.ndarray,
        fallback_rankings: list[list[int]] | None = None,
        micro_slots_override: int | None = None,
        bypass_trigger_gate: bool = False,
    ) -> EnvStepOutput:
        """Execute one 1 s macro handover step:
        admission -> handover pause -> service execution -> advance to t+1 topology & trigger.
        """
        assert self.snap is not None and self.links is not None and self.Z is not None and self.traffic is not None
        assert self.current_trigger is not None
        cur_t = self.t
        prev_sat_id = self.cell_sat_id.copy()
        cur_trig = self.current_trigger

        # Enforce non-triggered cells to Stay (unless periodic global baseline like ISA-HO requests bypass)
        if bypass_trigger_gate:
            enforced_prop = proposed_target_local.astype(np.int32)
        else:
            enforced_prop = np.where(cur_trig.i_trig > 0, proposed_target_local, -1).astype(np.int32)

        # Satellite-side unified concurrent admission
        adm = self.admission.admit(
            snap=self.snap,
            prev_cell_sat_id=prev_sat_id,
            proposed_target_local=enforced_prop,
            is_mandatory=cur_trig.i_force > 0,
            rt_urgency=self.cell_rt_urg,
            d_nm=self.d_nm,
            fallback_rankings=fallback_rankings,
        )
        self.update_association(adm.Z_new, self.snap.active_sat_ids)
        self.trigger.update_dwell(prev_sat_id, self.cell_sat_id)

        # Detect ping-pong handovers and apply handover interruption pause (queues preserved!)
        ping_pong = np.zeros(self.cfg.M, dtype=np.int32)
        for m in range(self.cfg.M):
            if adm.handover_flags[m] == 1:
                if (
                    int(self.cell_sat_id[m]) == int(self._last_ho_from_sat[m])
                    and (cur_t - int(self._last_ho_time[m])) <= self.cfg.ping_pong_window_s
                ):
                    ping_pong[m] = 1
                self._last_ho_from_sat[m] = int(prev_sat_id[m])
                self._last_ho_time[m] = cur_t
                self.execute_handover(m, int(self.cell_sat_id[m]))

        # Compute post-admission loads on current topology
        self.d_nm, self.sat_rho, self.sat_zeta = self._compute_demand_and_loads(
            cur_t, self.Z, self.links["nominal_rate_bps"]
        )

        # Execute 1 s of beam-hopping service
        srv = self._run_service(cur_t, micro_slots_override=micro_slots_override)

        # Accumulate episode telemetry
        step_dur_s = (
            float(micro_slots_override) * self.cfg.T_SLOT
            if (micro_slots_override is not None and not isinstance(self.service, SurrogateServiceModel))
            else self.cfg.T_H
        )
        self.ep_stats["steps"] += 1
        self.ep_stats["total_sim_time_s"] += step_dur_s
        self.ep_stats["total_handovers"] += int(adm.handover_flags.sum())
        self.ep_stats["total_ping_pongs"] += int(ping_pong.sum())
        self.ep_stats["total_outage_cell_steps"] += int(adm.outage_flags.sum())
        self.ep_stats["total_delivered_bits"] += float(srv.cell_delivered_bits.sum())
        self.ep_stats["total_served_rt_pkts"] += float(srv.cell_served_rt_pkts.sum())
        self.ep_stats["total_expired_rt_pkts"] += float(srv.cell_expired_rt_pkts.sum())
        self.ep_stats["total_rt_deliv_delay_weighted_ms"] += float(
            np.dot(srv.cell_served_rt_pkts, srv.cell_rt_deliv_delay_ms)
        )
        self.ep_stats["rejected_proactive"] += adm.rejected_proactive
        self.ep_stats["rejected_mandatory"] += adm.rejected_mandatory

        if len(self.sat_rho) > 0:
            self.ep_stats["load_var_sum"] += float(np.var(self.sat_rho))
            rho_sq_sum = float(np.dot(self.sat_rho, self.sat_rho))
            rho_sum = float(self.sat_rho.sum())
            jain = (rho_sum ** 2) / (len(self.sat_rho) * rho_sq_sum) if rho_sq_sum > 1e-12 else 1.0
            self.ep_stats["jain_sum"] += jain
            self.ep_stats["rho_mean_sum"] += float(self.sat_rho.mean())
            self.ep_stats["zeta_max_sum"] += float(self.sat_zeta.max())

        # Advance to next macro step t + 1
        next_t = cur_t + 1
        done = next_t >= self.duration_s
        if not done:
            next_snap, next_links = self.update_topology(next_t)
            # Map existing Z to new active satellite indices for next-step load preview
            Z_next = np.zeros((len(next_snap.active_sat_ids), self.cfg.M), dtype=np.int32)
            id_to_idx = {int(sid): i for i, sid in enumerate(next_snap.active_sat_ids)}
            for m in range(self.cfg.M):
                sid = int(self.cell_sat_id[m])
                if sid >= 0:
                    loc = id_to_idx.get(sid, -1)
                    if loc >= 0 and next_snap.V_geo[loc, m] == 1:
                        Z_next[loc, m] = 1
            self.Z = Z_next
            self.d_nm, self.sat_rho, self.sat_zeta = self._compute_demand_and_loads(
                next_t, self.Z, next_links["nominal_rate_bps"]
            )
            next_trig = self.trigger.evaluate(
                next_snap,
                self.cell_sat_id,
                next_links["nominal_sinr_db"],
                self.sat_rho,
                self.sat_zeta,
                self.cell_rt_urg,
                self.cell_recent_rt_drop,
                sat_counts=self.Z.sum(axis=1),
            )
            self.current_trigger = next_trig
        else:
            next_snap = self.snap
            next_links = self.links
            next_trig = cur_trig
            self.current_trigger = next_trig

        return EnvStepOutput(
            t=next_t,
            done=done,
            snap=next_snap,
            links=next_links,
            trigger=next_trig,
            admission=adm,
            service=srv,
            d_nm=self.d_nm.copy(),
            sat_rho=self.sat_rho.copy(),
            sat_zeta=self.sat_zeta.copy(),
            cell_sat_id=self.cell_sat_id.copy(),
            Z=self.Z.copy(),
            ping_pong_flags=ping_pong,
            cell_ho_cost=self._normalized_ho_cost(),
        )

    def get_episode_summary(self) -> dict[str, float]:
        """Return the 8 core Chapter 4 paper evaluation metrics for the finished episode."""
        s = self.ep_stats
        steps = max(s["steps"], 1)
        u_cnt = self.traffic.U if self.traffic is not None else self.cfg.num_users
        ho = s["total_handovers"]
        rt_exit = s["total_served_rt_pkts"] + s["total_expired_rt_pkts"]
        return {
            "total_handovers": float(ho),
            "ping_pong_rate": float(s["total_ping_pongs"]) / max(ho, 1),
            "outage_rate": float(s["total_outage_cell_steps"]) / float(steps * self.cfg.M),
            "rt_drop_rate": float(s["total_expired_rt_pkts"]) / max(rt_exit, 1.0),
            "rt_delivered_delay_ms": (
                float(s["total_rt_deliv_delay_weighted_ms"]) / max(s["total_served_rt_pkts"], 1.0)
            ),
            "effective_tput_mbps": float(s["total_delivered_bits"]) / (max(s["total_sim_time_s"], 1e-9) * max(u_cnt, 1) * 1e6),
            "load_variance": float(s["load_var_sum"]) / steps,
            "jain_fairness": float(s["jain_sum"]) / steps,
            "mean_sat_rho": float(s["rho_mean_sum"]) / steps,
            "rejected_proactive": float(s["rejected_proactive"]),
            "rejected_mandatory": float(s["rejected_mandatory"]),
        }
