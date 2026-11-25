# LEO Multi-Satellite Hierarchical Load Balancing & Cooperative Beam Hopping (MAPPO-BH)

PyTorch simulation, training, and inference evaluation codebase implementing the hierarchical two-timescale framework for multi-satellite direct-to-cell LEO networks:

- **Long-timescale cell-satellite association**: Improved Simulated Annealing (**ISA**) + baselines (`NLB`, `MEA`, `MMTG`) in [`association.py`](association.py).
- **Short-timescale cooperative beam hopping & power allocation**: **MAPPO-BH** (GAT with distance & common-visibility priors + GRU + hybrid discrete/continuous action heads) + DRL baselines (`PPO-BH`, `QMIX-BH`, `MADDPG-BH`) in [`models.py`](models.py) and heuristic baselines (`P-BH`, `QLF-BH`, `USWGP-BH`) in [`heuristics.py`](heuristics.py).
- **LEO multibeam environment**: 16 satellites, 144 ground cells, 3GPP TR 38.811 circular-aperture Bessel beam pattern, intra/inter-satellite co-channel interference, PF RB scheduling, and 40-slot TTL RT + BULK queueing with Markov-modulated bursty RT arrivals in [`env.py`](env.py).

## Directory Structure

```text
.
├── config.py                  # System, ISA, and MAPPO hyperparameters
├── association.py             # ISA, NLB, MEA, MMTG
├── env.py                     # Vectorized 16-sat / 144-cell LEO simulator
├── heuristics.py              # P-BH, QLF-BH, USWGP-BH
├── models.py                  # MAPPO-BH, PPO-BH, QMIX-BH, MADDPG-BH (PyTorch)
├── train.py                   # Training pipeline exporting .pt checkpoints
├── evaluate.py                # Multi-rate (10-50 Mbps) inference & latency benchmark
├── checkpoints/               # Trained PyTorch checkpoints (.pt)
│   ├── mappo_bh.pt
│   ├── maddpg_bh.pt
│   ├── qmix_bh.pt
│   └── ppo_bh.pt
└── results/                   # Evaluation metrics & comparison plot
    ├── eval_results.json
    └── bh_comparison.png
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Inference Evaluation (using included `.pt` checkpoints)

```bash
python evaluate.py --ckpt-dir checkpoints --test-cases 20 --slots 140 --out-dir results
```

### Retrain All DRL Checkpoints

```bash
python train.py --episodes 100 --slots 64 --ckpt-dir checkpoints
```

## Coordinated scheduling experiment

This independent copy starts from commit `2e1a3a0`. The original geometry, traffic, association, SINR, and PF service are shared by every policy. QLF and USWGP skip empty queues. P-BH retains fixed polling. Snapshot arrays remain valid across resets, including in behavior cloning and replay buffers.

Run the reference and two actor variants with identical rollout budgets.

```bash
python train.py --episodes 80 --slots 128 --ckpt-dir checkpoints/reference
python train.py --episodes 80 --slots 128 --threads 4 --algos mappo \
  --cross-satellite --ckpt-dir checkpoints/cross
python train.py --episodes 80 --slots 128 --algos mappo \
  --cross-satellite --search-teacher --sequence-length 16 --expiry-weight 0.2 \
  --ckpt-dir checkpoints/recurrent
python evaluate.py --test-cases 20 --slots 120 --rates 20 35 50 \
  --ckpt-dir checkpoints/reference --include-bc \
  --mappo-variant MAPPO-Neighbor checkpoints/cross/mappo_bh.pt \
  --mappo-variant MAPPO-Recurrent checkpoints/recurrent/mappo_bh.pt \
  --out-dir results/paired
```

The neighbor variant uses current aggregate neighbor demand and predicted mean link interference. Deployment requires those inputs to be available at decision time. The recurrent variant also unrolls the PPO update over 16 slots, adds an RT expiry penalty, and uses a beam swap to refine its demonstrations.

The optional swap pass ranks feasible same-satellite beam replacements and checks the shortlist against the current joint schedule with the simulator's PF and queue service rules. It transfers the replaced beam's power to the new beam. Apply the same search budget to both initial policies to measure the effect of their choices.

```bash
python evaluate.py --test-cases 20 --slots 120 --rates 50 \
  --ckpt-dir checkpoints/reference --include-bc \
  --mappo-variant MAPPO-Neighbor checkpoints/cross/mappo_bh.pt \
  --search-initializers MAPPO-BH MAPPO-BC MAPPO-Neighbor USWGP-BH --search-steps 2 \
  --out-dir results/search
```

Results include each case's seed, per-user throughput, mean RT queue age, delivered RT packet age, and the expired fraction among RT packets served or expired during measurement. Difference intervals use matched seeds. Decision timing covers policy inference and any requested search on one CPU thread. It excludes environment observation preparation and inter-satellite communications.

The one-swap confirmation uses seeds 2000 through 2019.

```bash
python evaluate.py --test-cases 20 --slots 120 --rates 50 --seed-start 2000 \
  --ckpt-dir checkpoints/reference \
  --mappo-variant MAPPO-Neighbor checkpoints/cross/mappo_bh.pt \
  --search-initializers MAPPO-BH MAPPO-Neighbor USWGP-BH --search-steps 1 \
  --policies MAPPO-BH MAPPO-Neighbor USWGP-BH MAPPO-BH+Swap MAPPO-Neighbor+Swap USWGP-BH+Swap \
  --out-dir results/confirmation
```

Run `python -m unittest -v` to verify snapshots, traffic replay, action feasibility, and agreement between the search scorer and the environment's delivery metric.

## Decoupled Hybrid MAPPO-APG & Cooperative Power GNN

To address the interference-limited bottleneck (`I_intra/N ~ 24 dB`, `I_inter/N ~ 26 dB`) and eliminate the RL-vs-BC stagnation without modifying environment physics, traffic, association, or heuristic baselines, three CTDE upgrades are implemented in `models.py` and `train.py`:

1. **Beam-conditioned hybrid actor (`--power-conditioned`)**: Power logits are predicted after beam selection conditioned on the active beam mask `b_mask`, intra-set interference `I_mat`, and selected-beam demand share, with zero-mean centering over selected beams and wide dynamic range (`[-6.5, 6.5]`) enabling smooth interference silencing.
2. **Exact counterfactual difference rewards (`--difference-reward`)**: During CTDE training, each satellite receives its exact marginal team contribution $D_n = U(a_n, a_{-n}) - U(0, a_{-n})$ paired with an attention-based centralized critic, aligning per-satellite credit assignment with global inter-satellite interference mitigation.
3. **Decoupled discrete PPO + Differentiable-Channel Power APG (`--analytic-ctde`) & 3-step `IterativeCoopPowerGNN` (`--coop-power-gnn`)**: Discrete beam selection (`GAT + GRU + node_disc_head + pair_penalty`) is updated by pure discrete PPO (`lp_d` only), while continuous power allocation is updated via exact differentiable SINR/PF/queue-saturation gradients during CTDE training. At inference time, execution is strictly neural (`< 0.9 ms/slot` on 1 CPU thread, zero test-time search).

### Reproduce Optimized Checkpoints & Evaluations

```bash
# Tier 1: Strictly local-only execution (zero ISL comms at inference)
python train.py --algos mappo --episodes 160 --slots 128 --threads 4 \
  --power-conditioned --difference-reward --analytic-ctde \
  --critic-warmup 8 --lr-actor 3e-4 --lr-power 2e-3 --apg-epochs 3 \
  --rt-diff-weight 0.65 --ent-d 0.008 --val-rates 20 35 50 \
  --ckpt-dir checkpoints/mappo-apg-local

# Tier 2: Existing coordination.py neighbor features
python train.py --algos mappo --episodes 160 --slots 128 --threads 4 \
  --cross-satellite --power-conditioned --difference-reward --analytic-ctde \
  --critic-warmup 8 --lr-actor 3e-4 --lr-power 2e-3 --apg-epochs 3 \
  --rt-diff-weight 0.65 --ent-d 0.008 --val-rates 20 35 50 \
  --ckpt-dir checkpoints/mappo-apg-neighbor

# Tier 3: Cooperative GNN-MAPPO (3-step IterativeCoopPowerGNN), with BC or pure RL from scratch (--bc-epochs 0)
python train.py --algos mappo --episodes 160 --slots 128 --threads 4 \
  --power-conditioned --coop-power-gnn --difference-reward --analytic-ctde \
  --critic-warmup 8 --lr-actor 3e-4 --lr-power 2e-3 --apg-epochs 3 \
  --rt-diff-weight 0.65 --ent-d 0.008 --val-rates 20 35 50 \
  --ckpt-dir checkpoints/mappo-coop-gnn

python train.py --algos mappo --episodes 160 --slots 128 --threads 4 --bc-epochs 0 \
  --power-conditioned --coop-power-gnn --difference-reward --analytic-ctde \
  --critic-warmup 8 --lr-actor 4e-4 --lr-power 2e-3 --apg-epochs 3 \
  --rt-diff-weight 0.65 --ent-d 0.01 --val-rates 20 35 50 \
  --ckpt-dir checkpoints/mappo-coop-gnn-nobc

# Multi-rate paired evaluation (10-50 Mbps, seeds 1000-1019)
python evaluate.py --test-cases 20 --slots 120 --rates 10 20 30 35 40 50 \
  --ckpt-dir checkpoints/reference --include-bc \
  --mappo-variant MAPPO-APG-Local checkpoints/mappo-apg-local/mappo_bh.pt \
  --mappo-variant MAPPO-APG-Neighbor checkpoints/mappo-apg-neighbor/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN checkpoints/mappo-coop-gnn/mappo_bh.pt \
  --out-dir results/optimized-paired

# Held-out confirmation (50 Mbps, seeds 2000-2019)
python evaluate.py --test-cases 20 --slots 120 --rates 50 --seed-start 2000 \
  --ckpt-dir checkpoints/reference \
  --mappo-variant MAPPO-APG-Local checkpoints/mappo-apg-local/mappo_bh.pt \
  --mappo-variant MAPPO-APG-Neighbor checkpoints/mappo-apg-neighbor/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN checkpoints/mappo-coop-gnn/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN-NoBC checkpoints/mappo-coop-gnn-nobc/mappo_bh.pt \
  --search-initializers USWGP-BH MAPPO-CoopGNN --search-steps 1 \
  --policies USWGP-BH USWGP-BH+Swap MAPPO-BH MAPPO-APG-Local MAPPO-APG-Neighbor MAPPO-CoopGNN MAPPO-CoopGNN-NoBC MAPPO-CoopGNN+Swap \
  --out-dir results/optimized-confirmation
```

## Real Starlink SGP4 Orbital Dynamics (`--orbit tle`)

`orbit.py` integrates real CelesTrak Starlink GP (OMM JSON) ephemerides (`data/starlink_shell_gp.json`, 3,469 operational satellites in the 480–570 km shell, snapshot from 2026-11-25) with `sgp4` orbital propagation, IAU-82 GMST TEME→ECEF rotation, and WGS-84 Earth-curved elevation angles over 144 fixed geographic ground cells (16 regional 3×3 clusters with 25 km contiguous 3 dB footprints around `40.0°N, 116.4°E`). During every episode, the 16 active serving satellites continuously move along their real SGP4 orbits (`~7.27 km/s` ECEF speed), dynamically updating slant ranges and off-axis Bessel beam gains.

```bash
# Native real-Starlink SGP4 training
python train.py --orbit tle --algos mappo --episodes 160 --slots 128 --threads 4 \
  --power-conditioned --coop-power-gnn --difference-reward --analytic-ctde \
  --critic-warmup 8 --lr-actor 3e-4 --lr-power 2e-3 --apg-epochs 3 \
  --rt-diff-weight 0.65 --ent-d 0.008 --val-rates 20 35 50 \
  --ckpt-dir checkpoints/tle-mappo-coop-gnn

# Multi-rate paired evaluation (10-50 Mbps) on real Starlink SGP4 orbits
python evaluate.py --orbit tle --test-cases 20 --slots 120 --rates 10 20 30 35 40 50 \
  --ckpt-dir checkpoints/reference \
  --mappo-variant MAPPO-APG-Local checkpoints/mappo-apg-local/mappo_bh.pt \
  --mappo-variant MAPPO-APG-Neighbor checkpoints/mappo-apg-neighbor/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN checkpoints/mappo-coop-gnn/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN-NoBC checkpoints/mappo-coop-gnn-nobc/mappo_bh.pt \
  --out-dir results/starlink-tle-paired

# Held-out confirmation (50 Mbps, seeds 2000-2019) comparing zero-shot vs native TLE-trained checkpoints
python evaluate.py --orbit tle --test-cases 20 --slots 120 --rates 50 --seed-start 2000 \
  --ckpt-dir checkpoints/reference \
  --mappo-variant MAPPO-APG-Local checkpoints/mappo-apg-local/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN-0Shot checkpoints/mappo-coop-gnn/mappo_bh.pt \
  --mappo-variant MAPPO-CoopGNN-NoBC-0Shot checkpoints/mappo-coop-gnn-nobc/mappo_bh.pt \
  --mappo-variant TLE-MAPPO-CoopGNN checkpoints/tle-mappo-coop-gnn/mappo_bh.pt \
  --search-initializers USWGP-BH --search-steps 1 \
  --policies USWGP-BH USWGP-BH+Swap MAPPO-BH MAPPO-APG-Local MAPPO-CoopGNN-0Shot MAPPO-CoopGNN-NoBC-0Shot TLE-MAPPO-CoopGNN \
  --out-dir results/starlink-tle-confirmation
```
