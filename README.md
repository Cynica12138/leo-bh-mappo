# LEO Multi-Satellite Hierarchical Load Balancing & Cooperative Beam Hopping (MAPPO-BH)

PyTorch simulation, training, and inference evaluation codebase implementing the hierarchical two-timescale framework for multi-satellite direct-to-cell LEO networks:

- **Long-timescale cell-satellite association**: Improved Simulated Annealing (**ISA**, Algorithm 1) + baselines (`NLB`, `MEA`, `MMTG`) in [`association.py`](association.py).
- **Short-timescale cooperative beam hopping & power allocation**: **MAPPO-BH** (Algorithm 2: GAT with distance & common-visibility priors + GRU + hybrid discrete/continuous action heads) + DRL baselines (`PPO-BH`, `QMIX-BH`, `MADDPG-BH`) in [`models.py`](models.py) and heuristic baselines (`P-BH`, `QLF-BH`, `USWGP-BH`) in [`heuristics.py`](heuristics.py).
- **LEO multibeam environment**: 16 satellites, 144 ground cells, 3GPP TR 38.811 circular-aperture Bessel beam pattern, intra/inter-satellite co-channel interference, PF RB scheduling, and 40-slot TTL RT + BULK queueing with Markov-modulated bursty RT arrivals in [`env.py`](env.py).

## Directory Structure

```text
.
├── config.py                  # System, ISA, and MAPPO hyperparameters
├── association.py             # ISA (Algorithm 1), NLB, MEA, MMTG
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
