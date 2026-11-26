# -*- coding: utf-8 -*-
"""Generate publication-quality Chapter 4 figures (PNG + PDF) and 11-method Markdown summary tables.

Organized into the two-group comparison architecture + unified 11-method overview:
  - Group 1 (Traditional, Graph Theory & Optimization vs. Proposed):
      Max-RVT, ISA-HO, Swap-HO, RA-HO, MCMF-HO, RH-STG, TRA-HGDRL
  - Group 2 (Multi-Agent Reinforcement Learning):
      MAPPO-HO, Hyst-MADQN, Cell-RL, MA-HGDQN, TRA-HGDRL
"""

import argparse
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Clean IEEE-style visual configuration
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

GROUP1_ORDER = ["Max-RVT", "ISA-HO", "Swap-HO", "RA-HO", "MCMF-HO", "RH-STG", "TRA-HGDRL"]
GROUP2_ORDER = ["MAPPO-HO", "Hyst-MADQN", "Cell-RL", "MA-HGDQN", "TRA-HGDRL"]
ALL_11_ORDER = [
    "Max-RVT",
    "ISA-HO",
    "Swap-HO",
    "RA-HO",
    "MCMF-HO",
    "RH-STG",
    "MAPPO-HO",
    "Hyst-MADQN",
    "Cell-RL",
    "MA-HGDQN",
    "TRA-HGDRL",
]
ALGO_ORDER = ALL_11_ORDER

ALGO_COLORS = {
    "Max-RVT": "#7F7F7F",
    "ISA-HO": "#1F77B4",
    "Swap-HO": "#FF7F0E",
    "RA-HO": "#17BECF",
    "MCMF-HO": "#8C564B",
    "RH-STG": "#BCBD22",
    "MAPPO-HO": "#E377C2",
    "Hyst-MADQN": "#1B9E77",
    "Cell-RL": "#2CA02C",
    "MA-HGDQN": "#9467BD",
    "TRA-HGDRL": "#D62728",
}
ALGO_MARKERS = {
    "Max-RVT": "s",
    "ISA-HO": "^",
    "Swap-HO": "D",
    "RA-HO": "X",
    "MCMF-HO": "p",
    "RH-STG": "h",
    "MAPPO-HO": "<",
    "Hyst-MADQN": ">",
    "Cell-RL": "v",
    "MA-HGDQN": "P",
    "TRA-HGDRL": "o",
}

ABL_ORDER = ["TRA-HGDRL", "w/o Risk", "w/o DistRL", "w/o HGAT", "w/o Traffic", "w/o Trigger"]
ABL_COLORS = {
    "TRA-HGDRL": "#D62728",
    "w/o Risk": "#E377C2",
    "w/o DistRL": "#9467BD",
    "w/o HGAT": "#2CA02C",
    "w/o Traffic": "#8C564B",
    "w/o Trigger": "#17BECF",
}


def _save_fig(fig, fig_dir: str, stem: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"{stem}.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, f"{stem}.pdf"))
    plt.close(fig)
    print(f"Saved figure -> {os.path.join(fig_dir, stem)}.png/.pdf")


def plot_fig1_convergence(ckpt_dir: str, fig_dir: str) -> None:
    files = [
        ("MAPPO-HO", "mappo_ho_history.json"),
        ("Hyst-MADQN", "hyst_madqn_history.json"),
        ("Cell-RL", "cell_rl_history.json"),
        ("MA-HGDQN", "ma_hgdqn_history.json"),
        ("TRA-HGDRL", "tra_hgdrl_history.json"),
    ]
    curves = {}
    for label, fname in files:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                curves[label] = json.load(f)["history"]
    if not curves:
        print("[Skip Fig 1] No *_history.json found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    panels = [
        ("val_ret", "Validation Episode Return", 1.0, False),
        ("outage_rate", "Service Outage Rate (%)", 100.0, True),
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0, True),
        ("total_handovers", "Total Handovers (300 s val)", 1.0, True),
    ]
    for ax, (key, ylabel, scale, in_val_metrics) in zip(axes.flat, panels):
        for label, hist in curves.items():
            xs, ys = [], []
            for rec in hist:
                if "val_ret" in rec:
                    xs.append(rec["episode"])
                    val = rec["val_metrics"][key] if in_val_metrics else rec[key]
                    ys.append(float(val) * scale)
            if xs:
                lw = 2.4 if label == "TRA-HGDRL" else 1.75
                ax.plot(
                    xs,
                    ys,
                    label=label,
                    color=ALGO_COLORS[label],
                    marker=ALGO_MARKERS[label],
                    markersize=4.8,
                    linewidth=lw,
                )
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels), frameon=True, fontsize=9)
    fig.suptitle("Training Convergence of Multi-Agent DRL Handover Policies (tight_45n)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save_fig(fig, fig_dir, "fig1_training_convergence")


def plot_fig1b_reward_curves(ckpt_dir: str, fig_dir: str) -> None:
    """Dedicated 120-episode training (10-ep moving avg) and validation reward curves for all 5 DRL methods."""
    files = [
        ("MAPPO-HO", "mappo_ho_history.json", ALGO_COLORS["MAPPO-HO"], "-"),
        ("Hyst-MADQN", "hyst_madqn_history.json", ALGO_COLORS["Hyst-MADQN"], "-"),
        ("Cell-RL", "cell_rl_history.json", ALGO_COLORS["Cell-RL"], "-"),
        ("MA-HGDQN", "ma_hgdqn_history.json", ALGO_COLORS["MA-HGDQN"], "-"),
        ("TRA-HGDRL", "tra_hgdrl_history.json", ALGO_COLORS["TRA-HGDRL"], "-"),
    ]
    loaded = []
    for label, fname, color, ls in files:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                loaded.append((label, json.load(f)["history"], color, ls))
    if not loaded:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    win = 10
    all_ma_vals = []
    for label, hist, color, ls in loaded:
        eps = np.array([r["episode"] for r in hist], dtype=np.float64)
        rets = np.array([r["train_ret"] for r in hist], dtype=np.float64)
        ma = np.array([rets[max(0, i - win + 1) : i + 1].mean() for i in range(len(rets))])
        all_ma_vals.extend(ma.tolist())
        lw = 2.5 if label == "TRA-HGDRL" else 1.75
        ax1.plot(eps, rets, color=color, alpha=0.12, linewidth=0.9)
        ax1.plot(eps, ma, label=label, color=color, linestyle=ls, linewidth=lw)

        veps = [r["episode"] for r in hist if "val_ret" in r]
        vrets = [r["val_ret"] for r in hist if "val_ret" in r]
        ax2.plot(
            veps,
            vrets,
            label=label,
            color=color,
            linestyle=ls,
            marker=ALGO_MARKERS[label],
            markersize=4.5,
            linewidth=lw,
        )

    # Mark two-stage CVaR annealing phases for TRA-HGDRL (ep 42 warmup end, ep 72 full CVaR)
    for ax in (ax1, ax2):
        ax.axvline(42, color="#555555", linestyle=":", linewidth=1.1, alpha=0.8)
        ax.axvline(72, color="#555555", linestyle=":", linewidth=1.1, alpha=0.8)

    y_lo, y_hi = ax1.get_ylim()
    y_top = y_hi + 0.16 * (y_hi - y_lo)
    ax1.set_ylim(y_lo, y_top)
    txt_y = y_top - 0.03 * (y_top - y_lo)
    ax1.text(21, txt_y, "Stage 1\nRisk-Neutral", ha="center", va="top", fontsize=8, color="#444444")
    ax1.text(57, txt_y, "CVaR\nAnneal", ha="center", va="top", fontsize=8, color="#444444")
    ax1.text(96, txt_y, "Stage 2\nFull Adaptive CVaR", ha="center", va="top", fontsize=8, color="#444444")

    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Training Episode Return (10-ep MA)")
    ax1.set_title("(a) Training Reward Curve (240 s Randomized Load)")

    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Validation Episode Return (300 s Fixed Seeds)")
    ax2.set_title("(b) Deterministic Validation Reward Curve")

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=len(labels), frameon=True, fontsize=9)
    fig.suptitle("Training & Validation Reward Convergence (120 Episodes, tight_45n)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save_fig(fig, fig_dir, "fig1b_reward_curves")


def _plot_main_bar_grid(
    res: dict,
    algos: list[str],
    suptitle: str,
    fig_dir: str,
    stem: str,
    figsize: tuple[float, float] = (13.8, 7.6),
    split_after: int | None = None,
) -> None:
    if not algos:
        return
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    metrics = [
        ("outage_rate", "Service Outage Rate (%)", 100.0, "{:.2f}%"),
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0, "{:.2f}%"),
        ("effective_tput_mbps", "Effective Throughput (Mbps)", 1.0, "{:.2f}"),
        ("total_handovers", "Total Handovers (600 s)", 1.0, "{:.0f}"),
        ("ping_pong_rate", "Ping-Pong Rate (%)", 100.0, "{:.1f}%"),
        ("load_variance", "Satellite Load Variance", 1.0, "{:.3f}"),
    ]
    x = np.arange(len(algos))
    txt_fs = 7.0 if len(algos) >= 9 else 7.8
    for ax, (key, title, scale, fmt) in zip(axes.flat, metrics):
        means = [res[a]["mean"][key] * scale for a in algos]
        cis = [res[a]["ci95"][key] * scale for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        bars = ax.bar(x, means, yerr=cis, capsize=3.0, color=colors, alpha=0.92, edgecolor="black", linewidth=0.55)
        for bar, a in zip(bars, algos):
            if a == "TRA-HGDRL":
                bar.set_linewidth(1.5)
        if split_after is not None and 0 < split_after < len(algos):
            ax.axvline(split_after - 0.5, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        ymax = max(m + c for m, c in zip(means, cis))
        ymin = 0.0 if key != "effective_tput_mbps" else max(0.0, min(means) * 0.88)
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin, ymax + span * 0.22)
        for bar, m, c in zip(bars, means, cis):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                m + c + span * 0.025,
                fmt.format(m),
                ha="center",
                va="bottom",
                fontsize=txt_fs,
                rotation=25 if len(algos) >= 9 else 0,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=28 if len(algos) >= 8 else 20, ha="right")
        ax.set_title(title)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, stem)


def plot_fig2_main_comparison(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite1_main_600s.json")
    if not os.path.exists(p):
        print("[Skip Fig 2] Missing suite1_main_600s.json")
        return None
    with open(p) as f:
        data = json.load(f)
    res = data["results"]

    g1 = [a for a in GROUP1_ORDER if a in res]
    g2 = [a for a in GROUP2_ORDER if a in res]
    all_a = [a for a in ALL_11_ORDER if a in res]

    _plot_main_bar_grid(
        res,
        g1,
        "Group 1: Traditional, Graph-Theoretic & Optimization Baselines vs. TRA-HGDRL (600 s, 5 Seeds, 95% CI)",
        fig_dir,
        "fig2a_main_traditional",
        figsize=(13.8, 7.6),
    )
    _plot_main_bar_grid(
        res,
        g2,
        "Group 2: Multi-Agent Reinforcement Learning Baselines vs. TRA-HGDRL (600 s, 5 Seeds, 95% CI)",
        fig_dir,
        "fig2b_main_rl",
        figsize=(13.0, 7.5),
    )
    trad_cnt = sum(1 for a in ["Max-RVT", "ISA-HO", "Swap-HO", "RA-HO", "MCMF-HO", "RH-STG"] if a in res)
    _plot_main_bar_grid(
        res,
        all_a,
        "Unified 11-Method 600 s Handover Performance Comparison (tight_45n, 5 Seeds, 95% CI)",
        fig_dir,
        "fig2_main_comparison",
        figsize=(15.5, 8.2),
        split_after=trad_cnt,
    )
    return data


def plot_fig3_ablation(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite2_ablation_600s.json")
    if not os.path.exists(p):
        print("[Skip Fig 3] Missing suite2_ablation_600s.json")
        return None
    with open(p) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    scenarios = [("standard", "Standard (U=200, 28 Mbps)"), ("stress", "Drift + 1.5x Shadowing (U=220, 32 Mbps)")]
    metrics = [
        ("outage_rate", "Outage Rate (%)", 100.0, "{:.2f}%"),
        ("rt_drop_rate", "RT Drop Rate (%)", 100.0, "{:.2f}%"),
        ("effective_tput_mbps", "Throughput (Mbps)", 1.0, "{:.2f}"),
    ]
    for row_idx, (sc_key, sc_title) in enumerate(scenarios):
        res = data[sc_key]
        abls = [a for a in ABL_ORDER if a in res]
        x = np.arange(len(abls))
        for col_idx, (m_key, m_title, scale, fmt) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            means = [res[a]["mean"][m_key] * scale for a in abls]
            cis = [res[a]["ci95"][m_key] * scale for a in abls]
            colors = [ABL_COLORS[a] for a in abls]
            bars = ax.bar(x, means, yerr=cis, capsize=3, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
            ymax = max(m + c for m, c in zip(means, cis))
            ymin = 0.0 if m_key != "effective_tput_mbps" else min(means) * 0.92
            ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.20)
            for bar, m, c in zip(bars, means, cis):
                ax.text(bar.get_x() + bar.get_width() / 2.0, m + c + (ymax - ymin) * 0.02, fmt.format(m), ha="center", va="bottom", fontsize=7.8)
            ax.set_xticks(x)
            ax.set_xticklabels(abls, rotation=22, ha="right")
            ax.set_title(f"{sc_title}\n{m_title}", fontsize=9.8)
    fig.suptitle("Ablation Study of TRA-HGDRL Components (600 s, 5 Seeds, 95% CI)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "fig3_ablation_study")
    return data


def _plot_load_group(data: dict, group_order: list[str], suptitle: str, fig_dir: str, stem: str) -> None:
    rates = [int(r) for r in data["rates"]]
    by_rate = data["by_rate"]
    algos = [a for a in group_order if a in by_rate[str(rates[0])]]
    if not algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))
    panels = [
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0),
        ("effective_tput_mbps", "Effective Throughput (Mbps)", 1.0),
        ("outage_rate", "Service Outage Rate (%)", 100.0),
        ("load_variance", "Satellite Load Variance", 1.0),
    ]
    for ax, (m_key, ylabel, scale) in zip(axes.flat, panels):
        for a in algos:
            ys = [by_rate[str(r)][a]["mean"][m_key] * scale for r in rates]
            es = [by_rate[str(r)][a]["ci95"][m_key] * scale for r in rates]
            lw = 2.5 if a == "TRA-HGDRL" else 1.7
            ax.errorbar(
                rates,
                ys,
                yerr=es,
                label=a,
                color=ALGO_COLORS[a],
                marker=ALGO_MARKERS[a],
                linewidth=lw,
                markersize=5.2,
                capsize=3,
            )
        ax.set_xlabel("Per-User Offered Traffic Rate (Mbps)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(rates)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(labels), 7)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=ncol, frameon=True, fontsize=8.8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save_fig(fig, fig_dir, stem)


def plot_fig4_load_drift(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite3_load_drift_600s.json")
    if not os.path.exists(p):
        print("[Skip Fig 4] Missing suite3_load_drift_600s.json")
        return None
    with open(p) as f:
        data = json.load(f)

    _plot_load_group(
        data,
        GROUP1_ORDER,
        "Group 1 Load Sensitivity: Traditional, Graph & Optimization vs. TRA-HGDRL (600 s, 3 Seeds)",
        fig_dir,
        "fig4a_load_traditional",
    )
    _plot_load_group(
        data,
        GROUP2_ORDER,
        "Group 2 Load Sensitivity: Multi-Agent RL vs. TRA-HGDRL (600 s, 3 Seeds)",
        fig_dir,
        "fig4b_load_rl",
    )
    return data


def _plot_robust_group(d4: dict | None, d5: dict | None, group_order: list[str], suptitle: str, fig_dir: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    if d4 is not None:
        scales = d4["shadow_scales"]
        algos = [a for a in group_order if a in d4["by_shadow"][f"{scales[0]:.2f}"]]
        ax = axes[0]
        for a in algos:
            ys = [d4["by_shadow"][f"{s:.2f}"][a]["mean"]["rt_drop_rate"] * 100.0 for s in scales]
            lw = 2.5 if a == "TRA-HGDRL" else 1.65
            ax.plot(scales, ys, label=a, color=ALGO_COLORS[a], marker=ALGO_MARKERS[a], linewidth=lw, markersize=5)
        ax.set_xlabel("Shadow Fading Scale Multiplier")
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(a) Channel Shadowing Severity")
        ax.set_xticks(scales)

        taus = [int(t) for t in d4["tau_hos"]]
        ax = axes[1]
        for a in algos:
            ys = [d4["by_tau"][str(t)][a]["mean"]["rt_drop_rate"] * 100.0 for t in taus]
            lw = 2.5 if a == "TRA-HGDRL" else 1.65
            ax.plot(taus, ys, label=a, color=ALGO_COLORS[a], marker=ALGO_MARKERS[a], linewidth=lw, markersize=5)
        ax.set_xlabel(r"Handover Interruption $\tau_{\mathrm{HO}}$ (ms)")
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(b) Interruption Window Impact")
        ax.set_xticks(taus)

    if d5 is not None:
        presets = d5["presets"]
        algos = [a for a in group_order if a in d5["by_preset"][presets[0]]]
        ax = axes[2]
        x = np.arange(len(presets))
        w = min(0.14, 0.80 / max(len(algos), 1))
        for idx, a in enumerate(algos):
            ys = [d5["by_preset"][pr][a]["mean"]["rt_drop_rate"] * 100.0 for pr in presets]
            offset = (idx - (len(algos) - 1) / 2.0) * w
            ax.bar(x + offset, ys, width=w, label=a, color=ALGO_COLORS[a], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Tight\n(45°N)", "Standard\n(47°N)", "Dense\n(35°N)"])
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(c) Constellation Generalization")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=len(labels), frameon=True, fontsize=8.8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save_fig(fig, fig_dir, stem)


def plot_fig5_robustness_constellation(out_dir: str, fig_dir: str) -> tuple[dict | None, dict | None]:
    p4 = os.path.join(out_dir, "suite4_robustness_600s.json")
    p5 = os.path.join(out_dir, "suite5_constellation_600s.json")
    d4 = json.load(open(p4)) if os.path.exists(p4) else None
    d5 = json.load(open(p5)) if os.path.exists(p5) else None
    if d4 is None and d5 is None:
        print("[Skip Fig 5] Missing suite4 and suite5 JSONs.")
        return None, None

    _plot_robust_group(
        d4,
        d5,
        GROUP1_ORDER,
        "Group 1 Robustness & Constellation Generalization: Traditional/Graph/Opt vs. TRA-HGDRL (600 s, 3 Seeds)",
        fig_dir,
        "fig5a_robust_traditional",
    )
    _plot_robust_group(
        d4,
        d5,
        GROUP2_ORDER,
        "Group 2 Robustness & Constellation Generalization: Multi-Agent RL vs. TRA-HGDRL (600 s, 3 Seeds)",
        fig_dir,
        "fig5b_robust_rl",
    )
    return d4, d5


def plot_fig6_closed_loop(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite6_mappo_closed_loop.json")
    if not os.path.exists(p):
        print("[Skip Fig 6] Missing suite6_mappo_closed_loop.json")
        return None
    with open(p) as f:
        data = json.load(f)
    res = data["results"]
    algos = [a for a in ALL_11_ORDER if a in res]
    trad_cnt = sum(1 for a in ["Max-RVT", "ISA-HO", "Swap-HO", "RA-HO", "MCMF-HO", "RH-STG"] if a in res)
    x = np.arange(len(algos))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    metrics = [
        ("outage_rate", "Service Outage Rate (%)", 100.0, "{:.2f}%"),
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0, "{:.2f}%"),
        ("effective_tput_mbps", "Effective Throughput (Mbps)", 1.0, "{:.2f}"),
    ]
    for ax, (key, title, scale, fmt) in zip(axes, metrics):
        means = [res[a]["mean"][key] * scale for a in algos]
        cis = [res[a]["ci95"][key] * scale for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        bars = ax.bar(x, means, yerr=cis, capsize=2.8, color=colors, alpha=0.92, edgecolor="black", linewidth=0.55)
        for bar, a in zip(bars, algos):
            if a == "TRA-HGDRL":
                bar.set_linewidth(1.5)
        if 0 < trad_cnt < len(algos):
            ax.axvline(trad_cnt - 0.5, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        ymax = max(m + c for m, c in zip(means, cis))
        ymin = 0.0 if key != "effective_tput_mbps" else max(0.0, min(means) * 0.90)
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin, ymax + span * 0.24)
        for bar, m, c in zip(bars, means, cis):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                m + c + span * 0.025,
                fmt.format(m),
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=28 if len(algos) >= 9 else 0,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=32, ha="right")
        ax.set_title(title)
    fig.suptitle(
        f"Cross-Timescale Verification Coupled with Frozen Chapter 3 MAPPO-APG-Local (2 ms Slots, {len(data['seeds'])} Seeds, All {len(algos)} Methods)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig6_mappo_closed_loop")
    return data


def write_markdown_summary(out_dir: str, d1, d2, d3, d4, d5, d6) -> None:
    lines = ["# Chapter 4 TRA-HGDRL — 11-Method Paper Experiment Summary\n"]

    def _table(title: str, res_dict: dict, order: list[str]) -> None:
        lines.append(f"## {title}\n")
        lines.append("| Algorithm | HO | Ping-Pong (%) | Outage (%) | RT Drop (%) | RT Delay (ms) | Tput (Mbps) | Load Var | Jain | Dec (ms) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for a in order:
            if a not in res_dict:
                continue
            m = res_dict[a]["mean"]
            c = res_dict[a]["ci95"]
            lines.append(
                f"| {a} | {m['total_handovers']:.1f} ± {c['total_handovers']:.1f} | "
                f"{m['ping_pong_rate']*100:.2f} ± {c['ping_pong_rate']*100:.2f} | "
                f"{m['outage_rate']*100:.2f} ± {c['outage_rate']*100:.2f} | "
                f"{m['rt_drop_rate']*100:.2f} ± {c['rt_drop_rate']*100:.2f} | "
                f"{m['rt_delivered_delay_ms']:.2f} | "
                f"{m['effective_tput_mbps']:.2f} ± {c['effective_tput_mbps']:.2f} | "
                f"{m['load_variance']:.3f} | {m['jain_fairness']:.3f} | {m['decision_ms_per_step']:.2f} |"
            )
        lines.append("")

    if d1 is not None:
        _table("1a. Group 1 — Traditional, Graph-Theoretic & Optimization vs. TRA-HGDRL (600 s, tight_45n, 5 Seeds)", d1["results"], GROUP1_ORDER)
        _table("1b. Group 2 — Multi-Agent Reinforcement Learning vs. TRA-HGDRL (600 s, tight_45n, 5 Seeds)", d1["results"], GROUP2_ORDER)
        _table("1c. Unified 11-Method Master Benchmark (600 s, tight_45n, U=200, 28 Mbps, 5 Seeds)", d1["results"], ALL_11_ORDER)
    if d2 is not None:
        _table("2a. Ablation Study — Standard (600 s, 5 Seeds)", d2["standard"], ABL_ORDER)
        _table("2b. Ablation Study — Hotspot Drift + 1.5x Shadowing (600 s, 5 Seeds)", d2["stress"], ABL_ORDER)
    if d3 is not None:
        _table("3. Hotspot Drift Scenario — All 11 Methods (U=220, 32 Mbps, 600 s, 3 Seeds)", d3["hotspot_drift"], ALL_11_ORDER)
    if d4 is not None:
        _table("4a. Severe Shadowing Stress — 2.0x Shadowing (600 s, 3 Seeds)", d4["by_shadow"]["2.00"], ALL_11_ORDER)
        _table("4b. High Interruption Stress — tau_HO = 50 ms (600 s, 3 Seeds)", d4["by_tau"]["50"], ALL_11_ORDER)
    if d5 is not None:
        for pr in d5["presets"]:
            _table(f"5. Constellation Generalization — {pr} (600 s, 3 Seeds)", d5["by_preset"][pr], ALL_11_ORDER)
    if d6 is not None:
        _table(
            f"6. Cross-Timescale Frozen Chapter 3 MAPPO-APG-Local Closed-Loop ({d6['duration']}s, {len(d6['seeds'])} Seeds)",
            d6["results"],
            ALL_11_ORDER,
        )

    out_md = os.path.join(out_dir, "PAPER_SUMMARY.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary report -> {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Generate Chapter 4 paper figures and summary tables")
    parser.add_argument("--ckpt-dir", type=str, default="handover/checkpoints_paper")
    parser.add_argument("--results-dir", type=str, default="handover/results/paper")
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    plot_fig1_convergence(args.ckpt_dir, fig_dir)
    plot_fig1b_reward_curves(args.ckpt_dir, fig_dir)
    d1 = plot_fig2_main_comparison(args.results_dir, fig_dir)
    d2 = plot_fig3_ablation(args.results_dir, fig_dir)
    d3 = plot_fig4_load_drift(args.results_dir, fig_dir)
    d4, d5 = plot_fig5_robustness_constellation(args.results_dir, fig_dir)
    d6 = plot_fig6_closed_loop(args.results_dir, fig_dir)
    write_markdown_summary(args.results_dir, d1, d2, d3, d4, d5, d6)


if __name__ == "__main__":
    main()
