# -*- coding: utf-8 -*-
"""Generate publication-quality Chapter 4 figures (PNG + PDF) and Markdown/LaTeX summary tables."""

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
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ALGO_ORDER = ["Max-RVT", "ISA-HO", "Swap-HO", "Cell-RL", "MA-HGDQN", "TRA-HGDRL"]
ALGO_COLORS = {
    "Max-RVT": "#7F7F7F",
    "ISA-HO": "#1F77B4",
    "Swap-HO": "#FF7F0E",
    "Cell-RL": "#2CA02C",
    "MA-HGDQN": "#9467BD",
    "TRA-HGDRL": "#D62728",
}
ALGO_MARKERS = {
    "Max-RVT": "s",
    "ISA-HO": "^",
    "Swap-HO": "D",
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
    files = [("Cell-RL", "cell_rl_history.json"), ("MA-HGDQN", "ma_hgdqn_history.json"), ("TRA-HGDRL", "tra_hgdrl_history.json")]
    curves = {}
    for label, fname in files:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                curves[label] = json.load(f)["history"]
    if not curves:
        print("[Skip Fig 1] No *_history.json found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
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
                lw = 2.4 if label == "TRA-HGDRL" else 1.8
                ax.plot(xs, ys, label=label, color=ALGO_COLORS[label], marker=ALGO_MARKERS[label], markersize=5, linewidth=lw)
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(frameon=True, loc="best")
    fig.suptitle("Training Convergence of DRL Handover Policies (tight_45n)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "fig1_training_convergence")


def plot_fig2_main_comparison(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite1_main_600s.json")
    if not os.path.exists(p):
        print("[Skip Fig 2] Missing suite1_main_600s.json")
        return None
    with open(p) as f:
        data = json.load(f)
    res = data["results"]
    algos = [a for a in ALGO_ORDER if a in res]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    metrics = [
        ("outage_rate", "Service Outage Rate (%)", 100.0, "{:.2f}%"),
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0, "{:.2f}%"),
        ("effective_tput_mbps", "Effective Throughput (Mbps)", 1.0, "{:.2f}"),
        ("total_handovers", "Total Handovers (600 s)", 1.0, "{:.0f}"),
        ("ping_pong_rate", "Ping-Pong Rate (%)", 100.0, "{:.1f}%"),
        ("load_variance", "Satellite Load Variance", 1.0, "{:.3f}"),
    ]
    x = np.arange(len(algos))
    for ax, (key, title, scale, fmt) in zip(axes.flat, metrics):
        means = [res[a]["mean"][key] * scale for a in algos]
        cis = [res[a]["ci95"][key] * scale for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        bars = ax.bar(x, means, yerr=cis, capsize=3.5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        # Highlight TRA-HGDRL bar border
        for bar, a in zip(bars, algos):
            if a == "TRA-HGDRL":
                bar.set_linewidth(1.5)
        ymax = max(m + c for m, c in zip(means, cis))
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        for bar, m, c in zip(bars, means, cis):
            ax.text(bar.get_x() + bar.get_width() / 2.0, m + c + ymax * 0.02, fmt.format(m), ha="center", va="bottom", fontsize=7.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=22, ha="right")
        ax.set_title(title)
    fig.suptitle("Primary 600 s Handover Performance Comparison (tight_45n, 5 Seeds, 95% CI)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "fig2_main_comparison")
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


def plot_fig4_load_drift(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite3_load_drift_600s.json")
    if not os.path.exists(p):
        print("[Skip Fig 4] Missing suite3_load_drift_600s.json")
        return None
    with open(p) as f:
        data = json.load(f)
    rates = [int(r) for r in data["rates"]]
    by_rate = data["by_rate"]
    algos = [a for a in ALGO_ORDER if a in by_rate[str(rates[0])]]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
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
            lw = 2.4 if a == "TRA-HGDRL" else 1.7
            ax.errorbar(
                rates,
                ys,
                yerr=es,
                label=a,
                color=ALGO_COLORS[a],
                marker=ALGO_MARKERS[a],
                linewidth=lw,
                capsize=3,
            )
        ax.set_xlabel("Per-User Offered Traffic Rate (Mbps)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(rates)
        ax.legend(frameon=True, loc="best")
    fig.suptitle("Sensitivity to Offered Traffic Load (600 s, tight_45n, U=200, 3 Seeds)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "fig4_load_sensitivity")
    return data


def plot_fig5_robustness_constellation(out_dir: str, fig_dir: str) -> tuple[dict | None, dict | None]:
    p4 = os.path.join(out_dir, "suite4_robustness_600s.json")
    p5 = os.path.join(out_dir, "suite5_constellation_600s.json")
    d4 = json.load(open(p4)) if os.path.exists(p4) else None
    d5 = json.load(open(p5)) if os.path.exists(p5) else None
    if d4 is None and d5 is None:
        print("[Skip Fig 5] Missing suite4 and suite5 JSONs.")
        return None, None

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    if d4 is not None:
        scales = d4["shadow_scales"]
        algos = [a for a in ALGO_ORDER if a in d4["by_shadow"][f"{scales[0]:.2f}"]]
        ax = axes[0]
        for a in algos:
            ys = [d4["by_shadow"][f"{s:.2f}"][a]["mean"]["rt_drop_rate"] * 100.0 for s in scales]
            lw = 2.4 if a == "TRA-HGDRL" else 1.6
            ax.plot(scales, ys, label=a, color=ALGO_COLORS[a], marker=ALGO_MARKERS[a], linewidth=lw)
        ax.set_xlabel("Shadow Fading Scale Multiplier")
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(a) Channel Shadowing Severity")
        ax.set_xticks(scales)

        taus = [int(t) for t in d4["tau_hos"]]
        ax = axes[1]
        for a in algos:
            ys = [d4["by_tau"][str(t)][a]["mean"]["rt_drop_rate"] * 100.0 for t in taus]
            lw = 2.4 if a == "TRA-HGDRL" else 1.6
            ax.plot(taus, ys, label=a, color=ALGO_COLORS[a], marker=ALGO_MARKERS[a], linewidth=lw)
        ax.set_xlabel(r"Handover Interruption $\tau_{\mathrm{HO}}$ (ms)")
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(b) Interruption Window Impact")
        ax.set_xticks(taus)

    if d5 is not None:
        presets = d5["presets"]
        algos = [a for a in ALGO_ORDER if a in d5["by_preset"][presets[0]]]
        ax = axes[2]
        x = np.arange(len(presets))
        w = 0.13
        for idx, a in enumerate(algos):
            ys = [d5["by_preset"][pr][a]["mean"]["rt_drop_rate"] * 100.0 for pr in presets]
            offset = (idx - (len(algos) - 1) / 2.0) * w
            ax.bar(x + offset, ys, width=w, label=a, color=ALGO_COLORS[a], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Tight\n(45°N)", "Standard\n(47°N)", "Dense\n(35°N)"])
        ax.set_ylabel("RT Drop Rate (%)")
        ax.set_title("(c) Constellation Generalization")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=6, frameon=True, fontsize=9)
    fig.suptitle("Robustness & Constellation Density Generalization (600 s, 3 Seeds)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save_fig(fig, fig_dir, "fig5_robustness_constellation")
    return d4, d5


def plot_fig6_closed_loop(out_dir: str, fig_dir: str) -> dict | None:
    p = os.path.join(out_dir, "suite6_mappo_closed_loop.json")
    if not os.path.exists(p):
        print("[Skip Fig 6] Missing suite6_mappo_closed_loop.json")
        return None
    with open(p) as f:
        data = json.load(f)
    res = data["results"]
    algos = [a for a in ALGO_ORDER if a in res]
    x = np.arange(len(algos))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    metrics = [
        ("outage_rate", "Service Outage Rate (%)", 100.0, "{:.2f}%"),
        ("rt_drop_rate", "RT Packet Drop Rate (%)", 100.0, "{:.2f}%"),
        ("effective_tput_mbps", "Effective Throughput (Mbps)", 1.0, "{:.2f}"),
    ]
    for ax, (key, title, scale, fmt) in zip(axes, metrics):
        means = [res[a]["mean"][key] * scale for a in algos]
        cis = [res[a]["ci95"][key] * scale for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        bars = ax.bar(x, means, yerr=cis, capsize=3.5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ymax = max(m + c for m, c in zip(means, cis))
        ymin = 0.0 if key != "effective_tput_mbps" else min(means) * 0.90
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.20)
        for bar, m, c in zip(bars, means, cis):
            ax.text(bar.get_x() + bar.get_width() / 2.0, m + c + (ymax - ymin) * 0.02, fmt.format(m), ha="center", va="bottom", fontsize=8.0)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=22, ha="right")
        ax.set_title(title)
    fig.suptitle(
        f"Cross-Timescale Verification Coupled with Frozen Chapter 3 MAPPO-APG-Local (2 ms Slots, {len(data['seeds'])} Seeds)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig6_mappo_closed_loop")
    return data


def write_markdown_summary(out_dir: str, d1, d2, d3, d4, d5, d6) -> None:
    lines = ["# Chapter 4 TRA-HGDRL — Paper Experiment Summary\n"]

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
        _table("1. Primary 600 s Benchmark (tight_45n, U=200, 28 Mbps, 5 Seeds)", d1["results"], ALGO_ORDER)
    if d2 is not None:
        _table("2a. Ablation Study — Standard (600 s, 5 Seeds)", d2["standard"], ABL_ORDER)
        _table("2b. Ablation Study — Hotspot Drift + 1.5x Shadowing (600 s, 5 Seeds)", d2["stress"], ABL_ORDER)
    if d3 is not None:
        _table("3. Hotspot Drift Scenario (U=220, 32 Mbps, 600 s, 3 Seeds)", d3["hotspot_drift"], ALGO_ORDER)
    if d5 is not None:
        for pr in d5["presets"]:
            _table(f"5. Constellation Generalization — {pr} (600 s, 3 Seeds)", d5["by_preset"][pr], ALGO_ORDER)
    if d6 is not None:
        _table(
            f"6. Cross-Timescale Frozen Chapter 3 MAPPO-APG-Local Closed-Loop ({d6['duration']}s, {len(d6['seeds'])} Seeds)",
            d6["results"],
            ALGO_ORDER,
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
    d1 = plot_fig2_main_comparison(args.results_dir, fig_dir)
    d2 = plot_fig3_ablation(args.results_dir, fig_dir)
    d3 = plot_fig4_load_drift(args.results_dir, fig_dir)
    d4, d5 = plot_fig5_robustness_constellation(args.results_dir, fig_dir)
    d6 = plot_fig6_closed_loop(args.results_dir, fig_dir)
    write_markdown_summary(args.results_dir, d1, d2, d3, d4, d5, d6)


if __name__ == "__main__":
    main()
