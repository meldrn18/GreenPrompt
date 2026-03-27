
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("classic")

COLORS = {
    "zero_shot": "C0",
    "few_shot": "C1",
    "chain_of_thought": "C2",
    "reflexion": "C3",
}

LABELS = {
    "chain_of_thought": "Chain-of-thought",
    "few_shot": "Few-shot",
    "reflexion": "Reflexion",
    "zero_shot": "Zero-shot",
}
STRAT_ORDER = ["zero_shot", "few_shot", "chain_of_thought", "reflexion"]

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


#load data
def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    df["carbon_execution"] = df["carbon"].apply(
        lambda d: d.get("execution") if isinstance(d, dict) else None
    )
    df["carbon_inference"] = df["carbon"].apply(
        lambda d: d.get("inference") if isinstance(d, dict) else None
    )
    df["carbon_execution"] = pd.to_numeric(df["carbon_execution"], errors="coerce")
    df["carbon_inference"] = pd.to_numeric(df["carbon_inference"], errors="coerce")
    df["exec_ng"] = df["carbon_execution"] * 1e9
    df["inf_ng"] = df["carbon_inference"] * 1e9

    def safe_cx(val):
        return val if isinstance(val, dict) else {}

    cx = pd.json_normalize(df["complexity"].apply(safe_cx).tolist())
    cx.columns = [f"cx_{c}" for c in cx.columns]
    df = pd.concat([df.drop(columns=["complexity"]), cx.set_index(df.index)], axis=1)

    for c in cx.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["memory_mb"] = pd.to_numeric(df.get("memory_mb"), errors="coerce")
    df["label"] = df["strategy"].map(LABELS)

    return df


def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    print(f"  Saved {path}")
    plt.close(fig)


def get_xtick_labels():
    return ["Zero-shot", "Few-shot", "Chain-of-\nthought", "Reflexion"]

#1. exec carbon box plots
def fig_boxplot(df):
    data = [df[df["strategy"] == s]["exec_ng"].dropna().values for s in STRAT_ORDER]
    xlabs = get_xtick_labels()

    #linear
    fig_linear, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.2),
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
        ),
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="gray",
            markeredgecolor="gray",
            alpha=0.5,
        ),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")

    ax.set_xticks(range(1, len(STRAT_ORDER) + 1))
    ax.set_xticklabels(xlabs)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel(r"Exec carbon ($\times 10^{-9}$ g CO$_2$e)")
    ax.set_title("Linear scale")

    save(fig_linear, "fig1a_linear.pdf")

    #log
    fig_log, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.2),
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
        ),
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="gray",
            markeredgecolor="gray",
            alpha=0.5,
        ),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(STRAT_ORDER) + 1))
    ax.set_xticklabels(xlabs)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel(r"Exec carbon ($\times 10^{-9}$ g CO$_2$e)")
    ax.set_title("Log scale")

    # legend only on (b)
    legend_els = [
        Line2D([0], [0], marker="D", color="black", linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markersize=5, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.2, label="Median"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=9)

    save(fig_log, "fig1b_log.pdf")

#2. inference vs exec carbon
def fig_inference_vs_execution(df):
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)

    means = df.groupby("strategy")[["inf_ng", "exec_ng"]].mean()
    x = np.arange(len(STRAT_ORDER))
    w = 0.35

    inf_vals = [means.loc[s, "inf_ng"] for s in STRAT_ORDER]
    exec_vals = [means.loc[s, "exec_ng"] for s in STRAT_ORDER]

    ax.bar(x - w / 2, inf_vals, w, label="Inference")
    ax.bar(x + w / 2, exec_vals, w, label="Execution")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(get_xtick_labels())
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel(r"Carbon ($\times 10^{-9}$ g CO$_2$e, log scale)")
    ax.set_title("Mean inference vs execution carbon by strategy")
    ax.legend(loc="upper right", fontsize=9)

    inf_mean = means["inf_ng"].mean()
    exc_mean = means["exec_ng"].mean()
    ratio = inf_mean / exc_mean
    ax.annotate(
        f"Inference ≈ {ratio:.0f}× execution (mean)",
        xy=(0.5, 0.93),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
    )

    save(fig, "fig2_inference_vs_execution.pdf")

#3. oracle
def fig_oracle(df):
    valid = df[["problem_id", "strategy", "exec_ng"]].dropna()
    counts = valid.groupby("problem_id")["strategy"].nunique()
    matched = valid[valid["problem_id"].isin(counts[counts == 4].index)]

    oracle_best = matched.groupby("problem_id")["exec_ng"].min().sum()
    oracle_worst = matched.groupby("problem_id")["exec_ng"].max().sum()
    fixed = {s: matched[matched["strategy"] == s]["exec_ng"].sum() for s in STRAT_ORDER}

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)

    all_labels = ["Oracle best"] + [LABELS[s] for s in STRAT_ORDER] + ["Oracle worst"]
    all_vals = [oracle_best] + [fixed[s] for s in STRAT_ORDER] + [oracle_worst]

    bars = ax.barh(all_labels[::-1], all_vals[::-1], color="lightgray")

    for bar, val in zip(bars, all_vals[::-1]):
        ax.text(
            bar.get_width() + max(all_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.1f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(r"Total exec carbon across 212 matched problems ($\times 10^{-9}$ g CO$_2$e)")
    ax.set_title(
        "Oracle best / worst vs fixed strategy totals\n"
        "(212 problems where all four strategies completed)"
    )

    gap_pct = (oracle_worst - oracle_best) / oracle_best * 100
    ax.axvline(oracle_best, linestyle="--", linewidth=1)
    ax.axvline(oracle_worst, linestyle="--", linewidth=1)
    ax.annotate(
        f"Gap: {gap_pct:.1f}%",
        xy=((oracle_best + oracle_worst) / 2, 0.08),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=9,
    )

    save(fig, "fig3_oracle.pdf")

#4. ranking heatmap
def fig_ranking_heatmap(df):
    valid = df[["problem_id", "strategy", "exec_ng"]].dropna()
    counts = valid.groupby("problem_id")["strategy"].nunique()
    matched = valid[valid["problem_id"].isin(counts[counts == 4].index)]

    def mean_no_outliers(grp):
        mu, sigma = grp.mean(), grp.std()
        return grp[grp <= mu + 2 * sigma].mean()

    lenses = {
        "Mean\n(all runs)": valid.groupby("strategy")["exec_ng"].mean(),
        "Median\n(all runs)": valid.groupby("strategy")["exec_ng"].median(),
        "Mean\n(outliers removed)": valid.groupby("strategy")["exec_ng"].apply(mean_no_outliers),
        "Mean\n(matched problems)": matched.groupby("strategy")["exec_ng"].mean(),
    }

    rank_df = pd.DataFrame({k: v.rank().astype(int) for k, v in lenses.items()})
    rank_df.index = [LABELS[s] for s in rank_df.index]
    rank_df = rank_df.loc[[LABELS[s] for s in STRAT_ORDER]]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    im = ax.imshow(rank_df.values, cmap="viridis", aspect="auto", vmin=1, vmax=4)

    ax.set_xticks(range(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns)
    ax.set_yticks(range(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=6)

    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            ax.text(
                j,
                i,
                str(rank_df.values[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color="white",
            )

    ax.set_title("Strategy ranking under four statistical lenses\n(1 = lowest emissions, 4 = highest)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Rank", fontsize=9)
    cb.set_ticks([1, 2, 3, 4])

    save(fig, "fig4_ranking_heatmap.pdf")

#5. win rates
def fig_win_rates(df):
    valid = df[["problem_id", "strategy", "exec_ng"]].dropna()
    counts = valid.groupby("problem_id")["strategy"].nunique()
    matched = valid[valid["problem_id"].isin(counts[counts == 4].index)]

    winners = matched.loc[matched.groupby("problem_id")["exec_ng"].idxmin(), "strategy"]
    win_counts = winners.value_counts()

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    strats = [s for s in STRAT_ORDER if s in win_counts.index]
    vals = [win_counts.get(s, 0) for s in strats]
    pcts = [v / len(winners) * 100 for v in vals]

    bars = ax.bar([LABELS[s] for s in strats], pcts)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("% of problems won (lowest exec carbon)")
    ax.set_title("Per-problem win rates across 212 matched problems")
    ax.set_ylim(0, max(pcts) * 1.2 if pcts else 1)
    ax.set_xticklabels(get_xtick_labels()[:len(strats)])
    ax.tick_params(axis="x", pad=6)

    save(fig, "fig5_win_rates.pdf")

#6. complexity vs exec scatter
def fig_complexity_scatter(df):
    cx_pairs = [
        ("cx_max_cyclomatic_complexity", "Max cyclomatic complexity", "fig6a_complexity_cyclomatic.pdf"),
        ("cx_halstead_volume", "Halstead volume", "fig6b_complexity_halstead.pdf"),
    ]

    for cx_col, cx_label, out_name in cx_pairs:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
        axes = axes.flatten()

        for ax, s in zip(axes, STRAT_ORDER):
            sub = df[df["strategy"] == s][[cx_col, "exec_ng"]].dropna()

            ax.scatter(
                sub[cx_col],
                sub["exec_ng"],
                s=14,
                alpha=0.5,
                color=COLORS[s],
            )

            if len(sub) >= 2:
                x = sub[cx_col].to_numpy()
                y = sub["exec_ng"].to_numpy()

                #linear trend line
                coeffs = np.polyfit(x, y, 1)
                xfit = np.linspace(x.min(), x.max(), 100)
                yfit = np.polyval(coeffs, xfit)
                ax.plot(xfit, yfit, color="black", linewidth=1)

                r = sub[[cx_col, "exec_ng"]].corr().iloc[0, 1]
                ax.text(
                    0.97, 0.93,
                    f"r = {r:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )

            ax.set_title(LABELS[s], fontsize=10)
            ax.set_xlabel(cx_label)
            ax.set_ylabel(r"Exec carbon ($\times 10^{-9}$ g CO$_2$e)")

        fig.suptitle(f"Execution carbon vs {cx_label.lower()} by strategy", fontsize=12)
        save(fig, out_name)

#7. carbon per sloc
def fig_carbon_per_sloc(df):
    if "cx_sloc" not in df.columns:
        print("  Skipping fig7_carbon_per_sloc.pdf (cx_sloc not found)")
        return

    eff = df[df["cx_sloc"] > 0][["strategy", "exec_ng", "cx_sloc"]].dropna().copy()
    if eff.empty:
        print("  Skipping fig7_carbon_per_sloc.pdf (no valid cx_sloc data)")
        return

    eff["cpp_sloc"] = eff["exec_ng"] / eff["cx_sloc"]

    agg = (
        eff.groupby("strategy")["cpp_sloc"]
        .agg(median="median", std="std")
        .reset_index()
        .sort_values("median")
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    labels = [LABELS[s] for s in agg["strategy"]]

    bars = ax.bar(
        labels,
        agg["median"],
        yerr=agg["std"].fillna(0),
        error_kw=dict(capsize=4),
    )

    offset = agg["std"].fillna(0).max() * 0.05 if len(agg) else 0
    for bar, val in zip(bars, agg["median"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel(r"Exec carbon per SLOC ($\times 10^{-9}$ g / line)")
    ax.set_title("Carbon efficiency: exec emissions per source line of code\n(lower = greener per line written)")
    ax.tick_params(axis="x", pad=6)
    ax.set_xlabel("Error bars: ±1 SD")

    save(fig, "fig7_carbon_per_sloc.pdf")

#8. percentiles
def fig_percentile_table(df):
    rows = []
    for s in STRAT_ORDER:
        sub = df[df["strategy"] == s]["exec_ng"].dropna()
        rows.append(
            {
                "Strategy": LABELS[s],
                "n": len(sub),
                "Median": f"{sub.quantile(.50):.2f}",
                "Mean": f"{sub.mean():.2f}",
                "p75": f"{sub.quantile(.75):.2f}",
                "p99": f"{sub.quantile(.99):.2f}",
                "Max": f"{sub.max():.2f}",
            }
        )
    tbl = pd.DataFrame(rows).set_index("Strategy")

    fig, ax = plt.subplots(figsize=(8.8, 2.8), constrained_layout=True)
    ax.axis("off")
    t = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        rowLabels=tbl.index,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.1, 1.8)

    for (row, col), cell in t.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontweight="bold")

    ax.set_title(r"Execution carbon percentiles ($\times 10^{-9}$ g CO$_2$e) by strategy", fontsize=11, pad=12)

    save(fig, "fig8_percentile_table.pdf")

#9. memory usage
def fig_memory_scatter(df):
    if "memory_mb" not in df.columns:
        print("  Skipping fig9_memory_scatter.pdf (memory_mb not found)")
        return
 
    sub_all = df[["strategy", "memory_mb", "exec_ng"]].dropna()
    if sub_all.empty:
        print("  Skipping fig9_memory_scatter.pdf (no valid memory_mb data)")
        return
 
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
    axes = axes.flatten()
 
    for ax, s in zip(axes, STRAT_ORDER):
        sub = sub_all[sub_all["strategy"] == s].copy()
 
        ax.scatter(
            sub["memory_mb"],
            sub["exec_ng"],
            s=14,
            alpha=0.5,
            color=COLORS[s],
        )
 
        if len(sub) >= 2:
            x = sub["memory_mb"].to_numpy()
            y = sub["exec_ng"].to_numpy()
 
            coeffs = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 100)
            yfit = np.polyval(coeffs, xfit)
            ax.plot(xfit, yfit, color="black", linewidth=1)
 
            r = sub[["memory_mb", "exec_ng"]].corr().iloc[0, 1]
            ax.text(
                0.97, 0.93,
                f"r = {r:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
 
        ax.set_title(LABELS[s], fontsize=10)
        ax.set_xlabel("Peak memory usage (MB)")
        ax.set_ylabel(r"Exec carbon ($\times 10^{-9}$ g CO$_2$e)")
 
    fig.suptitle("Execution carbon vs peak memory usage by strategy", fontsize=12)
    save(fig, "fig9_memory_scatter.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="new_final_results.jsonl")
    args = parser.parse_args()

    df = load(args.data)

    fig_boxplot(df)
    fig_inference_vs_execution(df)
    fig_oracle(df)
    fig_ranking_heatmap(df)
    fig_win_rates(df)
    fig_complexity_scatter(df)
    fig_carbon_per_sloc(df)
    fig_percentile_table(df)
    fig_memory_scatter(df)

    print(f"\nDone — all figures in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()