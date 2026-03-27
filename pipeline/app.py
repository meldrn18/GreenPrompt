import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="GreenPrompt Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    
)


# colours and labels for throughout
STRATEGY_COLORS = {
    "chain_of_thought": "#3B82F6",
    "reflexion":        "#10B981",
    "few_shot":         "#F59E0B",
    "zero_shot":        "#8B5CF6",
}
STRATEGY_LABELS = {
    "chain_of_thought": "Chain of thought",
    "reflexion":        "Reflexion",
    "few_shot":         "Few-shot",
    "zero_shot":        "Zero-shot",
}

#global styling
#header
st.markdown("## 🌱 GreenPrompt: Execution Emissions")
st.markdown(
    "<span style='color:#e2e8f0;font-size:1.0rem'>"
    "How do prompting strategies shape the execution carbon of LLM-generated code? "
    "Findings from the MBPP benchmark across zero-shot, few-shot, chain-of-thought and reflexion."
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")
st.markdown("""    
## About GreenPrompt

This dashboard shows the results of the GreenPrompt project. This is a research project investigating the impact of different
 prompting strategies on the carbon emissions of large language model (LLM) generated code. 

The dashboard compares the following prompting strategies against the mbpp dataset of programming problems; **zero-shot, few-shot, chain of thought, and reflexion**\n

Each strategy is evaluated using runtime carbon emissions, measured with
CodeCarbon during program execution in a sandbox.
            
**Key idea**: 
While prompting strategies do not explicitly optimise for energy, they can influence
**algorithmic structure and runtime behaviour**, which in turn affects carbon emissions.
This dashboard enables interactive exploration of these effects.
            
### What this dashboard shows
- Energy efficiency of generated code (lower emissions = greener)
- Consistency across problems, via per-problem comparisons and win rates
- Trade-offs between strategies, including emissions, memory usage, and code complexity


<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.insight-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    color: #e2e8f0;
    font-size: 1.0rem;
    line-height: 2.0;
}
.insight-card strong { color: #93c5fd; }
.insight-card.amber { border-left-color: #F59E0B; }
.insight-card.amber strong { color: #fcd34d; }
.insight-card.green { border-left-color: #10B981; }
.insight-card.green strong { color: #6ee7b7; }
.insight-card.red { border-left-color: #ef4444; }
.insight-card.red strong { color: #fca5a5; }

.kpi-row { display: flex; gap: 12px; margin-bottom: 1rem; }
.kpi { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px;
       padding: 0.85rem 1rem; flex: 1; }
.kpi .label { font-size: 1.0rem; color: #64748b; text-transform: uppercase;
              letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi .value { font-family: 'DM Mono', monospace; font-size: 1.4rem;
              font-weight: 500; color: #f1f5f9; }
.kpi .sub { font-size: 1.0rem; color: #475569; margin-top: 2px; }

.section-label { font-size: 1.0rem; text-transform: uppercase; letter-spacing: 0.12em;
                 color: #475569; margin-bottom: 0.5rem; font-family: 'DM Mono', monospace; }

div[data-testid="stTab"] button { font-family: 'DM Sans', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

#helpers
SCALE = 1e9  # display in ×10⁻⁹ g

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # flatten carbon dict
    if "carbon" in df.columns and df["carbon"].apply(lambda x: isinstance(x, dict)).any():
        df["carbon_inference"] = df["carbon"].apply(
            lambda d: d.get("inference") if isinstance(d, dict) else np.nan)
        df["carbon_execution"] = df["carbon"].apply(
            lambda d: d.get("execution") if isinstance(d, dict) else np.nan)
    for c in ["carbon_inference", "carbon_execution", "memory_mb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # flatten complexity dict
    if "complexity" in df.columns and df["complexity"].apply(lambda x: isinstance(x, dict)).any():
        flat = pd.json_normalize(df["complexity"].tolist())
        flat.columns = [f"cx_{c}" for c in flat.columns]
        df = pd.concat([df.drop(columns=["complexity"]), flat.set_index(df.index)], axis=1)
        for c in flat.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # readable labels & scaled exec carbon
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    df["exec_ng"] = df["carbon_execution"] * SCALE  # ×10⁻⁹ g

    # outcome classification
    def classify(o):
        if o == "": return "success"
        if o == "timed out": return "timeout"
        if isinstance(o, str) and "Error" in o: return "error"
        return "other"
    if "output" in df.columns:
        df["outcome"] = df["output"].apply(classify)

    return df

#convert hex to rgba for plotly
def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plotly_theme(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=12, r=12, t=36, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        **({"height": height} if height else {}),
    )
    fig.update_xaxes(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False)
    fig.update_yaxes(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False)
    return fig


#load data
st.sidebar.markdown("### Data")
path = st.sidebar.text_input("Results file", value="data/new_final_results.jsonl")

try:
    df = load_data(path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

strategies = sorted(df["strategy"].dropna().unique())
st.sidebar.markdown("### Filters")
chosen = st.sidebar.multiselect(
    "Strategies", strategies, default=strategies,
    format_func=lambda s: STRATEGY_LABELS.get(s, s)
)
fdf = df[df["strategy"].isin(chosen)].copy() if chosen else df.copy()




n_runs   = len(fdf)
n_probs  = fdf["problem_id"].nunique() if "problem_id" in fdf.columns else "-"
valid    = fdf["exec_ng"].dropna()
overall_median = valid.median()
overall_mean   = valid.mean()
success_rate   = (fdf["outcome"] == "success").mean() * 100 if "outcome" in fdf.columns else None

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="label">Runs</div><div class="value">{n_runs:,}</div></div>
  <div class="kpi"><div class="label">Problems</div><div class="value">{n_probs}</div></div>
  <div class="kpi"><div class="label">Median exec carbon</div><div class="value">{overall_median:.2f}</div>
    <div class="sub">×10⁻⁹ g CO₂e</div></div>
  <div class="kpi"><div class="label">Mean exec carbon</div><div class="value">{overall_mean:.2f}</div>
    <div class="sub">×10⁻⁹ g CO₂e - outlier-sensitive</div></div>
  {"" if success_rate is None else f'<div class="kpi"><div class="label">Success rate</div><div class="value">{success_rate:.1f}%</div><div class="sub">all strategies near-identical</div></div>'}
</div>
""", unsafe_allow_html=True)

#key findings - insight cards
st.markdown('<div class="section-label">Key findings</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="insight-card">
      <strong>The median can be misleading.</strong> All four strategies cluster within
      0.3 ×10⁻⁹ g of each other at the median (~15.1). Prompt strategy has little
      no effect on <em>typical-case</em> execution carbon.
    </div>
    <div class="insight-card amber">
      <strong>The mean ranking is driven by outliers.</strong> Removing the top 3% of runs and
      the rankings are reshuffled entirely. Chain-of-thought moves from 3rd to last and
      reflexion jumps to 1st. A handful of runaway executions skew the median.
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="insight-card green">
      <strong>Chain-of-thought has the lowest typical emissions.</strong> By median,
      CoT is 2nd lowest (15.17 ×10⁻⁹ g). Its higher mean is entirely explained
      by a single 192 ×10⁻⁹ outlier, ~13× the median.
    </div>
    <div class="insight-card red">
      <strong>Zero-shot carries the highest tail risk.</strong> Its p99 reaches
      83.6 ×10⁻⁹ g and it is the most expensive strategy on matched problems.
      The worst single run (180 ×10⁻⁹) is zero-shot.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

#set tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Distribution", "📈 Ranking robustness",
    "🔎 Per-problem", "💻 Explore generated code","⚖️ Tradeoffs", "📋 Raw data"
])

#1. Distribution
with tab1:
    st.markdown("### Execution carbon distribution by strategy")
    st.caption(
        "Box shows IQR (p25–p75). Whiskers extend to p10/p90. "
        "Mean shown as ◆. Points beyond p90 plotted individually."
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Box plot with log scale toggle
        use_log = st.toggle("Log scale (reveals tail structure)", value=True)

        strat_order = (
            fdf.groupby("strategy")["exec_ng"]
            .median()
            .sort_values()
            .index.tolist()
        )

        fig_box = go.Figure()
        for s in strat_order:
            sub = fdf[fdf["strategy"] == s]["exec_ng"].dropna()
            color = STRATEGY_COLORS.get(s, "#888888")
            # clip whiskers at p10/p90 manually 
            p10, p90 = sub.quantile(0.10), sub.quantile(0.90)
            outliers = sub[(sub < p10) | (sub > p90)]
            core = sub[(sub >= p10) & (sub <= p90)]
            fig_box.add_trace(go.Box(
                y=sub,
                name=STRATEGY_LABELS.get(s, s),
                marker_color=color,
                line_color=color,
                line_width=1.5,
                fillcolor=hex_to_rgba(color, 0.2),
                boxpoints="outliers",
                pointpos=0,
                jitter=0.4,
                marker_size=5,
                marker_opacity=0.7,
                whiskerwidth=0.5,
                boxmean=True,
            ))

        fig_box.update_layout(
            yaxis_title="Execution carbon (×10⁻⁹ g CO₂e)",
            yaxis_type="log" if use_log else "linear",
            showlegend=False,
        )
        plotly_theme(fig_box, height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        st.markdown("**Percentile summary**")
        pct_rows = []
        for s in strat_order:
            sub = fdf[fdf["strategy"] == s]["exec_ng"].dropna()
            pct_rows.append({
                "Strategy": STRATEGY_LABELS.get(s, s),
                "p50": f"{sub.median():.2f}",
                "Mean": f"{sub.mean():.2f}",
                "p75": f"{sub.quantile(0.75):.2f}",
                "p99": f"{sub.quantile(0.99):.2f}",
                "Max": f"{sub.max():.2f}",
            })
        st.dataframe(pd.DataFrame(pct_rows).set_index("Strategy"), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Mean–median gap** (outlier pull)")
        for s in strat_order:
            sub = fdf[fdf["strategy"] == s]["exec_ng"].dropna()
            gap = sub.mean() - sub.median()
            pct = gap / sub.median() * 100
            color = STRATEGY_COLORS.get(s, "#888")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:0.82rem;color:#94a3b8;margin-bottom:4px'>"
                f"<span style='color:{color}'>{STRATEGY_LABELS.get(s,s)}</span>"
                f"<span>+{pct:.1f}%</span></div>",
                unsafe_allow_html=True,
            )

#inference vs execution
    st.markdown("---")
    st.markdown("### Inference vs execution carbon")
    st.caption(
        "Inference dominates total footprint by ~280×. "
        "Bars show mean per strategy; note the log scale."
    )

    inf_exec = []
    for s in chosen:
        sub = fdf[fdf["strategy"] == s]
        inf_exec.append({
            "Strategy": STRATEGY_LABELS.get(s, s),
            "Inference": sub["carbon_inference"].mean() * SCALE if "carbon_inference" in sub.columns else np.nan,
            "Execution": sub["exec_ng"].mean(),
            "color": STRATEGY_COLORS.get(s, "#888"),
        })
    ie_df = pd.DataFrame(inf_exec)

    fig_ie = go.Figure()
    fig_ie.add_trace(go.Bar(
        name="Inference",
        x=ie_df["Strategy"], y=ie_df["Inference"],
        marker_color=[hex_to_rgba(c, 0.73) for c in ie_df["color"]],
    ))
    fig_ie.add_trace(go.Bar(
        name="Execution",
        x=ie_df["Strategy"], y=ie_df["Execution"],
        marker_color=ie_df["color"].tolist(),
    ))
    fig_ie.update_layout(
        barmode="group",
        yaxis_type="log",
        yaxis_title="Carbon (×10⁻⁹ g CO₂e, log scale)",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    plotly_theme(fig_ie, height=340)
    st.plotly_chart(fig_ie, use_container_width=True)

    st.markdown(
        "<div class='insight-card'>"
        "<strong>Inference carbon is ~280× larger than execution carbon</strong> across all strategies. "
        "This means any optimisation focused solely on execution emissions is working on the smaller term. "
        "If total LLM footprint matters, inference cost (driven by prompt length) is the more impactful lever."
        "</div>",
        unsafe_allow_html=True,
    )


#2. Ranking
with tab2:
    st.markdown("### Does the ranking hold up?")
    st.caption(
        "The mean-based ranking shifts dramatically depending on how you slice the data. "
        "This tab shows all four lenses side-by-side."
    )

    valid_df = fdf[["strategy", "exec_ng", "problem_id"]].dropna(subset=["exec_ng"])

    lenses = {}

    # mean (all runs)
    lenses["Mean\n(all runs)"] = (
        valid_df.groupby("strategy")["exec_ng"].mean().sort_values()
    )

    # median (all runs)
    lenses["Median\n(all runs)"] = (
        valid_df.groupby("strategy")["exec_ng"].median().sort_values()
    )

    # 3. mean outliers removed 
    def mean_no_outliers(grp):
        mu, sigma = grp.mean(), grp.std()
        return grp[grp <= mu + 2 * sigma].mean()

    lenses["Mean\n(outliers removed)"] = (
        valid_df.groupby("strategy")["exec_ng"]
        .apply(mean_no_outliers)
        .sort_values()
    )

    # 4. mean matched problems only
    if "problem_id" in valid_df.columns:
        counts = valid_df.groupby("problem_id")["strategy"].nunique()
        matched_ids = counts[counts == len(chosen)].index
        matched = valid_df[valid_df["problem_id"].isin(matched_ids)]
        lenses["Mean\n(matched problems)"] = (
            matched.groupby("strategy")["exec_ng"].mean().sort_values()
        )

    # Ranking heatmap
    rank_data = {}
    for lens_name, series in lenses.items():
        ranked = series.rank().astype(int)
        rank_data[lens_name] = ranked

    rank_df = pd.DataFrame(rank_data)
    rank_df.index = rank_df.index.map(lambda s: STRATEGY_LABELS.get(s, s))

    fig_heat = go.Figure(data=go.Heatmap(
        z=rank_df.values,
        x=rank_df.columns.tolist(),
        y=rank_df.index.tolist(),
        colorscale=[[0, "#10B981"], [0.33, "#3B82F6"], [0.66, "#F59E0B"], [1, "#ef4444"]],
        text=rank_df.values,
        texttemplate="%{text}",
        textfont=dict(size=16, family="DM Mono"),
        showscale=False,
        zmin=1, zmax=4,
    ))
    fig_heat.update_layout(
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    plotly_theme(fig_heat, height=260)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("1 = lowest (greenest), 4 = highest emissions. Green = best, red = worst.")

    st.markdown("---")
    st.markdown("### Why is the mean higher than the median?")

    col1, col2 = st.columns(2)
    with col1:
        # mean vs median bar 
        mean_med = []
        for s in chosen:
            sub = valid_df[valid_df["strategy"] == s]["exec_ng"]
            mean_med.append({
                "Strategy": STRATEGY_LABELS.get(s, s),
                "Median": sub.median(),
                "Mean": sub.mean(),
                "color": STRATEGY_COLORS.get(s, "#888"),
            })
        mm_df = pd.DataFrame(mean_med).sort_values("Median")

        mm_long = pd.melt(
            mm_df, id_vars=["Strategy", "color"],
            value_vars=["Mean", "Median"],
            var_name="Statistic", value_name="value",
        )
        fig_mm = go.Figure()
        for stat, opacity, pattern in [("Mean", 1.0, ""), ("Median", 0.45, "/")]:
            sub = mm_long[mm_long["Statistic"] == stat]
            fig_mm.add_trace(go.Bar(
                name=stat,
                x=sub["Strategy"],
                y=sub["value"],
                marker_color=sub["color"].tolist(),
                marker_opacity=opacity,
                marker_pattern_shape=pattern,
                text=[f"{v:.2f}" for v in sub["value"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
        fig_mm.update_layout(
            barmode="group",
            yaxis_title="×10⁻⁹ g CO₂e",
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(range=[0, mm_long["value"].max() * 1.2]),
        )
        plotly_theme(fig_mm, height=320)
        st.plotly_chart(fig_mm, use_container_width=True)
        st.caption("Solid bars = mean. Hatched bars = median. "
            "Where mean >> median, outliers are inflating the average.")
    with col2:
        # outlier count 
        outlier_rows = []
        for s in chosen:
            sub = valid_df[valid_df["strategy"] == s]["exec_ng"]
            mu, sigma = sub.mean(), sub.std()
            outliers = sub[sub > mu + 2 * sigma]
            outlier_rows.append({
                "Strategy": STRATEGY_LABELS.get(s, s),
                "Outlier runs": len(outliers),
                "Max value": outliers.max() if len(outliers) else 0,
                "color": STRATEGY_COLORS.get(s, "#888"),
            })
        out_df = pd.DataFrame(outlier_rows).sort_values("Outlier runs", ascending=False)

        fig_out = go.Figure(go.Bar(
            x=out_df["Strategy"],
            y=out_df["Outlier runs"],
            marker_color=out_df["color"].tolist(),
            text=out_df.apply(
                lambda r: f"{int(r['Outlier runs'])} runs<br>max {r['Max value']:.1f}", axis=1
            ),
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig_out.update_layout(
            yaxis_title="Runs > mean + 2σ",
            yaxis=dict(range=[0, out_df["Outlier runs"].max() * 1.5]),
        )
        plotly_theme(fig_out, height=320)
        st.plotly_chart(fig_out, use_container_width=True)
        st.caption("Number of outlier runs per strategy and their worst-case values.")


#3. per-problem
with tab3:
    st.markdown("### Per-problem Analysis")

    if "problem_id" not in fdf.columns:
        st.info("Need a problem_id column for per-problem analysis.")
    else:
        valid_pp = fdf[["problem_id", "strategy", "exec_ng"]].dropna(subset=["exec_ng"])
        pivot = valid_pp.pivot_table(
            index="problem_id", columns="strategy", values="exec_ng", aggfunc="first"
        )
        matched_mask = pivot.notna().all(axis=1)
        pivot_matched = pivot[matched_mask]
        n_matched = len(pivot_matched)

        st.caption(
            f"{n_matched} problems where all {len(chosen)} selected strategies produced "
            "a correct solution were used for win-rate and saving analysis below."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Win rate**: fraction of problems where each strategy is cheapest")
            win_counts = pivot_matched.idxmin(axis=1).value_counts()
            win_df = win_counts.reset_index()
            win_df.columns = ["strategy", "wins"]
            win_df["label"] = win_df["strategy"].map(STRATEGY_LABELS)
            win_df["rate"] = win_df["wins"] / n_matched
            win_df["color"] = win_df["strategy"].map(STRATEGY_COLORS)
            win_df = win_df.sort_values("rate", ascending=True)

            fig_win = go.Figure(go.Bar(
                x=win_df["rate"], y=win_df["label"],
                orientation="h",
                marker_color=win_df["color"].tolist(),
                text=[f"{r*100:.1f}%" for r in win_df["rate"]],
                textposition="outside",
            ))
            fig_win.update_layout(
                xaxis=dict(range=[0, 0.45], tickformat=".0%"),
                yaxis_title="",
                xaxis_title="Win rate",
            )
            plotly_theme(fig_win, height=280)
            st.plotly_chart(fig_win, use_container_width=True)
            st.caption("No strategy dominates, each wins ~25% of problems. Problem complexity drives emissions more than strategy.")

        with col2:
            st.markdown("**Total execution carbon** if you committed to one strategy across all matched problems")
            totals = {s: pivot_matched[s].sum() for s in pivot_matched.columns if s in chosen}
            totals["Oracle best"] = pivot_matched.min(axis=1).sum()
            totals["Oracle worst"] = pivot_matched.max(axis=1).sum()

            t_df = (
                pd.Series(totals)
                .reset_index()
                .rename(columns={"index": "strategy", 0: "total"})
                .sort_values("total")
            )
            t_df["label"] = t_df["strategy"].apply(
                lambda s: STRATEGY_LABELS.get(s, s)
            )
            t_df["color"] = t_df["strategy"].apply(
                lambda s: STRATEGY_COLORS.get(s, ("#10B981" if s == "Oracle best" else "#ef4444"))
            )

            fig_tot = go.Figure(go.Bar(
                x=t_df["total"], y=t_df["label"],
                orientation="h",
                marker_color=t_df["color"].tolist(),
                text=[f"{v:.0f}" for v in t_df["total"]],
                textposition="outside",
            ))
            fig_tot.update_layout(
                xaxis_title="Total exec carbon (×10⁻⁹ g CO₂e)",
                yaxis_title="",
            )
            plotly_theme(fig_tot, height=280)
            st.plotly_chart(fig_tot, use_container_width=True)
            st.caption("Gap between oracle best and oracle worst is ~79%, but no single strategy captures the oracle best.")

        #saving distribution 
        st.markdown("---")
        st.markdown("### Saving potential: best vs worst strategy per problem")
        savings_pct = (
            (pivot_matched.max(axis=1) - pivot_matched.min(axis=1))
            / pivot_matched.max(axis=1) * 100
        ).dropna()

        fig_sav = go.Figure()
        fig_sav.add_trace(go.Histogram(
            x=savings_pct,
            nbinsx=30,
            marker_color="#3B82F6",
            opacity=0.8,
            name="Problems",
        ))
        fig_sav.add_vline(
            x=savings_pct.median(), line_dash="dash", line_color="#F59E0B",
            annotation_text=f"Median: {savings_pct.median():.1f}%",
            annotation_position="top right",
        )
        fig_sav.update_layout(
            xaxis_title="% saving from choosing best vs worst strategy",
            yaxis_title="Number of problems",
        )
        plotly_theme(fig_sav, height=300)
        st.plotly_chart(fig_sav, use_container_width=True)
        st.caption(
            f"Median saving: {savings_pct.median():.1f}% · "
            f"Mean saving: {savings_pct.mean():.1f}% · "
            f"Over 25% of problems have >40% potential saving."
        )


#4. code explorer
with tab4:
    st.markdown("### Explore generated code by problem")

    req = {"problem_id", "strategy", "generated_code"}
    if not req.issubset(set(fdf.columns)):
        st.info(f"Missing columns: {req - set(fdf.columns)}")
    else:
        problem_ids = sorted(fdf["problem_id"].astype(str).unique())
        sel_pid = st.selectbox("Problem ID", problem_ids)

        ex = fdf[fdf["problem_id"].astype(str) == sel_pid].copy()

        if "prompt" in ex.columns:
            prompt_val = ex["prompt"].dropna().astype(str)
            prompt_val = prompt_val[prompt_val.str.strip() != ""]
            if len(prompt_val):
                st.markdown("**Problem**")
                st.code(prompt_val.iloc[0], language="text")

        strat_opts = sorted(ex["strategy"].dropna().unique())
        sel_strats = st.multiselect(
            "Strategies to display", strat_opts, default=strat_opts,
            format_func=lambda s: STRATEGY_LABELS.get(s, s),
        )

        if sel_strats:
            cols = st.columns(len(sel_strats))
            for col, s in zip(cols, sel_strats):
                row = ex[ex["strategy"] == s].head(1)
                with col:
                    color = STRATEGY_COLORS.get(s, "#888")
                    st.markdown(
                        f"<div style='border-top:3px solid {color};"
                        f"padding-top:8px;margin-bottom:6px;font-weight:500;"
                        f"color:{color};font-size:1.0rem'>"
                        f"{STRATEGY_LABELS.get(s, s)}</div>",
                        unsafe_allow_html=True,
                    )
                    if row.empty:
                        st.info("No data")
                        continue

                    meta = []
                    if "exec_ng" in row.columns and pd.notna(row["exec_ng"].iloc[0]):
                        meta.append(f"exec: {row['exec_ng'].iloc[0]:.3g} ×10⁻⁹ g")
                    if "memory_mb" in row.columns and pd.notna(row["memory_mb"].iloc[0]):
                        meta.append(f"mem: {row['memory_mb'].iloc[0]:.3g} MB")
                    if "outcome" in row.columns:
                        meta.append(f"outcome: {row['outcome'].iloc[0]}")
                    if meta:
                        st.caption(" · ".join(meta))

                    code_text = str(row["generated_code"].iloc[0]) if "generated_code" in row.columns else ""
                    st.code(code_text, language="python")

                    if "output" in row.columns and pd.notna(row["output"].iloc[0]) and str(row["output"].iloc[0]).strip():
                        with st.expander("Output / logs"):
                            st.code(str(row["output"].iloc[0]))

#5. trade-offs
with tab5:
    st.markdown("### Emissions vs code complexity & memory")
    st.caption(
        "Do strategies that produce more complex or memory-hungry code also emit more? "
        "Each point is one run; hover for details."
    )
 
    cx_cols = {
        "cx_max_cyclomatic_complexity": "Cyclomatic complexity",
        "cx_maintainability_index":     "Maintainability index",
        "cx_sloc":                      "Source lines of code (SLOC)",
        "cx_halstead_volume":           "Halstead volume",
    }
    available_cx = {k: v for k, v in cx_cols.items() if k in fdf.columns}
 
    # scatter: exec carbon vs chosen complexity metric 
    st.markdown("#### Execution carbon vs complexity metric")
    col_pick, _ = st.columns([1, 2])
    with col_pick:
        cx_choice = st.selectbox(
            "Complexity metric",
            options=list(available_cx.keys()),
            format_func=lambda k: available_cx[k],
            key="cx_scatter_choice",
        )
 
    scatter_df = fdf[["strategy", "exec_ng", cx_choice, "problem_id"]].dropna()
    scatter_df["label"] = scatter_df["strategy"].map(STRATEGY_LABELS)
 
    fig_sc = go.Figure()
    for s in chosen:
        sub = scatter_df[scatter_df["strategy"] == s]
        fig_sc.add_trace(go.Scatter(
            x=sub[cx_choice],
            y=sub["exec_ng"],
            mode="markers",
            name=STRATEGY_LABELS.get(s, s),
            marker=dict(
                color=hex_to_rgba(STRATEGY_COLORS.get(s, "#888"), 0.55),
                size=6,
                line=dict(color=STRATEGY_COLORS.get(s, "#888"), width=0.5),
            ),
            customdata=sub[["problem_id", "label"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                f"{available_cx[cx_choice]}: %{{x:.2f}}<br>"
                "Exec carbon: %{y:.2f} ×10⁻⁹ g<br>"
                "Problem: %{customdata[0]}<extra></extra>"
            ),
        ))
 
    # add per-strategy trend lines
    for s in chosen:
        sub = scatter_df[scatter_df["strategy"] == s].sort_values(cx_choice)
        if len(sub) < 5:
            continue
        z = np.polyfit(sub[cx_choice], sub["exec_ng"], 1)
        x_line = np.linspace(sub[cx_choice].min(), sub[cx_choice].max(), 60)
        y_line = np.polyval(z, x_line)
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            showlegend=False,
            line=dict(color=STRATEGY_COLORS.get(s, "#888"), width=1.5, dash="dot"),
        ))
 
    fig_sc.update_layout(
        xaxis_title=available_cx[cx_choice],
        yaxis_title="Exec carbon (×10⁻⁹ g CO₂e)",
        legend=dict(orientation="h", y=1.1, x=0),
    )
    plotly_theme(fig_sc, height=400)
    st.plotly_chart(fig_sc, use_container_width=True)
 
    # Pearson r per strategy
    r_rows = []
    for s in chosen:
        sub = scatter_df[scatter_df["strategy"] == s]
        if len(sub) < 5:
            continue
        r = sub[["exec_ng", cx_choice]].corr().iloc[0, 1]
        r_rows.append({"Strategy": STRATEGY_LABELS.get(s, s), "Pearson r": round(r, 3)})
    if r_rows:
        r_df = pd.DataFrame(r_rows).set_index("Strategy")
        st.caption(f"Pearson r between exec carbon and {available_cx[cx_choice]}:")
        st.dataframe(r_df.T, use_container_width=False)
 
    st.markdown("---")
 
    # scatter: exec carbon vs memory 
    st.markdown("#### Execution carbon vs memory usage")
    if "memory_mb" in fdf.columns and fdf["memory_mb"].notna().any():
        mem_sc = fdf[["strategy", "exec_ng", "memory_mb", "problem_id"]].dropna()
 
        fig_mem = go.Figure()
        for s in chosen:
            sub = mem_sc[mem_sc["strategy"] == s]
            fig_mem.add_trace(go.Scatter(
                x=sub["memory_mb"],
                y=sub["exec_ng"],
                mode="markers",
                name=STRATEGY_LABELS.get(s, s),
                marker=dict(
                    color=hex_to_rgba(STRATEGY_COLORS.get(s, "#888"), 0.55),
                    size=6,
                    line=dict(color=STRATEGY_COLORS.get(s, "#888"), width=0.5),
                ),
                customdata=sub[["problem_id"]].values,
                hovertemplate=(
                    f"<b>{STRATEGY_LABELS.get(s, s)}</b><br>"
                    "Memory: %{x:.4f} MB<br>"
                    "Exec carbon: %{y:.2f} ×10⁻⁹ g<br>"
                    "Problem: %{customdata[0]}<extra></extra>"
                ),
            ))
        fig_mem.update_layout(
            xaxis_title="Memory (MB)",
            yaxis_title="Exec carbon (×10⁻⁹ g CO₂e)",
            legend=dict(orientation="h", y=1.1, x=0),
        )
        plotly_theme(fig_mem, height=360)
        st.plotly_chart(fig_mem, use_container_width=True)
    else:
        st.info("No memory_mb data available.")
 
    st.markdown("---")
 
#strategy summary: multi-metric comparison
    st.markdown("#### Multi-metric strategy profile")
    st.caption(
        "Each axis is normalised 0–1 (higher = better for that metric). "
        "Maintainability: higher is better. All others: lower is better (inverted)."
    )
 
    radar_metrics = {
        "exec_ng":                      ("Exec carbon", False),   # lower better, invert
        "memory_mb":                    ("Memory",       False),
        "cx_max_cyclomatic_complexity": ("Cyclomatic CC",False),
        "cx_sloc":                      ("SLOC",         False),
        "cx_maintainability_index":     ("Maintainability", True),  # higher better
    }
    radar_metrics = {k: v for k, v in radar_metrics.items() if k in fdf.columns}
 
    # compute per-strategy medians
    radar_rows = {}
    for s in chosen:
        sub = fdf[fdf["strategy"] == s]
        radar_rows[s] = {k: sub[k].median() for k in radar_metrics}
 
    radar_df = pd.DataFrame(radar_rows).T 
 
    # normalise: for each metric scale to [0,1]; invert if lower=better
    radar_norm = radar_df.copy()
    for col, (_, higher_better) in radar_metrics.items():
        mn, mx = radar_df[col].min(), radar_df[col].max()
        if mx == mn:
            radar_norm[col] = 1.0
        else:
            radar_norm[col] = (radar_df[col] - mn) / (mx - mn)
            if not higher_better:
                radar_norm[col] = 1 - radar_norm[col]
 
    axis_labels = [v[0] for v in radar_metrics.values()]
    axis_labels_closed = axis_labels + [axis_labels[0]]
 
    fig_rad = go.Figure()
    for s in chosen:
        vals = radar_norm.loc[s].tolist()
        vals_closed = vals + [vals[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=axis_labels_closed,
            fill="toself",
            name=STRATEGY_LABELS.get(s, s),
            line_color=STRATEGY_COLORS.get(s, "#888"),
            fillcolor=hex_to_rgba(STRATEGY_COLORS.get(s, "#888888"), 0.15),
        ))
 
    fig_rad.update_layout(
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color="#475569", size=9),
                gridcolor="#1e293b",
                linecolor="#1e293b",
            ),
            angularaxis=dict(
                tickfont=dict(color="#94a3b8", size=11),
                gridcolor="#1e293b",
                linecolor="#1e293b",
            ),
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        margin=dict(l=60, r=60, t=40, b=60),
        height=440,
    )
    st.plotly_chart(fig_rad, use_container_width=True)
    st.caption(
        "All axes scaled so that further from centre = better. "
        "A strategy that dominates all axes would fill the polygon completely."
    )
 
    st.markdown("---")
 
#exec carbon per SLOC 
    st.markdown("#### Carbon efficiency: exec emissions per line of code")
    st.caption(
        "Emission per SLOC penalises strategies that write verbose code without a runtime benefit. "
        "Lower = more efficient."
    )
 
    if "cx_sloc" in fdf.columns:
        eff_df = fdf[fdf["cx_sloc"] > 0][["strategy", "exec_ng", "cx_sloc"]].dropna().copy()
        eff_df["carbon_per_sloc"] = eff_df["exec_ng"] / eff_df["cx_sloc"]
 
        eff_agg = (
            eff_df.groupby("strategy")["carbon_per_sloc"]
            .agg(median="median", mean="mean", std="std")
            .reset_index()
            .sort_values("median")
        )
        eff_agg["label"] = eff_agg["strategy"].map(STRATEGY_LABELS)
        eff_agg["color"] = eff_agg["strategy"].map(STRATEGY_COLORS)
 
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(
            x=eff_agg["label"],
            y=eff_agg["median"],
            error_y=dict(type="data", array=eff_agg["std"].fillna(0), visible=True,
                         color="#475569", thickness=1.2),
            marker_color=eff_agg["color"].tolist(),
            text=[f"{v:.2f}" for v in eff_agg["median"]],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig_eff.update_layout(
            yaxis_title="Exec carbon per SLOC (×10⁻⁹ g / line)",
            xaxis_title="",
        )
        plotly_theme(fig_eff, height=320)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption("Error bars show ±1 std. Lower = greener per line written.")
    else:
        st.info("cx_sloc column not available.")
 
#6. raw data
with tab6:
    display_cols = [c for c in [
        "problem_id", "strategy", "outcome", "exec_ng",
        "carbon_inference", "memory_mb",
        "cx_maintainability_index", "cx_max_cyclomatic_complexity", "cx_sloc",
    ] if c in fdf.columns]
    show_df = fdf[display_cols].copy()
    if "exec_ng" in show_df.columns:
        show_df = show_df.rename(columns={"exec_ng": "exec_carbon_ng"})

    st.dataframe(show_df, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        fdf.to_csv(index=False).encode("utf-8"),
        file_name="greenprompt_filtered.csv",
        mime="text/csv",
    )