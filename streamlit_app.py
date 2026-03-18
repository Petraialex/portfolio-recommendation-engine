"""
EDI 3600 — Personality-Based Portfolio Recommendation Engine (Streamlit)
=========================================================================
Run with:   streamlit run streamlit_app.py

Files required in the SAME folder:
  - portfolio_algorithm.py
  - Macroeconomic_Dataset_DBA.xlsx

Requirements:
  pip install streamlit pandas numpy yfinance openpyxl scipy matplotlib plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_algorithm import (
    fetch_data,
    classify_regime,
    classify_risk_profile,
    get_portfolio,
    regime_conditioned_forecast,
    extract_forecast_stats,
    portfolio_return_series,
    QUESTIONNAIRE,
    ASSET_TICKERS,
    BASE_WEIGHTS,
    REGIME_TILTS,
    RISK_PROFILES,
    NUM_QUESTIONS,
    MAX_SCORE,
)

# =============================================================================
# PAGE CONFIG & THEME
# =============================================================================

st.set_page_config(
    page_title="Portfolio Advisor — EDI 3600",
    page_icon="📊",
    layout="wide",
)

PALETTE = {
    "primary": "#2E75B6", "secondary": "#1ABC9C",
    "warning": "#F39C12", "danger": "#C0392B",
}

PROFILE_STYLE = {
    "Conservative":             {"color": "#27AE60", "emoji": "🟢", "desc": "Capital preservation focus"},
    "Moderately Conservative":  {"color": "#2E86C1", "emoji": "🔵", "desc": "Income with moderate growth"},
    "Moderately Aggressive":    {"color": "#F39C12", "emoji": "🟠", "desc": "Growth-oriented, accepts risk"},
    "Aggressive":               {"color": "#C0392B", "emoji": "🔴", "desc": "Maximum growth, high risk tolerance"},
}

REGIME_COLORS = {
    "Expansion": "#27ae60", "Recovery": "#2e86c1",
    "Slowdown": "#f39c12", "Contraction": "#c0392b",
}

# =============================================================================
# CACHE DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading macro data from Excel & fetching asset prices...")
def load_data():
    dataset, returns_monthly = fetch_data()
    dataset["Regime"] = classify_regime(dataset)
    return dataset, returns_monthly

# =============================================================================
# SIDEBAR — MiFID RISK-PROFILE QUESTIONNAIRE
# =============================================================================

st.sidebar.title("📋 MiFID Risk Questionnaire")
st.sidebar.caption("Based on the MiFID II suitability framework — 6 dimensions, 14 questions.")
st.sidebar.markdown("---")

answers = []
answers_by_index = {}
current_dim = None

for i, q in enumerate(QUESTIONNAIRE):
    if q["dimension"] != current_dim:
        current_dim = q["dimension"]
        st.sidebar.markdown(f"### {q['dimension']}. {q['dimension_name']}")

    st.sidebar.markdown(f"**{q['question']}**")
    option_labels = [text for key, (text, _) in q["options"].items()]
    option_scores = [score for key, (text, score) in q["options"].items()]
    choice = st.sidebar.radio(
        label=f"q{i}",
        options=range(len(option_labels)),
        format_func=lambda x, labels=option_labels: labels[x],
        key=f"q_{i}",
        label_visibility="collapsed",
    )
    answers.append(option_scores[choice])
    answers_by_index[i] = option_scores[choice]

total_score = sum(answers)
risk_profile = classify_risk_profile(total_score)
pstyle = PROFILE_STYLE[risk_profile]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {pstyle['emoji']}  {risk_profile}")
st.sidebar.markdown(f"Score: **{total_score}** / {MAX_SCORE}  ·  {pstyle['desc']}")
st.sidebar.markdown("---")
initial_investment = st.sidebar.number_input(
    "💰 Initial Investment ($)", min_value=100, max_value=10_000_000,
    value=10_000, step=1_000,
)

# =============================================================================
# LOAD DATA
# =============================================================================

dataset, returns_monthly = load_data()
current_regime = dataset["Regime"].iloc[-1]
portfolio = get_portfolio(risk_profile, current_regime)

# =============================================================================
# HEADER — KEY METRICS
# =============================================================================

st.title("📊 Personality-Based Portfolio Recommendation Engine")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Risk Profile", risk_profile)
m2.metric("Macro Regime", current_regime)
m3.metric("Investment", f"${initial_investment:,.0f}")
equity_pct = sum(portfolio[t] for t in ["SPY", "IWM", "EFA", "EEM"]) * 100
m4.metric("Equity Allocation", f"{equity_pct:.0f}%")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🥧 Portfolio Allocation",
    "📈 5-Year Forecast",
    "🔄 Profile Comparison",
    "🧠 Risk Profile",
    "🗓️ Regime Analysis",
    "📊 Data & Stats",
    "ℹ️ About",
])

# =============================================================================
# TAB 1 — PORTFOLIO ALLOCATION
# =============================================================================

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Recommended Allocation")
        active = {k: v for k, v in portfolio.items() if v > 0}
        labels = [f"{ASSET_TICKERS[t]} ({t})" for t in active]
        values = list(active.values())

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.42,
            textinfo="label+percent", textposition="outside",
            pull=[0.03 if v == max(values) else 0 for v in values],
            marker=dict(colors=[
                "#2E86C1", "#1ABC9C", "#F39C12", "#E74C3C", "#8E44AD",
                "#3498DB", "#2ECC71", "#E67E22", "#9B59B6", "#1F77B4",
            ][:len(values)]),
        )])
        fig_pie.update_layout(
            title=dict(text=f"{risk_profile}  ·  {current_regime} Regime", font=dict(size=16)),
            height=520, showlegend=False, margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Weight Breakdown")
        rows = []
        for t in sorted(portfolio, key=lambda x: -portfolio[x]):
            if portfolio[t] > 0:
                rows.append({
                    "Ticker": t, "Asset Class": ASSET_TICKERS[t],
                    "Weight": f"{portfolio[t] * 100:.1f}%",
                    "Base": f"{BASE_WEIGHTS[risk_profile][t] * 100:.1f}%",
                    "Tilt": f"{REGIME_TILTS[current_regime][t]:+.0%}",
                    "Amount": f"${initial_investment * portfolio[t]:,.0f}",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown(
            f"**How it works:** Your base allocation comes from the "
            f"**{risk_profile}** profile (MiFID questionnaire). "
            f"The current **{current_regime}** regime applies tilts — "
            f"{'overweighting equities and growth assets' if current_regime in ('Expansion', 'Recovery') else 'shifting toward bonds, gold, and cash for protection'}. "
            f"Weights are clipped to ≥0% and normalised to 100%."
        )

        st.markdown("---")
        st.subheader("Asset Class Summary")
        groups = {
            "Equities": ["SPY", "IWM", "EFA", "EEM"],
            "Fixed Income": ["AGG", "TLT", "LQD"],
            "Alternatives": ["VNQ", "GLD"],
            "Cash": ["SHV"],
        }
        group_rows = []
        for gname, tickers in groups.items():
            pct = sum(portfolio.get(t, 0) for t in tickers) * 100
            amt = initial_investment * pct / 100
            group_rows.append({"Group": gname, "Weight": f"{pct:.1f}%", "Amount": f"${amt:,.0f}"})
        st.dataframe(pd.DataFrame(group_rows), hide_index=True, use_container_width=True)

# =============================================================================
# TAB 2 — 5-YEAR P/L FORECAST
# =============================================================================

with tab2:
    st.subheader("Monte Carlo 5-Year Forecast")
    st.caption("10,000 simulated paths · Two-phase model: 12 months current-regime stats → 48 months full-history mean reversion")

    paths, _ = regime_conditioned_forecast(returns_monthly, portfolio, dataset)
    forecast = extract_forecast_stats(paths)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["P95 (Best)"], mode="lines", line=dict(width=0), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["P5 (Worst)"], fill="tonexty", fillcolor="rgba(46,117,182,0.10)", line=dict(width=0), name="90% band (P5–P95)"))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["P75"], mode="lines", line=dict(width=0), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["P25"], fill="tonexty", fillcolor="rgba(46,117,182,0.25)", line=dict(width=0), name="50% band (P25–P75)"))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["P50 (Median)"], mode="lines", line=dict(color=PALETTE["primary"], width=3), name=f'Median: {forecast.loc[60, "P50 (Median)"]:+.1f}%'))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast["Mean"], mode="lines", line=dict(color=PALETTE["danger"], width=2, dash="dash"), name=f'Mean: {forecast.loc[60, "Mean"]:+.1f}%'))
    fig_fc.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)

    fig_fc.update_layout(
        title=f"5-Year P/L — {risk_profile} · {current_regime}",
        xaxis_title="Months", yaxis_title="Cumulative Return (%)", yaxis_ticksuffix="%",
        xaxis=dict(tickvals=[0, 12, 24, 36, 48, 60], ticktext=["Start", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]),
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Year-by-Year Summary")
        summary_rows = []
        for m in [12, 24, 36, 48, 60]:
            med = forecast.loc[m, "P50 (Median)"]
            summary_rows.append({
                "Year": m // 12,
                "Worst (5th)": f'{forecast.loc[m, "P5 (Worst)"]:+.1f}%',
                "Lower (25th)": f'{forecast.loc[m, "P25"]:+.1f}%',
                "Median": f'{med:+.1f}%',
                "Upper (75th)": f'{forecast.loc[m, "P75"]:+.1f}%',
                "Best (95th)": f'{forecast.loc[m, "P95 (Best)"]:+.1f}%',
                "Median $": f"${initial_investment * (1 + med / 100):,.0f}",
            })
        st.dataframe(pd.DataFrame(summary_rows).set_index("Year"), use_container_width=True)

    with col2:
        st.subheader("5-Year Outcome")
        med_5y = forecast.loc[60, "P50 (Median)"]
        worst_5y = forecast.loc[60, "P5 (Worst)"]
        best_5y = forecast.loc[60, "P95 (Best)"]
        st.metric("Median Return", f"{med_5y:+.1f}%", delta=f"${initial_investment * med_5y / 100:+,.0f}")
        st.metric("Downside (5th)", f"{worst_5y:+.1f}%")
        st.metric("Upside (95th)", f"{best_5y:+.1f}%")
        prob_loss = (paths[:, -1] < 0).mean() * 100
        st.metric("Prob. of Loss (5yr)", f"{prob_loss:.1f}%")

# =============================================================================
# TAB 3 — PROFILE COMPARISON
# =============================================================================

with tab3:
    st.subheader("Side-by-Side Profile Comparison")
    st.markdown(f"All four profiles under the current **{current_regime}** regime:")

    comparison_data = {p: get_portfolio(p, current_regime) for p in RISK_PROFILES}

    asset_order = ["SPY", "IWM", "EFA", "EEM", "AGG", "TLT", "LQD", "VNQ", "GLD", "SHV"]
    bar_colors = ["#2E86C1", "#1ABC9C", "#F39C12", "#E74C3C", "#8E44AD", "#3498DB", "#2ECC71", "#E67E22", "#9B59B6", "#95A5A6"]

    fig_comp = go.Figure()
    for j, ticker in enumerate(asset_order):
        fig_comp.add_trace(go.Bar(
            name=f"{ASSET_TICKERS[ticker]} ({ticker})",
            x=RISK_PROFILES,
            y=[comparison_data[p].get(ticker, 0) * 100 for p in RISK_PROFILES],
            marker_color=bar_colors[j],
        ))

    fig_comp.update_layout(
        barmode="stack",
        title=f"Portfolio Allocation by Profile — {current_regime} Regime",
        yaxis_title="Allocation (%)", yaxis_ticksuffix="%",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
    )

    for i, profile in enumerate(RISK_PROFILES):
        if profile == risk_profile:
            fig_comp.add_annotation(x=profile, y=105, text="◆ YOU", showarrow=False,
                                    font=dict(size=12, color=PALETTE["primary"]))

    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Metrics by Profile")
    metric_rows = []
    for profile in RISK_PROFILES:
        p = comparison_data[profile]
        eq = sum(p[t] for t in ["SPY", "IWM", "EFA", "EEM"]) * 100
        fi = sum(p[t] for t in ["AGG", "TLT", "LQD"]) * 100
        alt = sum(p[t] for t in ["VNQ", "GLD"]) * 100
        cash = p.get("SHV", 0) * 100

        pths, _ = regime_conditioned_forecast(returns_monthly, p, dataset, n_sims=2000)
        med_ret = np.percentile(pths[:, -1], 50) * 100
        p_loss = (pths[:, -1] < 0).mean() * 100

        marker = " ← YOU" if profile == risk_profile else ""
        metric_rows.append({
            "Profile": f"{PROFILE_STYLE[profile]['emoji']} {profile}{marker}",
            "Equities": f"{eq:.0f}%", "Fixed Income": f"{fi:.0f}%",
            "Alternatives": f"{alt:.0f}%", "Cash": f"{cash:.0f}%",
            "5yr Median": f"{med_ret:+.1f}%", "5yr Loss Prob.": f"{p_loss:.1f}%",
        })
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)

# =============================================================================
# TAB 4 — RISK PROFILE BREAKDOWN
# =============================================================================

with tab4:
    st.subheader("MiFID Dimension Analysis")
    st.markdown("Your risk profile is built from **6 dimensions** based on MiFID II. Each dimension scores 1–4.")

    dim_labels = {
        "A": "Financial Knowledge\n& Experience", "B": "Financial Situation\n& Capacity",
        "C": "Investment Objectives\n& Time Horizon", "D": "Risk Tolerance\n& Loss Capacity",
        "E": "Behavioural Response\nto Volatility", "F": "Liquidity Needs\n& Constraints",
    }

    dim_scores = {}
    for dim_letter, dim_name in dim_labels.items():
        scores_in_dim = [answers_by_index[i] for i, q in enumerate(QUESTIONNAIRE) if q["dimension"] == dim_letter]
        dim_scores[dim_name] = np.mean(scores_in_dim)

    categories = list(dim_scores.keys())
    values_list = list(dim_scores.values())
    values_closed = values_list + [values_list[0]]

    col1, col2 = st.columns([3, 2])

    with col1:
        pc = pstyle["color"]
        r, g, b = int(pc[1:3], 16), int(pc[3:5], 16), int(pc[5:7], 16)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values_closed, theta=categories + [categories[0]],
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.2)",
            line=dict(color=pstyle["color"], width=2.5), name="Your Profile",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 4.5], tickvals=[1, 2, 3, 4])),
            title=f"Dimension Scores — {pstyle['emoji']} {risk_profile}",
            height=480, showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("**Dimension Scores**")
        dim_table = []
        for dim_name, score in dim_scores.items():
            bar = "█" * int(round(score)) + "░" * (4 - int(round(score)))
            level = "Low" if score <= 1.5 else "Moderate" if score <= 2.5 else "High" if score <= 3.5 else "Very High"
            dim_table.append({"Dimension": dim_name.replace("\n", " "), "Score": f"{score:.1f}", "Visual": bar, "Level": level})
        st.dataframe(pd.DataFrame(dim_table), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**Profile Bands**")
        st.markdown("""
| Score | Profile |
|-------|---------|
| 14–24 | 🟢 Conservative |
| 25–35 | 🔵 Moderately Conservative |
| 36–46 | 🟠 Moderately Aggressive |
| 47–56 | 🔴 Aggressive |
        """)
        st.markdown(f"**Your score: {total_score} / {MAX_SCORE} → {pstyle['emoji']} {risk_profile}**")

        st.markdown("---")
        st.markdown("**MiFID Validation**")
        st.caption(
            "The sample MiFID client (Diaphanum assessment) scored highest "
            "on all dimensions → classified as Aggressive. Our algorithm "
            "replicates this (score 56/56 → Aggressive)."
        )

# =============================================================================
# TAB 5 — REGIME ANALYSIS
# =============================================================================

with tab5:
    st.subheader("Macroeconomic Regime Analysis")

    regime_series = dataset["Regime"]

    fig_timeline = go.Figure()
    for regime, color in REGIME_COLORS.items():
        mask = regime_series == regime
        dates = regime_series.index[mask]
        for d in dates:
            fig_timeline.add_vrect(x0=d, x1=d + pd.DateOffset(months=1), fillcolor=color, opacity=0.6, layer="below", line_width=0)
    for regime, color in REGIME_COLORS.items():
        fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=color), name=regime))
    fig_timeline.update_layout(
        title="Regime Timeline (Dec 2020 – Sep 2025)", xaxis_title="Date", yaxis_visible=False,
        height=220, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Key Macro Indicators Over Time")
    fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                              subplot_titles=("Interest Rates & Yields", "Inflation (CPI Indices)", "Real Economy"))

    for col, color, name in [("Fed Funds Rate", "#2E86C1", "Fed Funds"), ("10Y Treasury Yield", "#E74C3C", "10Y Yield"), ("2Y Treasury Yield", "#F39C12", "2Y Yield")]:
        if col in dataset.columns:
            fig_macro.add_trace(go.Scatter(x=dataset.index, y=dataset[col], name=name, line=dict(color=color, width=2)), row=1, col=1)

    for col, color, name in [("CPI All Items", "#8E44AD", "CPI All Items"), ("Core CPI", "#1ABC9C", "Core CPI")]:
        if col in dataset.columns:
            fig_macro.add_trace(go.Scatter(x=dataset.index, y=dataset[col], name=name, line=dict(color=color, width=2)), row=2, col=1)

    for col, color, name in [("Industrial Production Index", "#2E86C1", "Industrial Prod."), ("Unemployment Rate", "#C0392B", "Unemployment (%)")]:
        if col in dataset.columns:
            fig_macro.add_trace(go.Scatter(x=dataset.index, y=dataset[col], name=name, line=dict(color=color, width=2)), row=3, col=1)

    fig_macro.update_layout(height=700, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=-0.08), margin=dict(t=40))
    st.plotly_chart(fig_macro, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Regime Distribution")
        dist = regime_series.value_counts()
        fig_dist = go.Figure(data=[go.Bar(
            x=dist.index, y=dist.values,
            marker_color=[REGIME_COLORS.get(r, "grey") for r in dist.index],
            text=[f"{v} mo" for v in dist.values], textposition="auto",
        )])
        fig_dist.update_layout(height=300, template="plotly_white", yaxis_title="Months", margin=dict(t=20))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader(f"Current Signals → {current_regime}")
        latest = dataset.iloc[-1]
        yc = latest.get("10Y Treasury Yield", 0) - latest.get("2Y Treasury Yield", 0)
        signal_data = {
            "Yield Curve (10Y−2Y)": (f"{yc:.2f}%", "⚠️ Inverted" if yc < 0 else "✅ Normal"),
            "Fed Funds Rate": (f"{latest.get('Fed Funds Rate', 0):.2f}%", ""),
            "Unemployment Rate": (f"{latest.get('Unemployment Rate', 0):.1f}%", ""),
            "VIX": (f"{latest.get('VIX', 0):.1f}", "⚠️ Elevated" if latest.get('VIX', 0) > 25 else "✅ Normal"),
            "CPI All Items": (f"{latest.get('CPI All Items', 0):.1f}", ""),
            "Industrial Production": (f"{latest.get('Industrial Production Index', 0):.1f}", ""),
        }
        signal_rows = [{"Indicator": k, "Value": v, "Status": s} for k, (v, s) in signal_data.items()]
        st.dataframe(pd.DataFrame(signal_rows), hide_index=True, use_container_width=True)

# =============================================================================
# TAB 6 — RAW DATA & STATS
# =============================================================================

with tab6:
    st.subheader("Merged Dataset (Last 24 Months)")
    st.dataframe(dataset.tail(24), use_container_width=True)

    st.subheader("Monthly Asset Returns (Last 12)")
    st.dataframe(returns_monthly.tail(12).style.format("{:.4f}"), use_container_width=True)

    st.subheader("Asset Correlation Matrix")
    corr = returns_monthly.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu_r", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig_corr.update_layout(height=500, title="Return Correlations", template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Descriptive Statistics (Monthly Returns)")
    desc = returns_monthly.describe().T
    desc["Ann. Return"] = returns_monthly.mean() * 12
    desc["Ann. Volatility"] = returns_monthly.std() * np.sqrt(12)
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

# =============================================================================
# TAB 7 — ABOUT / METHODOLOGY
# =============================================================================

with tab7:
    st.subheader("About This Tool")

    st.markdown("""
**Business Problem**

Inexperienced investors face fragmented and complex financial information that is
not tailored to their individual risk tolerance, financial situation, or behavioural
preferences. The EU's MiFID II regulation requires financial advisors to assess
client suitability through structured questionnaires — but translating those
assessments into actionable, personalised portfolio recommendations remains
a manual and opaque process.

**Our Solution**

This tool bridges the gap between regulatory risk assessment and portfolio
construction by combining three analytical layers:

1. **MiFID II Risk Profiling** — A 14-question suitability assessment across
   6 dimensions (knowledge, financial capacity, objectives, risk tolerance,
   behavioural tendencies, and liquidity needs) that classifies investors into
   one of four profiles.

2. **Macroeconomic Regime Analysis** — A rule-based classifier that evaluates
   five macro signals (yield curve, unemployment trends, industrial production,
   VIX, and Fed Funds hiking pace) to categorise the current economic environment
   as Expansion, Slowdown, Contraction, or Recovery.

3. **Regime-Adjusted Portfolio Optimisation** — Strategic base allocations for
   each risk profile are dynamically tilted based on the current regime, producing
   a personalised portfolio across 10 asset classes.

**Forecasting**

The 5-year P/L forecast uses a two-phase Monte Carlo simulation (10,000 paths):
- **Phase 1** (months 1–12): Return statistics drawn from the current regime
- **Phase 2** (months 13–60): Full-history statistics reflecting mean reversion

This captures short-term regime persistence while acknowledging that economic
conditions evolve over longer horizons.
    """)

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown("""
| Source | Data | Period |
|--------|------|--------|
| Custom Excel Dataset | 7 macroeconomic indicators (Fed Funds, 10Y/2Y yields, CPI, Core CPI, Industrial Production, Unemployment) | Dec 2020 – Sep 2025 |
| Yahoo Finance (yfinance) | 10 ETF prices (SPY, IWM, EFA, EEM, AGG, TLT, LQD, VNQ, GLD, SHV) + CBOE VIX | Aligned to macro range |
    """)

    st.markdown("---")
    st.subheader("Technical Architecture")
    st.code("""
Excel Macro Data ──┐
                    ├──→ Monthly Merged Dataset ──→ Regime Classifier ──┐
Yahoo Finance ─────┘                                (4 states)          │
                                                                         │
MiFID Questionnaire ──→ Risk Profile (4 types) ────────────────────────┤
                                                                         │
                                              ┌──────────────────────────┘
                                              ▼
                                    Portfolio Optimizer
                                    (base weights + regime tilts)
                                              │
                                              ▼
                                    Monte Carlo Engine
                                    (two-phase, 10K × 60 months)
                                              │
                                              ▼
                                    Streamlit Dashboard
    """, language=None)

    st.markdown("---")
    st.markdown("**EDI 3600 Digital Business Analysis** · Spring 2026 · BI Norwegian Business School")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "EDI 3600 Digital Business Analysis · Personality-Based Portfolio "
    "Recommendation Engine · MiFID II Risk Profiling + Macroeconomic "
    "Regime Analysis + Monte Carlo Forecasting"
)
