"""
EDI 3600 — Personality-Based Portfolio Recommendation Algorithm
================================================================
Modules:
  1. Data Pipeline        — Macro data from Excel + asset prices from yfinance
  2. Regime Classifier    — Expansion / Slowdown / Contraction / Recovery
  3. MiFID Risk Profiler  — Conservative / Moderately Conservative /
                            Moderately Aggressive / Aggressive
  4. Portfolio Optimizer   — Regime-adjusted weights (Markowitz-inspired)
  5. 5-Year P/L Forecast  — Monte Carlo simulation with confidence bands

Requirements:
  pip install pandas numpy yfinance openpyxl scipy matplotlib

IMPORTANT: Place 'Macroeconomic_Dataset_DBA.xlsx' in the SAME folder as this script.
"""

import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize


# =============================================================================
# MODULE 1 — DATA PIPELINE
# =============================================================================

# --- Mapping: Excel sheet name → (value column, clean name) ---
EXCEL_SHEET_MAP = {
    "Federal Funds Rate":            ("FEDFUNDS",   "Fed Funds Rate"),
    "10-Year Treasury Yield":        ("GS10",       "10Y Treasury Yield"),
    "2-Year Treasury Yield ":        ("HQMCB2YRP",  "2Y Treasury Yield"),
    "CPI All Items":                 ("CPIAUCSL",   "CPI All Items"),
    "Core CPI ":                     ("CPILFESL",   "Core CPI"),
    "Industrial Production Index ":  ("INDPRO",     "Industrial Production Index"),
    "Unemployment Rate (UNRATE)":    ("UNRATE",     "Unemployment Rate"),
}

MACRO_EXCEL_FILE = "Macroeconomic_Dataset_DBA.xlsx"

# --- Asset-class ETF tickers ---
ASSET_TICKERS = {
    "SPY": "US Large-Cap Equities",
    "IWM": "US Small-Cap Equities",
    "EFA": "International Equities",
    "EEM": "Emerging Markets",
    "AGG": "US Aggregate Bonds",
    "TLT": "Long-Term Treasuries",
    "LQD": "Corporate Bonds",
    "VNQ": "Real Estate (REITs)",
    "GLD": "Gold / Commodities",
    "SHV": "Cash / Money Market",
}

TICKERS = list(ASSET_TICKERS.keys())


def load_macro_from_excel(filepath=MACRO_EXCEL_FILE):
    """
    Load macroeconomic data from the provided Excel workbook.
    Each sheet contains one indicator with columns: observation_date, <value>.
    Returns a single DataFrame indexed by date with clean column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cannot find '{filepath}'. "
            f"Place Macroeconomic_Dataset_DBA.xlsx in the same folder as this script."
        )

    print(f"[1/3] Loading macro data from {filepath}...")
    frames = []
    xls = pd.ExcelFile(filepath)

    for sheet_name, (value_col, clean_name) in EXCEL_SHEET_MAP.items():
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df = df[["observation_date", value_col]].copy()
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df = df.set_index("observation_date")
            df.columns = [clean_name]
            frames.append(df)
            print(f"       ✓ {clean_name}")
        except Exception as e:
            print(f"       ✗ {clean_name} — {e}")

    macro = pd.concat(frames, axis=1).sort_index()
    print(f"       {len(macro)} rows  |  {macro.index[0].date()} → {macro.index[-1].date()}")
    return macro


def fetch_data(macro_file=MACRO_EXCEL_FILE):
    """
    Build the full dataset:
      - Macro indicators from YOUR Excel file
      - Asset prices + VIX from yfinance (aligned to the same date range)
    """
    macro = load_macro_from_excel(macro_file)
    start_date = macro.index[0].strftime("%Y-%m-%d")

    print("[2/3] Fetching asset prices & VIX from yfinance...")
    prices = yf.download(TICKERS, start=start_date, auto_adjust=True)["Close"]
    returns = prices.pct_change().dropna()

    vix = yf.download("^VIX", start=start_date, auto_adjust=True)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.squeeze()
    vix.name = "VIX"

    print("[3/3] Aligning to monthly frequency...")
    macro_m = macro.resample("ME").last()
    returns_m = returns.resample("ME").sum()
    vix_m = vix.resample("ME").last()

    dataset = macro_m.join(vix_m).join(returns_m)
    dataset = dataset.dropna(how="all").ffill()

    print(f"    Final: {dataset.shape[0]} months × {dataset.shape[1]} columns")
    print(f"    Range: {dataset.index[0].date()} → {dataset.index[-1].date()}")
    return dataset, returns_m


# =============================================================================
# MODULE 2 — REGIME CLASSIFIER
# =============================================================================

def classify_regime(df):
    """
    Rule-based macro-regime classifier using Excel indicators:
      - Yield curve (10Y − 2Y)
      - Unemployment trend
      - Industrial production growth
      - VIX level
      - Fed Funds rate hiking pace
    """
    signals = pd.DataFrame(index=df.index)

    # 1. Yield Curve Spread (10Y − 2Y): negative = recession warning
    signals["yield_curve"] = df["10Y Treasury Yield"] - df["2Y Treasury Yield"]

    # 2. Unemployment trend: rising over 3 months = deteriorating labour market
    signals["unemp_rising"] = (df["Unemployment Rate"].diff(3) > 0).astype(int)

    # 3. Industrial Production: 6-month % change
    signals["ip_growth"] = df["Industrial Production Index"].pct_change(6)

    # 4. VIX level: > 25 = elevated fear
    signals["high_vix"] = (df["VIX"] > 25).astype(int)

    # 5. Fed Funds Rate: rapid tightening (rising > 1pp in 6 months) = stress
    signals["rate_hiking"] = (df["Fed Funds Rate"].diff(6) > 1.0).astype(int)

    # Composite stress score (higher = worse)
    signals["stress_score"] = (
        (signals["yield_curve"] < 0).astype(int) * 2  # inverted curve (double weight)
        + signals["unemp_rising"]
        + (signals["ip_growth"] < 0).astype(int)
        + signals["high_vix"]
        + signals["rate_hiking"]
    )

    conditions = [
        signals["stress_score"] >= 4,
        (signals["stress_score"] >= 2) & (signals["ip_growth"] < 0),
        (signals["stress_score"] >= 2) & (signals["ip_growth"] >= 0),
    ]
    choices = ["Contraction", "Slowdown", "Recovery"]
    return pd.Series(
        np.select(conditions, choices, default="Expansion"),
        index=df.index, name="Regime",
    )


def validate_regime(dataset):
    """Print regime distribution and spot-check key periods."""
    print("\n--- Regime Distribution ---")
    print(dataset["Regime"].value_counts().to_string())
    covid_data = dataset.loc["2020-01":"2021-06", "Regime"]
    if len(covid_data) > 0:
        print("\n--- COVID / post-COVID period ---")
        print(covid_data.to_string())
    print(f"\nCurrent regime: {dataset['Regime'].iloc[-1]}")


# =============================================================================
# MODULE 3 — MiFID-BASED RISK-PROFILE QUESTIONNAIRE
# =============================================================================
#
# Inspired by MiFID II suitability assessment (Diaphanum framework).
# The questionnaire evaluates 6 dimensions that map to EU regulatory
# requirements for investor classification:
#
#   Dimension A — Financial Knowledge & Experience (MiFID Appropriateness)
#   Dimension B — Financial Situation & Capacity (MiFID Suitability §1-9)
#   Dimension C — Investment Objectives & Time Horizon (MiFID Suitability §F1-F3)
#   Dimension D — Risk Tolerance & Loss Capacity (MiFID Suitability §F4-F5)
#   Dimension E — Behavioural Response to Volatility (MiFID Suitability §F8-F9)
#   Dimension F — Liquidity Needs & Constraints (MiFID Suitability §F6-F7)
#
# Each question awards 1-4 points. Total score maps to four profiles:
#   Conservative / Moderately Conservative / Moderately Aggressive / Aggressive
#
# Validation: The MiFID sample client (Kent Stig Hagbarth) scored
#   "Aggressive" on the Diaphanum assessment. His answers — university
#   education, discretionary management experience, >€2M financial wealth,
#   >€500K income, >5yr horizon, >15% loss tolerance, >50% equity preference,
#   and willingness to increase positions after a 25% drawdown — should
#   produce an "Aggressive" classification from our algorithm as well.
# =============================================================================

RISK_PROFILES = ["Conservative", "Moderately Conservative",
                 "Moderately Aggressive", "Aggressive"]

QUESTIONNAIRE = [
    # ── Dimension A: Financial Knowledge & Experience ──────────────────
    {
        "dimension": "A",
        "dimension_name": "Financial Knowledge & Experience",
        "question": "1. What is your level of financial education and experience?",
        "options": {
            "a": ("No formal financial education; limited investment experience", 1),
            "b": ("University-level education; familiar with basic products (deposits, bonds)", 2),
            "c": ("University education with some finance courses; experience with equities and funds", 3),
            "d": ("Finance degree or professional experience; active with diverse financial instruments", 4),
        },
    },
    {
        "dimension": "A",
        "dimension_name": "Financial Knowledge & Experience",
        "question": "2. Which investment services have you used in the past two years?",
        "options": {
            "a": ("None — I have not used any investment services", 1),
            "b": ("Basic order execution (buying/selling through a broker)", 2),
            "c": ("Investment advisory services", 3),
            "d": ("Discretionary portfolio management (a professional manages my portfolio)", 4),
        },
    },
    {
        "dimension": "A",
        "dimension_name": "Financial Knowledge & Experience",
        "question": "3. How well do you understand financial risk concepts (e.g., credit risk, "
                    "liquidity risk, market volatility)?",
        "options": {
            "a": ("I am not familiar with these concepts", 1),
            "b": ("I have a basic understanding of one or two of them", 2),
            "c": ("I understand most of them and how they affect my investments", 3),
            "d": ("I have a thorough understanding and can explain how they interact", 4),
        },
    },

    # ── Dimension B: Financial Situation & Capacity ────────────────────
    {
        "dimension": "B",
        "dimension_name": "Financial Situation & Capacity",
        "question": "4. What is the approximate value of your investable financial assets?",
        "options": {
            "a": ("Less than €50,000", 1),
            "b": ("€50,000 – €250,000", 2),
            "c": ("€250,000 – €1,000,000", 3),
            "d": ("More than €1,000,000", 4),
        },
    },
    {
        "dimension": "B",
        "dimension_name": "Financial Situation & Capacity",
        "question": "5. What portion of your total wealth will this investment represent?",
        "options": {
            "a": ("More than 75% — this is nearly all of my wealth", 1),
            "b": ("50% – 75%", 2),
            "c": ("25% – 50%", 3),
            "d": ("Less than 25% — I have substantial other assets", 4),
        },
    },
    {
        "dimension": "B",
        "dimension_name": "Financial Situation & Capacity",
        "question": "6. How would you describe your annual savings capacity "
                    "(income minus expenses, averaged over the past 3 years)?",
        "options": {
            "a": ("I have minimal or no savings capacity", 1),
            "b": ("I save a small amount (less than €10,000/year)", 2),
            "c": ("I save a moderate amount (€10,000 – €30,000/year)", 3),
            "d": ("I save significantly (more than €30,000/year)", 4),
        },
    },

    # ── Dimension C: Investment Objectives & Time Horizon ──────────────
    {
        "dimension": "C",
        "dimension_name": "Investment Objectives & Time Horizon",
        "question": "7. What is the primary purpose of your investment?",
        "options": {
            "a": ("Preserve my capital and protect against inflation", 1),
            "b": ("Generate regular income with moderate capital growth", 2),
            "c": ("Achieve capital growth over the medium to long term", 3),
            "d": ("Maximise returns and achieve high capital growth", 4),
        },
    },
    {
        "dimension": "C",
        "dimension_name": "Investment Objectives & Time Horizon",
        "question": "8. What is your expected investment time horizon?",
        "options": {
            "a": ("Less than 1 year", 1),
            "b": ("1 to 3 years", 2),
            "c": ("3 to 5 years", 3),
            "d": ("More than 5 years", 4),
        },
    },

    # ── Dimension D: Risk Tolerance & Loss Capacity ────────────────────
    {
        "dimension": "D",
        "dimension_name": "Risk Tolerance & Loss Capacity",
        "question": "9. What maximum loss would you be willing to accept on your "
                    "portfolio at the end of your investment horizon?",
        "options": {
            "a": ("0% — I cannot accept any loss of capital", 1),
            "b": ("Up to 10% of the invested value", 2),
            "c": ("Up to 15% of the invested value", 3),
            "d": ("More than 15% — I accept significant risk for higher returns", 4),
        },
    },
    {
        "dimension": "D",
        "dimension_name": "Risk Tolerance & Loss Capacity",
        "question": "10. What percentage of equities (stocks) would you like to "
                     "maintain in your portfolio?",
        "options": {
            "a": ("0% – 5% (almost no equities)", 1),
            "b": ("5% – 25%", 2),
            "c": ("25% – 50%", 3),
            "d": ("More than 50%", 4),
        },
    },

    # ── Dimension E: Behavioural Response to Volatility ────────────────
    {
        "dimension": "E",
        "dimension_name": "Behavioural Response to Volatility",
        "question": "11. If the value of your investments decreased by more than 25%, "
                     "what would you do?",
        "options": {
            "a": ("Sell all my investment positions immediately", 1),
            "b": ("Sell part of the investment to reduce exposure", 2),
            "c": ("Keep my investment and wait for recovery", 3),
            "d": ("Increase my investment positions to take advantage of lower prices", 4),
        },
    },
    {
        "dimension": "E",
        "dimension_name": "Behavioural Response to Volatility",
        "question": "12. Which statement best describes your attitude toward "
                     "investment risk and return?",
        "options": {
            "a": ("Capital preservation — I accept lower returns for greater security", 1),
            "b": ("Some risk for moderate returns — I tolerate small potential losses", 2),
            "c": ("Higher risk for higher returns — I tolerate losses for some time", 3),
            "d": ("Very high growth — I accept significant risk in my investments", 4),
        },
    },

    # ── Dimension F: Liquidity Needs & Constraints ─────────────────────
    {
        "dimension": "F",
        "dimension_name": "Liquidity Needs & Constraints",
        "question": "13. What percentage of your investment do you expect to need "
                     "for liquidity (cash) purposes within the next 24 months?",
        "options": {
            "a": ("More than 75%", 1),
            "b": ("50% – 75%", 2),
            "c": ("25% – 50%", 3),
            "d": ("Less than 25%", 4),
        },
    },
    {
        "dimension": "F",
        "dimension_name": "Liquidity Needs & Constraints",
        "question": "14. What percentage of your annual financial returns do you need "
                     "to cover your current financial commitments and living expenses?",
        "options": {
            "a": ("More than 75%", 1),
            "b": ("50% – 75%", 2),
            "c": ("25% – 50%", 3),
            "d": ("Less than 25%", 4),
        },
    },
]

NUM_QUESTIONS = len(QUESTIONNAIRE)     # 14
MAX_SCORE = NUM_QUESTIONS * 4          # 56
MIN_SCORE = NUM_QUESTIONS * 1          # 14


def run_questionnaire():
    """Interactive CLI questionnaire. Returns total score."""
    print("\n" + "=" * 60)
    print("   MiFID-BASED INVESTOR RISK-PROFILE QUESTIONNAIRE")
    print("=" * 60)
    print(f"   ({NUM_QUESTIONS} questions · 6 dimensions)\n")

    total = 0
    current_dim = None
    for q in QUESTIONNAIRE:
        # Print dimension header when it changes
        if q["dimension"] != current_dim:
            current_dim = q["dimension"]
            print(f"\n── Dimension {current_dim}: {q['dimension_name']} ──")

        print(f"\n{q['question']}")
        for key, (text, _) in q["options"].items():
            print(f"   {key}) {text}")
        while True:
            answer = input("Your choice (a/b/c/d): ").strip().lower()
            if answer in q["options"]:
                total += q["options"][answer][1]
                break
            print("   Invalid — please enter a, b, c, or d.")

    print(f"\n   Total score: {total} / {MAX_SCORE}")
    return total


def classify_risk_profile(total_score):
    """
    Map questionnaire score to one of four MiFID-aligned risk profiles.

    Scoring bands (out of 56):
      14–24  →  Conservative           (≤ 43% of max)
      25–35  →  Moderately Conservative (≤ 63% of max)
      36–46  →  Moderately Aggressive   (≤ 82% of max)
      47–56  →  Aggressive              (> 82% of max)

    Validation: Kent's MiFID answers map to all (d) choices → score 56 → Aggressive ✓
    """
    if total_score <= 24:
        return "Conservative"
    elif total_score <= 35:
        return "Moderately Conservative"
    elif total_score <= 46:
        return "Moderately Aggressive"
    else:
        return "Aggressive"


def score_dimension(answers_dict, dimension_letter):
    """
    Given a dict of {question_index: score}, compute the average score
    for questions belonging to the specified dimension.
    Useful for the Streamlit dashboard's dimension breakdown.
    """
    dim_scores = []
    for i, q in enumerate(QUESTIONNAIRE):
        if q["dimension"] == dimension_letter and i in answers_dict:
            dim_scores.append(answers_dict[i])
    return np.mean(dim_scores) if dim_scores else 0.0


# =============================================================================
# MODULE 4 — PORTFOLIO OPTIMIZER (REGIME-ADJUSTED WEIGHTS)
# =============================================================================
#
# Four risk profiles with distinct strategic asset allocations.
# The "Moderately Conservative" and "Moderately Aggressive" tiers
# fill the gap between the original Conservative and Aggressive,
# providing finer granularity aligned with MiFID suitability bands.
# =============================================================================

BASE_WEIGHTS = {
    "Conservative": {
        "SPY": 0.10, "IWM": 0.00, "EFA": 0.05, "EEM": 0.00,
        "AGG": 0.30, "TLT": 0.20, "LQD": 0.10, "VNQ": 0.05,
        "GLD": 0.10, "SHV": 0.10,
    },
    "Moderately Conservative": {
        "SPY": 0.18, "IWM": 0.02, "EFA": 0.08, "EEM": 0.02,
        "AGG": 0.25, "TLT": 0.15, "LQD": 0.08, "VNQ": 0.05,
        "GLD": 0.07, "SHV": 0.10,
    },
    "Moderately Aggressive": {
        "SPY": 0.28, "IWM": 0.07, "EFA": 0.12, "EEM": 0.06,
        "AGG": 0.15, "TLT": 0.08, "LQD": 0.05, "VNQ": 0.06,
        "GLD": 0.05, "SHV": 0.08,
    },
    "Aggressive": {
        "SPY": 0.40, "IWM": 0.10, "EFA": 0.15, "EEM": 0.10,
        "AGG": 0.05, "TLT": 0.05, "LQD": 0.00, "VNQ": 0.05,
        "GLD": 0.05, "SHV": 0.05,
    },
}

REGIME_TILTS = {
    "Expansion": {
        "SPY": +0.05, "IWM": +0.03, "EFA": +0.02, "EEM": +0.02,
        "AGG": -0.05, "TLT": -0.03, "LQD":  0.00, "VNQ": +0.02,
        "GLD": -0.03, "SHV": -0.03,
    },
    "Slowdown": {
        "SPY": -0.03, "IWM": -0.02, "EFA": -0.02, "EEM": -0.03,
        "AGG": +0.03, "TLT": +0.02, "LQD": +0.02, "VNQ": -0.02,
        "GLD": +0.03, "SHV": +0.02,
    },
    "Contraction": {
        "SPY": -0.05, "IWM": -0.05, "EFA": -0.03, "EEM": -0.05,
        "AGG": +0.05, "TLT": +0.05, "LQD": -0.02, "VNQ": -0.03,
        "GLD": +0.05, "SHV": +0.08,
    },
    "Recovery": {
        "SPY": +0.03, "IWM": +0.02, "EFA": +0.01, "EEM": +0.01,
        "AGG": -0.02, "TLT": -0.02, "LQD": +0.01, "VNQ": +0.01,
        "GLD":  0.00, "SHV": -0.05,
    },
}


def get_portfolio(risk_profile, regime):
    """Combine base weights with regime tilts, clip and normalise."""
    base = BASE_WEIGHTS[risk_profile].copy()
    tilt = REGIME_TILTS[regime]
    adjusted = {a: max(base[a] + tilt[a], 0) for a in base}
    total = sum(adjusted.values())
    return {k: round(v / total, 4) for k, v in adjusted.items()}


def markowitz_optimize(returns_df, risk_aversion=2):
    """Optional: mean-variance optimisation overlay."""
    mu = returns_df.mean() * 12
    cov = returns_df.cov() * 12
    n = len(mu)

    def objective(w):
        return -(w @ mu - risk_aversion * (w @ cov @ w))

    result = minimize(
        objective, np.ones(n) / n, method="SLSQP",
        bounds=[(0, 0.4)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    return dict(zip(returns_df.columns, np.round(result.x, 4)))


# =============================================================================
# MODULE 5 — 5-YEAR P/L FORECAST (MONTE CARLO)
# =============================================================================

def portfolio_return_series(returns_df, weights_dict):
    """Weighted monthly return series for the portfolio."""
    t = list(weights_dict.keys())
    w = np.array([weights_dict[tk] for tk in t])
    return pd.Series(returns_df[t].values @ w, index=returns_df.index, name="Portfolio")


def monte_carlo_forecast(mu_monthly, sigma_monthly, months=60, n_sims=10_000, seed=42):
    """GBM Monte Carlo → array (n_sims, months+1) of cumulative P/L."""
    np.random.seed(seed)
    drift = mu_monthly - 0.5 * sigma_monthly**2
    shocks = np.random.normal(drift, sigma_monthly, size=(n_sims, months))
    cum = np.exp(np.cumsum(shocks, axis=1)) - 1
    return np.hstack([np.zeros((n_sims, 1)), cum])


def regime_conditioned_forecast(returns_monthly, weights, dataset,
                                months=60, regime_persistence=12, n_sims=10_000):
    """
    Two-phase forecast:
      Phase 1 (months 1–12):  stats from CURRENT regime
      Phase 2 (months 13–60): full-history stats (mean reversion)
    """
    port = portfolio_return_series(returns_monthly, weights)
    current = dataset["Regime"].iloc[-1]

    mask = dataset["Regime"] == current
    r1 = port.reindex(mask[mask].index).dropna()
    mu1, s1 = r1.mean(), r1.std()
    mu2, s2 = port.mean(), port.std()

    np.random.seed(42)
    p1 = np.random.normal(mu1 - 0.5 * s1**2, s1,
                          size=(n_sims, min(regime_persistence, months)))
    remaining = months - min(regime_persistence, months)
    if remaining > 0:
        p2 = np.random.normal(mu2 - 0.5 * s2**2, s2, size=(n_sims, remaining))
        all_r = np.hstack([p1, p2])
    else:
        all_r = p1

    cum = np.exp(np.cumsum(all_r, axis=1)) - 1
    return np.hstack([np.zeros((n_sims, 1)), cum]), current


def extract_forecast_stats(paths):
    """Percentile bands month-by-month."""
    m = np.arange(paths.shape[1])
    return pd.DataFrame({
        "Month":        m,
        "P5 (Worst)":   np.percentile(paths, 5,  axis=0) * 100,
        "P25":          np.percentile(paths, 25, axis=0) * 100,
        "P50 (Median)": np.percentile(paths, 50, axis=0) * 100,
        "P75":          np.percentile(paths, 75, axis=0) * 100,
        "P95 (Best)":   np.percentile(paths, 95, axis=0) * 100,
        "Mean":         np.mean(paths, axis=0) * 100,
    }).set_index("Month")


# =============================================================================
# VISUALISATIONS (CLI / matplotlib)
# =============================================================================

def plot_forecast(forecast, risk_profile, regime, save_path="forecast_plot.png"):
    """Fan chart: 90% and 50% confidence bands + median + mean."""
    fig, ax = plt.subplots(figsize=(12, 6))
    m = forecast.index

    ax.fill_between(m, forecast["P5 (Worst)"], forecast["P95 (Best)"],
                    alpha=0.15, color="#2E75B6", label="90% confidence band")
    ax.fill_between(m, forecast["P25"], forecast["P75"],
                    alpha=0.30, color="#2E75B6", label="50% confidence band")
    ax.plot(m, forecast["P50 (Median)"], color="#2E75B6", lw=2.5,
            label=f'Median: {forecast.loc[60, "P50 (Median)"]:+.1f}%')
    ax.plot(m, forecast["Mean"], color="#C0392B", lw=1.5, ls="--",
            label=f'Mean: {forecast.loc[60, "Mean"]:+.1f}%')
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    for yr in [12, 24, 36, 48, 60]:
        ax.axvline(yr, color="grey", lw=0.5, ls=":", alpha=0.5)

    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_xticklabels(["Start", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.set_xlabel("Months")
    ax.set_ylabel("Cumulative P/L (%)")
    ax.set_title(f"5-Year Portfolio P/L Forecast\n"
                 f"{risk_profile} investor  ·  {regime} regime  ·  10,000 simulations",
                 fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def plot_portfolio_pie(weights, risk_profile, regime, save_path="portfolio_pie.png"):
    """Pie chart of the final portfolio allocation."""
    labels = [f"{ASSET_TICKERS[t]}\n({t})" for t in weights if weights[t] > 0]
    sizes = [v for v in weights.values() if v > 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=140, pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"Portfolio Allocation\n{risk_profile}  ·  {regime}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def plot_regime_timeline(dataset, save_path="regime_timeline.png"):
    """Colour-coded timeline of historical regimes."""
    from matplotlib.patches import Patch
    colors = {"Expansion": "#27ae60", "Recovery": "#2e86c1",
              "Slowdown": "#f39c12", "Contraction": "#c0392b"}
    fig, ax = plt.subplots(figsize=(14, 3))
    for date, regime in dataset["Regime"].items():
        ax.axvspan(date, date + pd.DateOffset(months=1),
                   color=colors.get(regime, "grey"), alpha=0.7)
    ax.legend(handles=[Patch(color=c, label=r) for r, c in colors.items()],
              loc="upper left", ncol=4)
    ax.set_title("Macro-Regime Timeline", fontweight="bold")
    ax.set_xlim(dataset.index[0], dataset.index[-1])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def forecast_summary_table(forecast, initial_investment=10_000):
    """Year-by-year summary at key percentiles."""
    rows = []
    for m in [12, 24, 36, 48, 60]:
        med = forecast.loc[m, "P50 (Median)"]
        rows.append({
            "Year": m // 12,
            "Worst (5th)":  f'{forecast.loc[m, "P5 (Worst)"]:+.1f}%',
            "Lower (25th)": f'{forecast.loc[m, "P25"]:+.1f}%',
            "Median":       f'{med:+.1f}%',
            "Upper (75th)": f'{forecast.loc[m, "P75"]:+.1f}%',
            "Best (95th)":  f'{forecast.loc[m, "P95 (Best)"]:+.1f}%',
            f"Median $ (${initial_investment:,})":
                f"${initial_investment * (1 + med / 100):,.0f}",
        })
    summary = pd.DataFrame(rows).set_index("Year")
    print("\n=== 5-YEAR FORECAST SUMMARY ===")
    print(summary.to_string())
    return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PORTFOLIO RECOMMENDATION ALGORITHM (MiFID-Based)")
    print("=" * 60)

    dataset, returns_monthly = fetch_data()

    dataset["Regime"] = classify_regime(dataset)
    current_regime = dataset["Regime"].iloc[-1]
    validate_regime(dataset)

    score = run_questionnaire()
    risk_profile = classify_risk_profile(score)
    print(f"Your risk profile: ** {risk_profile} **")

    portfolio = get_portfolio(risk_profile, current_regime)
    print(f"\nRecommended Portfolio ({risk_profile} / {current_regime}):")
    for asset, weight in sorted(portfolio.items(), key=lambda x: -x[1]):
        if weight > 0:
            print(f"  {ASSET_TICKERS[asset]:30s} ({asset}): {weight*100:5.1f}%")

    paths, regime_used = regime_conditioned_forecast(returns_monthly, portfolio, dataset)
    forecast = extract_forecast_stats(paths)
    forecast_summary_table(forecast)

    plot_portfolio_pie(portfolio, risk_profile, current_regime)
    plot_regime_timeline(dataset)
    plot_forecast(forecast, risk_profile, current_regime)

    print("\nDone ✓")
