
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "outputs"
LOCKDOWN_START = pd.Timestamp("2020-03-01")  # India's national lockdown began 25 Mar 2020
LOCKDOWN_END = pd.Timestamp("2020-06-30")

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ----------------------------------------------------------------------------
def standardize_columns(df):
    """Rename messy column names to short, consistent ones."""
    mapping = {}
    for col in df.columns:
        c = col.strip().lower()
        if "unemployment" in c:
            mapping[col] = "unemployment_rate"
        elif "employed" in c:
            mapping[col] = "employed"
        elif "participation" in c:
            mapping[col] = "labour_participation"
        elif c == "region":
            mapping[col] = "region"
        elif c == "region.1":
            mapping[col] = "zone"
        elif c == "area":
            mapping[col] = "area"
        elif c == "date":
            mapping[col] = "date"
        elif c == "frequency":
            mapping[col] = "frequency"
    return df.rename(columns=mapping)


def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = standardize_columns(df)

    required = {"region", "date", "unemployment_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Could not find required columns: {missing}. "
                         f"Columns found: {list(df.columns)}")

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    print(f"Raw shape: {df.shape}")
    print("\nMissing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "None")

    # Strip stray whitespace in text columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Parse dates (the source file uses day-first dates with leading spaces)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Drop rows with no usable date or rate, and exact duplicates
    before = len(df)
    df = df.dropna(subset=["date", "unemployment_rate"])
    df = df.drop_duplicates()
    print(f"\nRows removed (missing/duplicate): {before - len(df)}")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    # Label each row with its Covid phase
    def phase(d):
        if d < LOCKDOWN_START:
            return "Pre-Covid"
        if d <= LOCKDOWN_END:
            return "Lockdown (Mar-Jun 2020)"
        return "Post-lockdown"

    df["phase"] = df["date"].apply(phase)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Clean shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Regions: {df['region'].nunique()}")
    return df


# ----------------------------------------------------------------------------
# 2. EXPLORE
# ----------------------------------------------------------------------------
def explore(df):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    num_cols = [c for c in ["unemployment_rate", "employed", "labour_participation"]
                if c in df.columns]
    print(df[num_cols].describe().round(2))

    if "labour_participation" in df.columns:
        print("\nCorrelation matrix:")
        print(df[num_cols].corr().round(2))


# ----------------------------------------------------------------------------
# 3. VISUALISE
# ----------------------------------------------------------------------------
def plot_national_trend(df):
    """Average unemployment per month with the lockdown period shaded."""
    monthly = df.groupby("date")["unemployment_rate"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#C44E52", linewidth=2)
    ax.axvspan(LOCKDOWN_START, LOCKDOWN_END, color="grey", alpha=0.2,
               label="Covid lockdown")
    ax.set_title("Average Unemployment Rate Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "national_trend.png"), dpi=150)
    plt.close(fig)
    return monthly


def plot_phase_comparison(df):
    """Average unemployment by Covid phase."""
    order = [p for p in ["Pre-Covid", "Lockdown (Mar-Jun 2020)", "Post-lockdown"]
             if p in df["phase"].unique()]
    summary = df.groupby("phase")["unemployment_rate"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=summary.index, y=summary.values, hue=summary.index, legend=False,
                ax=ax, palette=["#4C72B0", "#C44E52", "#55A868"][:len(order)])
    for i, v in enumerate(summary.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Average Unemployment Rate by Covid Phase")
    ax.set_xlabel("")
    ax.set_ylabel("Unemployment rate (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase_comparison.png"), dpi=150)
    plt.close(fig)
    return summary


def plot_region_impact(df):
    """Regions ranked by increase in unemployment: lockdown vs pre-Covid."""
    pre = df[df["phase"] == "Pre-Covid"].groupby("region")["unemployment_rate"].mean()
    lock = df[df["phase"] == "Lockdown (Mar-Jun 2020)"].groupby("region")["unemployment_rate"].mean()
    impact = (lock - pre).dropna().sort_values(ascending=False)
    if impact.empty:
        return impact

    top = impact.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", color="#C44E52", ax=ax)
    ax.set_title("Top 10 Regions by Rise in Unemployment\n(Lockdown avg minus Pre-Covid avg)")
    ax.set_xlabel("Percentage-point increase")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "region_impact.png"), dpi=150)
    plt.close(fig)
    return impact


def plot_region_heatmap(df):
    pivot = df.pivot_table(index="region", columns="date",
                           values="unemployment_rate", aggfunc="mean")
    pivot.columns = [c.strftime("%b %y") for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Unemployment rate (%)"})
    ax.set_title("Unemployment Rate by Region and Month")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "region_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_group_boxplot(df):
    """Distribution by zone or by urban/rural area, if those columns exist."""
    for col, title in [("zone", "Zone"), ("area", "Area (Urban/Rural)")]:
        if col in df.columns and df[col].nunique() > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x=col, y="unemployment_rate", ax=ax)
            ax.set_title(f"Unemployment Rate Distribution by {title}")
            ax.set_xlabel("")
            ax.set_ylabel("Unemployment rate (%)")
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{col}.png"), dpi=150)
            plt.close(fig)


def plot_seasonality(df):
    """Month-of-year profile, one line per year."""
    seasonal = (df.groupby(["year", "month"])["unemployment_rate"]
                  .mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=seasonal, x="month", y="unemployment_rate", hue="year",
                 marker="o", palette="Set1", ax=ax)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Unemployment by Month of Year")
    ax.set_xlabel("")
    ax.set_ylabel("Average unemployment rate (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "seasonality.png"), dpi=150)
    plt.close(fig)


def plot_participation(df):
    if "labour_participation" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="labour_participation", y="unemployment_rate",
                    hue="phase", alpha=0.7, ax=ax)
    ax.set_title("Labour Participation vs Unemployment Rate")
    ax.set_xlabel("Labour participation rate (%)")
    ax.set_ylabel("Unemployment rate (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "participation_vs_unemployment.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. INSIGHTS
# ----------------------------------------------------------------------------
def write_insights(df, monthly, phase_summary, impact):
    lines = ["KEY INSIGHTS", "=" * 60]

    peak_date = monthly.idxmax()
    lines.append(f"- Peak average unemployment: {monthly.max():.1f}% in "
                 f"{peak_date.strftime('%B %Y')}.")

    if "Pre-Covid" in phase_summary.index and "Lockdown (Mar-Jun 2020)" in phase_summary.index:
        pre = phase_summary["Pre-Covid"]
        lock = phase_summary["Lockdown (Mar-Jun 2020)"]
        lines.append(f"- Average unemployment rose from {pre:.1f}% before Covid to "
                     f"{lock:.1f}% during the lockdown (+{lock - pre:.1f} points, "
                     f"about {lock / pre:.1f}x).")
    if "Post-lockdown" in phase_summary.index and "Lockdown (Mar-Jun 2020)" in phase_summary.index:
        post = phase_summary["Post-lockdown"]
        lines.append(f"- After the lockdown the average fell to {post:.1f}%, "
                     f"showing a recovery.")

    if not impact.empty:
        worst = impact.head(3)
        lines.append("- Hardest-hit regions (rise in points): " +
                     ", ".join(f"{r} (+{v:.1f})" for r, v in worst.items()) + ".")
        best = impact.tail(3)
        lines.append("- Least affected regions: " +
                     ", ".join(f"{r} (+{v:.1f})" for r, v in best.items()) + ".")

    if "area" in df.columns and df["area"].nunique() > 1:
        by_area = df.groupby("area")["unemployment_rate"].mean().round(1)
        lines.append("- Average unemployment by area: " +
                     ", ".join(f"{k} {v}%" for k, v in by_area.items()) + ".")
    if "zone" in df.columns and df["zone"].nunique() > 1:
        by_zone = df.groupby("zone")["unemployment_rate"].mean().sort_values(ascending=False).round(1)
        lines.append("- Zone with highest average unemployment: "
                     f"{by_zone.index[0]} ({by_zone.iloc[0]}%).")

    n_years = df["year"].nunique()
    lines.append(f"- Seasonality caveat: the data covers only about {n_years} calendar "
                 "years, so genuine seasonal patterns cannot be separated from the "
                 "Covid shock. Treat the month-of-year chart as indicative only.")

    lines += ["", "POLICY IMPLICATIONS",
              "- Targeted support (cash transfers, job guarantees, skilling) should go "
              "first to the hardest-hit regions.",
              "- Regions with large, sudden spikes point to heavy dependence on "
              "informal or contact-intensive work; diversifying local employment "
              "would improve resilience.",
              "- Tracking labour participation alongside unemployment matters: a "
              "falling participation rate can hide people who have stopped looking "
              "for work."]

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "insights.txt"), "w") as f:
        f.write(text)


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the unemployment CSV")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean(args.csv)
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
    explore(df)

    monthly = plot_national_trend(df)
    phase_summary = plot_phase_comparison(df)
    impact = plot_region_impact(df)
    plot_region_heatmap(df)
    plot_group_boxplot(df)
    plot_seasonality(df)
    plot_participation(df)

    write_insights(df, monthly, phase_summary, impact)
    print(f"\nAll charts, cleaned data and insights saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
