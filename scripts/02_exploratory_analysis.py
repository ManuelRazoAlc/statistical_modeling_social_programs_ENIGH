import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# ---------------------------------------------------------
# 02_exploratory_analysis.py
# Purpose:
# Generate exploratory tables and figures for the ENIGH 2022
# household-level government benefits modeling dataset.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# Plot style
# ---------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans"
})

BAR_COLOR = "#2E86AB"
SECONDARY_COLOR = "#F18F01"
DARK_GRAY = "#333333"

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")

    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan

    return np.average(values[mask], weights=weights[mask])


def weighted_share(binary_values, weights):
    return weighted_mean(binary_values, weights)


def weighted_group_summary(group_col):
    rows = []

    for category, group in df.groupby(group_col, dropna=False):
        rows.append({
            group_col: category,
            "households_unweighted": len(group),
            "households_weighted": group["expansion_factor"].sum(),
            "share_with_government_benefits_unweighted": group["received_government_benefits"].mean(),
            "share_with_government_benefits_weighted": weighted_share(
                group["received_government_benefits"],
                group["expansion_factor"]
            ),
            "mean_current_income_unweighted": group["current_income"].mean(),
            "mean_current_income_weighted": weighted_mean(
                group["current_income"],
                group["expansion_factor"]
            ),
            "mean_government_benefits_income_among_recipients": group.loc[
                group["received_government_benefits"] == 1,
                "government_benefits_income"
            ].mean()
        })

    return pd.DataFrame(rows)


def add_bar_labels(ax, values, orientation="vertical"):
    for i, value in enumerate(values):
        if pd.isna(value):
            continue

        label = f"{value:.1%}"

        if orientation == "vertical":
            ax.text(
                i,
                value + 0.015,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color=DARK_GRAY
            )
        else:
            ax.text(
                value + 0.01,
                i,
                label,
                ha="left",
                va="center",
                fontsize=10,
                color=DARK_GRAY
            )


# ---------------------------------------------------------
# General descriptive statistics
# ---------------------------------------------------------

numeric_cols = [
    "current_income",
    "income_per_capita",
    "labor_income",
    "transfers_income",
    "government_benefits_income",
    "government_benefits_share",
    "monetary_spending",
    "food_spending_share",
    "head_age",
    "household_members",
    "members_65_plus",
    "minor_members",
    "employed_members"
]

desc = df[numeric_cols].describe().T
desc.to_csv(TABLES_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")

# ---------------------------------------------------------
# Grouped tables
# ---------------------------------------------------------

by_stratum = weighted_group_summary("socioeconomic_stratum_label")
by_locality = weighted_group_summary("locality_size_label")
by_head_sex = weighted_group_summary("head_sex_label")
by_elderly = weighted_group_summary("has_elderly_member")

by_stratum.to_csv(TABLES_DIR / "government_benefits_by_stratum.csv", index=False, encoding="utf-8-sig")
by_locality.to_csv(TABLES_DIR / "government_benefits_by_locality.csv", index=False, encoding="utf-8-sig")
by_head_sex.to_csv(TABLES_DIR / "government_benefits_by_head_sex.csv", index=False, encoding="utf-8-sig")
by_elderly.to_csv(TABLES_DIR / "government_benefits_by_elderly_presence.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Figure 1. Benefits by socioeconomic stratum
# ---------------------------------------------------------

stratum_order = ["Low", "Lower-middle", "Upper-middle", "High"]

plot_stratum = by_stratum.dropna(subset=["socioeconomic_stratum_label"]).copy()
plot_stratum["socioeconomic_stratum_label"] = pd.Categorical(
    plot_stratum["socioeconomic_stratum_label"],
    categories=stratum_order,
    ordered=True
)
plot_stratum = plot_stratum.sort_values("socioeconomic_stratum_label")

fig, ax = plt.subplots(figsize=(9, 5.5))

values = plot_stratum["share_with_government_benefits_weighted"]

ax.bar(
    plot_stratum["socioeconomic_stratum_label"].astype(str),
    values,
    color=BAR_COLOR
)

add_bar_labels(ax, values, orientation="vertical")

ax.set_title("Share of households receiving government benefits by socioeconomic stratum", pad=15)
ax.set_xlabel("Socioeconomic stratum")
ax.set_ylabel("Weighted share of households")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, max(values) + 0.12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "benefits_by_stratum.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2. Benefits by locality size
# ---------------------------------------------------------

locality_order = [
    "100,000 or more inhabitants",
    "15,000 to 99,999 inhabitants",
    "2,500 to 14,999 inhabitants",
    "Less than 2,500 inhabitants"
]

plot_locality = by_locality.dropna(subset=["locality_size_label"]).copy()
plot_locality["locality_size_label"] = pd.Categorical(
    plot_locality["locality_size_label"],
    categories=locality_order,
    ordered=True
)
plot_locality = plot_locality.sort_values(
    "share_with_government_benefits_weighted",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 5.5))

values = plot_locality["share_with_government_benefits_weighted"]

ax.barh(
    plot_locality["locality_size_label"].astype(str),
    values,
    color=BAR_COLOR
)

add_bar_labels(ax, values, orientation="horizontal")

ax.set_title("Share of households receiving government benefits by locality size", pad=15)
ax.set_xlabel("Weighted share of households")
ax.set_ylabel("Locality size")
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlim(0, max(values) + 0.12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "benefits_by_locality.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 3. Income distribution by benefits status
# Normalized density, trimmed at 99th percentile for readability
# ---------------------------------------------------------

plot_df = df[
    ["received_government_benefits", "income_per_capita"]
].copy()

plot_df = plot_df[
    (plot_df["income_per_capita"].notna()) &
    (plot_df["income_per_capita"] > 0)
]

upper_limit = plot_df["income_per_capita"].quantile(0.99)
plot_df = plot_df[plot_df["income_per_capita"] <= upper_limit]

without_benefits = plot_df.loc[
    plot_df["received_government_benefits"] == 0,
    "income_per_capita"
]

with_benefits = plot_df.loc[
    plot_df["received_government_benefits"] == 1,
    "income_per_capita"
]

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.hist(
    without_benefits,
    bins=45,
    density=True,
    alpha=0.55,
    label="Without benefits",
    color=BAR_COLOR
)

ax.hist(
    with_benefits,
    bins=45,
    density=True,
    alpha=0.55,
    label="With benefits",
    color=SECONDARY_COLOR
)

ax.set_title("Income per capita distribution by government benefits status", pad=15)
ax.set_xlabel("Quarterly income per capita")
ax.set_ylabel("Density")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "income_distribution_by_benefits.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 4. Income per capita boxplot by benefits status
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5.5))

box_data = [without_benefits, with_benefits]

ax.boxplot(
    box_data,
    labels=["Without benefits", "With benefits"],
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor=BAR_COLOR, alpha=0.65),
    medianprops=dict(color="black", linewidth=1.5)
)

ax.set_title("Income per capita by government benefits status", pad=15)
ax.set_xlabel("Government benefits status")
ax.set_ylabel("Quarterly income per capita")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "income_boxplot_by_benefits.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------

overall_weighted_share = weighted_share(
    df["received_government_benefits"],
    df["expansion_factor"]
)

print("Exploratory analysis completed.")
print(f"Rows analyzed: {len(df)}")
print(f"Unweighted share with government benefits: {df['received_government_benefits'].mean():.4f}")
print(f"Weighted share with government benefits: {overall_weighted_share:.4f}")
print(f"Tables saved to: {TABLES_DIR}")
print(f"Figures saved to: {FIGURES_DIR}")