import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 03_logistic_regression_model.py
# Purpose:
# Estimate a logistic regression model to analyze factors
# associated with household receipt of government benefits
# using ENIGH 2022 household-level microdata.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = BASE_DIR / "outputs" / "model_summary.txt"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# Modeling variables
# ---------------------------------------------------------

target = "received_government_benefits"

model_columns = [
    target,
    "income_per_capita",
    "head_age",
    "household_members",
    "has_elderly_member",
    "has_minor_member",
    "employed_members",
    "food_spending_share",
    "locality_size_label",
    "socioeconomic_stratum_label",
    "head_sex_label"
]

model_df = df[model_columns].copy()

# Remove rows with missing values in model variables
model_df = model_df.dropna()

# Reduce skewness of income using log transformation
model_df["log_income_per_capita"] = np.log1p(model_df["income_per_capita"])

# Keep only variables used directly in the model
model_df = model_df[
    [
        target,
        "log_income_per_capita",
        "head_age",
        "household_members",
        "has_elderly_member",
        "has_minor_member",
        "employed_members",
        "food_spending_share",
        "locality_size_label",
        "socioeconomic_stratum_label",
        "head_sex_label"
    ]
]

# Create dummy variables
X = pd.get_dummies(
    model_df.drop(columns=[target]),
    columns=[
        "locality_size_label",
        "socioeconomic_stratum_label",
        "head_sex_label"
    ],
    drop_first=True
)

y = model_df[target]

# Ensure numeric format
X = X.astype(float)
y = y.astype(int)

# Add intercept
X = sm.add_constant(X)

# ---------------------------------------------------------
# Train-test split for performance evaluation
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Logistic regression model
# ---------------------------------------------------------

logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit(maxiter=200, disp=False)

# Predictions
y_pred_prob = result.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# ---------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

performance = pd.DataFrame([
    ["accuracy", accuracy_score(y_test, y_pred)],
    ["precision", precision_score(y_test, y_pred, zero_division=0)],
    ["recall", recall_score(y_test, y_pred, zero_division=0)],
    ["f1_score", f1_score(y_test, y_pred, zero_division=0)],
    ["roc_auc", roc_auc_score(y_test, y_pred_prob)],
    ["true_negatives", tn],
    ["false_positives", fp],
    ["false_negatives", fn],
    ["true_positives", tp],
    ["modeling_rows", len(model_df)],
    ["train_rows", len(X_train)],
    ["test_rows", len(X_test)]
], columns=["metric", "value"])

performance.to_csv(
    TABLES_DIR / "model_performance.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Odds ratios
# ---------------------------------------------------------

params = result.params
conf = result.conf_int()
pvalues = result.pvalues

odds_ratios = pd.DataFrame({
    "variable": params.index,
    "coefficient": params.values,
    "odds_ratio": np.exp(params.values),
    "conf_low": np.exp(conf[0].values),
    "conf_high": np.exp(conf[1].values),
    "p_value": pvalues.values
})

odds_ratios = odds_ratios[odds_ratios["variable"] != "const"].copy()

odds_ratios["significant_5pct"] = odds_ratios["p_value"] < 0.05

odds_ratios.to_csv(
    TABLES_DIR / "logistic_regression_odds_ratios.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Odds ratio figure
# ---------------------------------------------------------

label_map = {
    "log_income_per_capita": "Log income per capita",
    "head_age": "Household head age",
    "household_members": "Household members",
    "has_elderly_member": "Has elderly member",
    "has_minor_member": "Has minor member",
    "employed_members": "Employed members",
    "food_spending_share": "Food spending share",
    "locality_size_label_15,000 to 99,999 inhabitants": "Locality: 15k-99k inhabitants",
    "locality_size_label_2,500 to 14,999 inhabitants": "Locality: 2.5k-14k inhabitants",
    "locality_size_label_Less than 2,500 inhabitants": "Locality: <2,500 inhabitants",
    "socioeconomic_stratum_label_Low": "Socioeconomic stratum: Low",
    "socioeconomic_stratum_label_Lower-middle": "Socioeconomic stratum: Lower-middle",
    "socioeconomic_stratum_label_Upper-middle": "Socioeconomic stratum: Upper-middle",
    "head_sex_label_Male": "Household head: Male"
}

plot_df = odds_ratios.copy()
plot_df["label"] = plot_df["variable"].map(label_map).fillna(plot_df["variable"])

# Sort by odds ratio
plot_df = plot_df.sort_values("odds_ratio")

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans"
})

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(plot_df))

colors = np.where(plot_df["odds_ratio"] >= 1, "#2E86AB", "#F18F01")

ax.errorbar(
    plot_df["odds_ratio"],
    y_pos,
    xerr=[
        plot_df["odds_ratio"] - plot_df["conf_low"],
        plot_df["conf_high"] - plot_df["odds_ratio"]
    ],
    fmt="none",
    ecolor="#555555",
    alpha=0.8,
    capsize=3
)

ax.scatter(
    plot_df["odds_ratio"],
    y_pos,
    color=colors,
    s=45,
    zorder=3
)

ax.axvline(1, color="#333333", linestyle="--", linewidth=1)

ax.set_xscale("log")
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["label"])
ax.set_xlabel("Odds ratio, log scale")
ax.set_title("Odds ratios from logistic regression model")

# Helpful x-axis ticks
ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "odds_ratios.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Text summary
# ---------------------------------------------------------

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Logistic Regression Model Summary\n")
    f.write("=================================\n\n")
    f.write("Objective:\n")
    f.write(
        "Estimate factors associated with the probability that a household "
        "receives government benefits using ENIGH 2022 household-level microdata.\n\n"
    )

    f.write("Target variable:\n")
    f.write("- received_government_benefits: 1 if government_benefits_income > 0; 0 otherwise.\n\n")

    f.write("Modeling sample:\n")
    f.write(f"- Rows used after dropping missing values: {len(model_df)}\n")
    f.write(f"- Training rows: {len(X_train)}\n")
    f.write(f"- Test rows: {len(X_test)}\n\n")

    f.write("Performance metrics:\n")
    for _, row in performance.iterrows():
        f.write(f"- {row['metric']}: {row['value']}\n")

    f.write("\nInterpretation guide:\n")
    f.write(
        "- Odds ratios above 1 are associated with higher odds of receiving government benefits, "
        "holding the other variables constant.\n"
    )
    f.write(
        "- Odds ratios below 1 are associated with lower odds of receiving government benefits, "
        "holding the other variables constant.\n"
    )
    f.write(
        "- This model estimates statistical associations, not causal effects.\n\n"
    )

    f.write("Top variables by odds ratio:\n")
    top_or = odds_ratios.sort_values("odds_ratio", ascending=False).head(10)
    for _, row in top_or.iterrows():
        f.write(
            f"- {row['variable']}: odds ratio = {row['odds_ratio']:.3f}, "
            f"p-value = {row['p_value']:.4g}\n"
        )

    f.write("\nFull statsmodels summary:\n\n")
    f.write(str(result.summary()))

print("Logistic regression model completed.")
print(f"Rows used in model: {len(model_df)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Model performance saved to: {TABLES_DIR / 'model_performance.csv'}")
print(f"Odds ratios saved to: {TABLES_DIR / 'logistic_regression_odds_ratios.csv'}")
print(f"Model summary saved to: {SUMMARY_PATH}")
print(f"Odds ratio figure saved to: {FIGURES_DIR / 'odds_ratios.png'}")