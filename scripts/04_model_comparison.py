import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# 04_model_comparison.py
# Purpose:
# Compare multiple logistic regression specifications for
# modeling household receipt of government benefits using
# ENIGH 2022 household-level microdata.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = BASE_DIR / "outputs" / "model_comparison_summary.txt"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "received_government_benefits"

# ---------------------------------------------------------
# Feature engineering used across model specifications
# ---------------------------------------------------------

df["log_income_per_capita"] = np.log1p(df["income_per_capita"])

# ---------------------------------------------------------
# Model specifications
# ---------------------------------------------------------

model_specs = {
    "Minimal model": {
        "numeric": [
            "log_income_per_capita",
            "household_members",
            "has_elderly_member"
        ],
        "categorical": [
            "locality_size_label",
            "socioeconomic_stratum_label"
        ],
        "description": (
            "Uses five core conceptual predictors: income per capita, household size, "
            "elderly presence, locality size and socioeconomic stratum."
        )
    },

    "Base model": {
        "numeric": [
            "log_income_per_capita",
            "head_age",
            "household_members",
            "has_elderly_member",
            "has_minor_member",
            "employed_members",
            "food_spending_share"
        ],
        "categorical": [
            "locality_size_label",
            "socioeconomic_stratum_label",
            "head_sex_label"
        ],
        "description": (
            "Includes demographic, household composition, income, spending, "
            "territorial and socioeconomic variables."
        )
    },

    "Economic-composition model": {
        "numeric": [
            "log_income_per_capita",
            "head_age",
            "household_members",
            "has_elderly_member",
            "has_minor_member",
            "employed_members",
            "food_spending_share",
            "labor_income_share"
        ],
        "categorical": [
            "locality_size_label",
            "socioeconomic_stratum_label",
            "head_sex_label"
        ],
        "description": (
            "Adds labor income share to the base model to test whether household "
            "income composition improves classification performance."
        )
    }
}

# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def fit_and_evaluate_model(model_name, spec):
    required_columns = [target] + spec["numeric"] + spec["categorical"]

    model_df = df[required_columns].copy()
    model_df = model_df.dropna()

    X_raw = model_df.drop(columns=[target])
    y = model_df[target].astype(int)

    X = pd.get_dummies(
        X_raw,
        columns=spec["categorical"],
        drop_first=True
    )

    X = X.astype(float)
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(maxiter=200, disp=False)

    y_pred_prob = result.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model_name": model_name,
        "description": spec["description"],
        "rows_used": len(model_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "number_of_conceptual_predictors": len(spec["numeric"]) + len(spec["categorical"]),
        "number_of_predictors_after_dummies": X.shape[1] - 1,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "aic": result.aic,
        "bic": result.bic
    }


# ---------------------------------------------------------
# Run comparison
# ---------------------------------------------------------

results = []

for model_name, spec in model_specs.items():
    print(f"Fitting {model_name}...")
    model_result = fit_and_evaluate_model(model_name, spec)
    results.append(model_result)

comparison = pd.DataFrame(results)

comparison = comparison[
    [
        "model_name",
        "description",
        "rows_used",
        "train_rows",
        "test_rows",
        "number_of_conceptual_predictors",
        "number_of_predictors_after_dummies",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "aic",
        "bic",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives"
    ]
]

comparison.to_csv(
    TABLES_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Plot 1: Model comparison by ROC AUC
# ---------------------------------------------------------

plot_df = comparison.sort_values("roc_auc", ascending=True).copy()

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans"
})

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    plot_df["model_name"],
    plot_df["roc_auc"],
    color="#2E86AB"
)

for i, value in enumerate(plot_df["roc_auc"]):
    ax.text(
        value + 0.005,
        i,
        f"{value:.3f}",
        va="center",
        fontsize=10,
        color="#333333"
    )

ax.set_title("ROC AUC comparison across logistic regression specifications")
ax.set_xlabel("ROC AUC")
ax.set_ylabel("Model")
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_roc_auc.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Main metrics comparison
# ---------------------------------------------------------

metrics_long = comparison.melt(
    id_vars=["model_name"],
    value_vars=["accuracy", "f1_score", "roc_auc"],
    var_name="metric",
    value_name="value"
)

metric_order = ["accuracy", "f1_score", "roc_auc"]
model_order = comparison["model_name"].tolist()

x = np.arange(len(model_order))
width = 0.24

fig, ax = plt.subplots(figsize=(11, 6))

colors = {
    "accuracy": "#2E86AB",
    "f1_score": "#F18F01",
    "roc_auc": "#6A994E"
}

for i, metric in enumerate(metric_order):
    metric_values = (
        metrics_long[metrics_long["metric"] == metric]
        .set_index("model_name")
        .loc[model_order, "value"]
    )

    offset = (i - 1) * width

    ax.bar(
        x + offset,
        metric_values,
        width,
        label=metric.replace("_", " ").title(),
        color=colors[metric]
    )

    for j, value in enumerate(metric_values):
        ax.text(
            x[j] + offset,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333"
        )

ax.set_title("Main classification metrics by model specification")
ax.set_xlabel("Model")
ax.set_ylabel("Metric value")
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=15, ha="right")
ax.set_ylim(0, 1.08)

ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3
)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_main_metrics.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Text summary
# ---------------------------------------------------------

best_auc_model = comparison.sort_values("roc_auc", ascending=False).iloc[0]
best_bic_model = comparison.sort_values("bic", ascending=True).iloc[0]

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Model Comparison Summary\n")
    f.write("========================\n\n")

    f.write("Objective:\n")
    f.write(
        "Compare multiple logistic regression specifications for predicting "
        "household receipt of government benefits using ENIGH 2022 microdata.\n\n"
    )

    f.write("Models compared:\n")
    for model_name, spec in model_specs.items():
        f.write(f"- {model_name}: {spec['description']}\n")

    f.write("\nMain comparison metrics:\n")
    for _, row in comparison.iterrows():
        f.write(
            f"- {row['model_name']}: "
            f"ROC AUC = {row['roc_auc']:.4f}, "
            f"Accuracy = {row['accuracy']:.4f}, "
            f"Precision = {row['precision']:.4f}, "
            f"Recall = {row['recall']:.4f}, "
            f"F1 = {row['f1_score']:.4f}, "
            f"Conceptual predictors = {int(row['number_of_conceptual_predictors'])}, "
            f"Predictors after dummies = {int(row['number_of_predictors_after_dummies'])}, "
            f"BIC = {row['bic']:.2f}\n"
        )

    f.write("\nBest model by ROC AUC:\n")
    f.write(
        f"- {best_auc_model['model_name']} "
        f"(ROC AUC = {best_auc_model['roc_auc']:.4f})\n"
    )

    f.write("\nBest model by BIC:\n")
    f.write(
        f"- {best_bic_model['model_name']} "
        f"(BIC = {best_bic_model['bic']:.2f})\n"
    )

    f.write("\nInterpretation note:\n")
    f.write(
        "The preferred model should not be selected only by predictive performance. "
        "For applied social analysis, interpretability and methodological coherence "
        "are also important. A simpler model may be preferable if it performs similarly "
        "to a more complex specification.\n"
    )

print("Model comparison completed.")
print("\nComparison table:")
print(
    comparison[
        [
            "model_name",
            "number_of_conceptual_predictors",
            "number_of_predictors_after_dummies",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "bic"
        ]
    ].to_string(index=False)
)
print(f"\nModel comparison saved to: {TABLES_DIR / 'model_comparison.csv'}")
print(f"Summary saved to: {SUMMARY_PATH}")
print(f"ROC AUC figure saved to: {FIGURES_DIR / 'model_comparison_roc_auc.png'}")
print(f"Main metrics figure saved to: {FIGURES_DIR / 'model_comparison_main_metrics.png'}")