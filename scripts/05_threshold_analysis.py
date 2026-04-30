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
    confusion_matrix
)

# ---------------------------------------------------------
# 05_threshold_analysis.py
# Purpose:
# Analyze how different classification thresholds affect
# false positives, false negatives, precision, recall and F1
# for the selected logistic regression base model.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = BASE_DIR / "outputs" / "threshold_analysis_summary.txt"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "received_government_benefits"

# ---------------------------------------------------------
# Base model specification
# ---------------------------------------------------------

df["log_income_per_capita"] = np.log1p(df["income_per_capita"])

model_columns = [
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

model_df = df[model_columns].dropna().copy()

X_raw = model_df.drop(columns=[target])
y = model_df[target].astype(int)

X = pd.get_dummies(
    X_raw,
    columns=[
        "locality_size_label",
        "socioeconomic_stratum_label",
        "head_sex_label"
    ],
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

model = sm.Logit(y_train, X_train)
result = model.fit(maxiter=200, disp=False)

y_pred_prob = result.predict(X_test)

# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

thresholds = np.arange(0.20, 0.81, 0.05)

rows = []

for threshold in thresholds:
    y_pred = (y_pred_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "predicted_positive_rate": y_pred.mean()
    })

threshold_df = pd.DataFrame(rows)

threshold_df.to_csv(
    TABLES_DIR / "threshold_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

# Select useful thresholds
best_f1 = threshold_df.sort_values("f1_score", ascending=False).iloc[0]
best_accuracy = threshold_df.sort_values("accuracy", ascending=False).iloc[0]

# ---------------------------------------------------------
# Figure 1: Precision, recall and F1 by threshold
# ---------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 6),
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

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    marker="o",
    label="Precision"
)

ax.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    marker="o",
    label="Recall"
)

ax.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    marker="o",
    label="F1 score"
)

ax.axvline(
    0.50,
    linestyle="--",
    linewidth=1,
    color="#333333",
    label="Default threshold = 0.50"
)

ax.axvline(
    best_f1["threshold"],
    linestyle=":",
    linewidth=1.5,
    color="#666666",
    label=f"Best F1 threshold = {best_f1['threshold']:.2f}"
)

ax.set_title("Precision, recall and F1 score by classification threshold")
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "threshold_metrics.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2: Confusion matrix at threshold 0.50
# ---------------------------------------------------------

threshold_050 = 0.50
y_pred_050 = (y_pred_prob >= threshold_050).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_050).ravel()

conf_matrix = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(conf_matrix)

ax.set_title("Confusion matrix at threshold 0.50")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")

ax.set_xticks([0, 1])
ax.set_xticklabels(["No benefits", "Benefits"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["No benefits", "Benefits"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{conf_matrix[i, j]:,}",
            ha="center",
            va="center",
            fontsize=13,
            color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
        )

plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix_threshold_050.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Text summary
# ---------------------------------------------------------

row_050 = threshold_df.loc[
    np.isclose(threshold_df["threshold"], 0.50)
].iloc[0]

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Threshold Analysis Summary\n")
    f.write("==========================\n\n")

    f.write("Objective:\n")
    f.write(
        "Evaluate how different classification thresholds affect precision, recall, "
        "F1 score, false positives and false negatives for the selected base logistic "
        "regression model.\n\n"
    )

    f.write("Default threshold 0.50:\n")
    f.write(f"- Accuracy: {row_050['accuracy']:.4f}\n")
    f.write(f"- Precision: {row_050['precision']:.4f}\n")
    f.write(f"- Recall: {row_050['recall']:.4f}\n")
    f.write(f"- F1 score: {row_050['f1_score']:.4f}\n")
    f.write(f"- False positives: {int(row_050['false_positives'])}\n")
    f.write(f"- False negatives: {int(row_050['false_negatives'])}\n")
    f.write(f"- False positive rate: {row_050['false_positive_rate']:.4f}\n")
    f.write(f"- False negative rate: {row_050['false_negative_rate']:.4f}\n\n")

    f.write("Best threshold by F1 score:\n")
    f.write(f"- Threshold: {best_f1['threshold']:.2f}\n")
    f.write(f"- Accuracy: {best_f1['accuracy']:.4f}\n")
    f.write(f"- Precision: {best_f1['precision']:.4f}\n")
    f.write(f"- Recall: {best_f1['recall']:.4f}\n")
    f.write(f"- F1 score: {best_f1['f1_score']:.4f}\n")
    f.write(f"- False positives: {int(best_f1['false_positives'])}\n")
    f.write(f"- False negatives: {int(best_f1['false_negatives'])}\n\n")

    f.write("Best threshold by accuracy:\n")
    f.write(f"- Threshold: {best_accuracy['threshold']:.2f}\n")
    f.write(f"- Accuracy: {best_accuracy['accuracy']:.4f}\n")
    f.write(f"- Precision: {best_accuracy['precision']:.4f}\n")
    f.write(f"- Recall: {best_accuracy['recall']:.4f}\n")
    f.write(f"- F1 score: {best_accuracy['f1_score']:.4f}\n")
    f.write(f"- False positives: {int(best_accuracy['false_positives'])}\n")
    f.write(f"- False negatives: {int(best_accuracy['false_negatives'])}\n\n")

    f.write("Interpretation note:\n")
    f.write(
        "Lower thresholds classify more households as benefit recipients, which tends "
        "to increase recall and reduce false negatives, but also increases false positives. "
        "Higher thresholds classify fewer households as recipients, which tends to increase "
        "precision but may exclude more true recipient households.\n"
    )

print("Threshold analysis completed.")
print(f"Rows in test set: {len(y_test)}")
print("\nDefault threshold 0.50:")
print(row_050[[
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positives",
    "false_negatives",
    "false_positive_rate",
    "false_negative_rate"
]].to_string())

print("\nBest threshold by F1 score:")
print(best_f1[[
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positives",
    "false_negatives"
]].to_string())

print("\nBest threshold by accuracy:")
print(best_accuracy[[
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positives",
    "false_negatives"
]].to_string())

print(f"\nThreshold table saved to: {TABLES_DIR / 'threshold_analysis.csv'}")
print(f"Threshold figure saved to: {FIGURES_DIR / 'threshold_metrics.png'}")
print(f"Confusion matrix saved to: {FIGURES_DIR / 'confusion_matrix_threshold_050.png'}")
print(f"Summary saved to: {SUMMARY_PATH}")