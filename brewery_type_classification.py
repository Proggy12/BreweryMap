"""
Brewery Type Classification from Location Features
===================================================

Research question:
    Can the brewery type (micro, brewpub, regional, ...) be predicted
    from location features (country, state/province, lat/lon)?

Pipeline:
    1. Load data, drop rows with missing coordinates or type
    2. Merge rare classes (< MIN_CLASS_SIZE samples) into 'other' to
       avoid stratification issues in cross-validation
    3. Features: latitude, longitude, country, state_province
       Categorical columns are one-hot encoded
    4. Stratified train/test split (80/20)
    5. Four classifiers are compared:
         - Logistic Regression   (linear baseline)
         - Random Forest         (tree ensemble baseline)
         - XGBoost               (gradient boosting)
         - LightGBM              (fast gradient boosting)
       Each is tuned with RandomizedSearchCV (faster than GridSearchCV
       for large hyperparameter spaces; covers the same space with fewer
       fits by sampling randomly)
    6. Best models are evaluated on the held-out test set:
         - Accuracy, Macro-F1
         - Per-class Classification Report (CSV)
         - Confusion Matrix (PNG)
         - Feature Importances for tree-based models (PNG + CSV)
    7. A final summary table compares all models side-by-side

Why these four models?
    - Logistic Regression: fast, interpretable, good sanity check
    - Random Forest: handles non-linear boundaries, robust to outliers,
      naturally provides feature importances
    - XGBoost: sequential boosting corrects errors of previous trees,
      often outperforms RF on tabular data
    - LightGBM: same idea as XGBoost but uses histogram-based splits,
      much faster on high-dimensional OHE features (many countries /
      states produce hundreds of binary columns)

Installation:
    pip install scikit-learn xgboost lightgbm matplotlib seaborn
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")  # suppress verbose sklearn/lgbm warnings

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV = "breweries_clean.csv"
OUTPUT_DIR = Path("classification_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classes with fewer than this many samples are merged into 'other'.
# This prevents cross-validation folds from containing only 1 sample
# of a class, which would break stratified splitting.
MIN_CLASS_SIZE = 50

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Number of hyperparameter combinations tried per model in RandomizedSearchCV.
# Higher = better chance of finding optimum, but slower.
N_ITER_SEARCH = 20
CV_FOLDS = 5  # cross-validation folds during grid search

# ── Load & prepare data ───────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV, dtype={"postal_code": str})

# Drop rows where any feature or the target is missing
df = df.dropna(
    subset=["latitude", "longitude", "brewery_type", "country", "state_province"]
).copy()

print(f"Rows with complete features: {len(df)}")

# Merge rare classes into 'other' to keep class distribution manageable
class_counts = df.loc[:, "brewery_type"].value_counts()
rare_classes = class_counts.loc[class_counts < MIN_CLASS_SIZE].index.tolist()

df.loc[:, "brewery_type_grouped"] = df.loc[:, "brewery_type"].replace(
    {c: "other" for c in rare_classes}
)

print(f"\nOriginal class distribution:\n{class_counts.to_string()}")
print(f"\nClasses merged into 'other' (< {MIN_CLASS_SIZE} samples): {rare_classes}")
print(
    f"\nFinal class distribution:\n{df.loc[:, 'brewery_type_grouped'].value_counts().to_string()}"
)

# ── Features & target ─────────────────────────────────────────────────────────

numeric_features = ["latitude", "longitude"]
categorical_features = ["country", "state_province"]

X = df.loc[:, numeric_features + categorical_features]
y = df.loc[:, "brewery_type_grouped"]

# Encode string labels to integers (required by XGBoost and LightGBM)
label_encoder = LabelEncoder()
y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
class_names = label_encoder.classes_

# Stratified split ensures each class is proportionally represented in
# both train and test sets — important given the large class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Preprocessing ─────────────────────────────────────────────────────────────

# One-hot encode categorical columns; unknown categories at test time
# (e.g. a country seen only in test) are silently ignored.
# Numeric features pass through unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            categorical_features,
        ),
    ]
)

# ── Model definitions & hyperparameter search spaces ─────────────────────────

# Each entry: (display_name, estimator, param_grid)
# Param keys must be prefixed with "classifier__" because the estimator
# lives inside a Pipeline step named "classifier".

models = [
    (
        "Logistic Regression",
        LogisticRegression(
            class_weight="balanced",  # compensates for class imbalance
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        {
            # C: inverse regularization strength; smaller = stronger penalty
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["lbfgs", "saga"],
        },
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    ),
    (
        "XGBoost",
        XGBClassifier(
            eval_metric="mlogloss",  # multi-class log loss
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            # subsample & colsample reduce overfitting by using random
            # subsets of rows / columns per tree
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        },
    ),
    (
        "LightGBM",
        LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,  # suppress LightGBM's own logging
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [-1, 10, 20],  # -1 = unlimited
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__num_leaves": [31, 63, 127],
            # min_child_samples: prevents leaves with very few samples
            "classifier__min_child_samples": [20, 50, 100],
        },
    ),
]

# ── Train, tune & evaluate ────────────────────────────────────────────────────

summary_rows = []  # collects per-model metrics for the final table
best_pipelines = {}  # stores the best fitted pipeline per model name

for name, estimator, param_grid in models:
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", estimator),
        ]
    )

    # RandomizedSearchCV samples N_ITER_SEARCH combinations at random.
    # Scoring on macro-F1 is better than accuracy here because we have
    # a strong class imbalance — accuracy would be dominated by the two
    # large classes (micro, brewpub).
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring="f1_macro",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,  # refit best model on full training set
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_pipelines[name] = search.best_estimator_

    print(f"Best CV Macro-F1 : {search.best_score_:.3f}")
    print(f"Best params      : {search.best_params_}")
    print(f"Time elapsed     : {elapsed:.1f}s")

    # Evaluate on held-out test set
    y_pred = search.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTest Accuracy : {acc:.3f}")
    print(f"Test Macro-F1 : {macro_f1:.3f}")

    # Decode integer predictions back to readable class names
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Per-class report → CSV
    report_dict = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=class_names,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    report_path = OUTPUT_DIR / f"report_{name.replace(' ', '_').lower()}.csv"
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix → PNG
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    summary_rows.append(
        {
            "model": name,
            "cv_macro_f1": round(search.best_score_, 3),
            "test_accuracy": round(acc, 3),
            "test_macro_f1": round(macro_f1, 3),
            "best_params": str(search.best_params_),
            "fit_time_s": round(elapsed, 1),
        }
    )

# ── Feature importances (tree-based models only) ──────────────────────────────

tree_models = ["Random Forest", "XGBoost", "LightGBM"]
TOP_N_FEATURES = 15

for name in tree_models:
    pipeline = best_pipelines[name]
    clf = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = clf.feature_importances_

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    imp_path = OUTPUT_DIR / f"feature_importances_{name.replace(' ', '_').lower()}.csv"
    imp_df.to_csv(imp_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp_df.head(TOP_N_FEATURES).set_index("feature")["importance"].plot(
        kind="barh", ax=ax, color="#4c72b0"
    )
    ax.invert_yaxis()
    ax.set_title(f"Top {TOP_N_FEATURES} Feature Importances — {name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = OUTPUT_DIR / f"feature_importances_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"\nFeature importances ({name}) saved to {fi_path}")

# ── Summary table ─────────────────────────────────────────────────────────────

summary_df = pd.DataFrame(summary_rows).sort_values("test_macro_f1", ascending=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(
    summary_df.loc[
        :, ["model", "cv_macro_f1", "test_accuracy", "test_macro_f1", "fit_time_s"]
    ].to_string(index=False)
)

summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved to {summary_path}")

# Visual summary: grouped bar chart comparing Accuracy and Macro-F1
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(summary_df))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    summary_df.loc[:, "test_accuracy"],
    width,
    label="Test Accuracy",
    color="#4c72b0",
)
bars2 = ax.bar(
    x + width / 2,
    summary_df.loc[:, "test_macro_f1"],
    width,
    label="Test Macro-F1",
    color="#c44e52",
)

ax.set_xticks(x)
ax.set_xticklabels(summary_df.loc[:, "model"], rotation=15, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Accuracy vs. Macro-F1")
ax.legend()
ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
plt.tight_layout()
summary_plot_path = OUTPUT_DIR / "model_comparison_summary.png"
plt.savefig(summary_plot_path, dpi=150)
plt.close()
print(f"Summary plot saved to {summary_plot_path}")

print("\nDone.")
