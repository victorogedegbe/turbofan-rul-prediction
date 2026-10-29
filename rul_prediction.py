# ==============================================================================
# Remaining Useful Life (RUL) Prediction - NASA C-MAPSS FD001
# MSc Data Science Dissertation - DS7010
# ==============================================================================
#
# Dependencies: see requirements.txt (install with `pip install -r requirements.txt`
# before running this script).

import os
import time
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

import tensorflow as tf

# Force CPU-only execution. On Apple Silicon, TensorFlow's Metal GPU backend
# does not implement fully deterministic kernels for LSTM/RNN layers, even
# with enable_op_determinism() below — this is why LSTM results kept
# changing between runs while the scikit-learn models (which run on CPU)
# stayed identical. Running on CPU trades a little speed for exact,
# reportable reproducibility, which matters far more for a dissertation.
tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Global settings (used across multiple steps) ----------------------------
SEED = 42         # Reproducibility seed for all random operations
DATA_DIR = "Data"        # Folder containing the NASA C-MAPSS CSV files
OUTPUT_DIR = "Outputs"   # Root folder for all generated outputs
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
# Reproducibility: fix every source of randomness, including TensorFlow's
# internal operation kernels. tf.random.set_seed() alone is NOT enough for
# the LSTM to reproduce exactly between runs — without enable_op_determinism(),
# parallel execution can still consume random numbers in a different order
# each run, which is why LSTM results (unlike RF/SVR) previously varied.

# Create output folders
for folder in ["Figures", "Tables", "Models"]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


# ==============================================================================
# STEP 1: DATA LOADING
# ==============================================================================
print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)

# The data files were labelled with column headers in Excel before loading,
# so the column names are already present in the first row of each CSV.
# encoding='utf-8-sig' removes the byte-order mark (BOM) that Excel adds.
train_data = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.csv"),
                         encoding="utf-8-sig")

test_data = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.csv"),
                        encoding="utf-8-sig")

rul_data = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.csv"),
                       encoding="utf-8-sig")

# Inspect raw data
print("\n--- Training data: first 5 rows ---")
print(train_data.head())
print("\n--- Training data: last 5 rows ---")
print(train_data.tail())
print(f"\nTraining data shape: {train_data.shape}")
print(f"\nData types:\n{train_data.dtypes}")
print(f"\n--- Data info ---")
train_data.info()
print(f"\n--- Test data shape: {test_data.shape} ---")
print("--- RUL data shape:", rul_data.shape, "---")


# ==============================================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# --- 2a. Data quality checks -------------------------------------------------
print("\n--- Data Quality Checks ---")
print(f"Training engines: {train_data['engine_id'].nunique()}")
print(f"Test engines:     {test_data['engine_id'].nunique()}")
print(f"RUL entries:      {len(rul_data)}")
print(f"Missing values (train): {train_data.isnull().sum().sum()}")
print(f"Missing values (test):  {test_data.isnull().sum().sum()}")
print(f"Duplicate rows (train): {train_data.duplicated().sum()}")
print(f"Duplicate rows (test):  {test_data.duplicated().sum()}")

# Missingness map — visual evidence of where (if anywhere) values are missing,
# in addition to the raw counts above. Built directly with seaborn (already
# imported for every other plot in this script) rather than a separate
# missing-data library, so no extra install is required to reproduce this
# pipeline. Each row is one record, each column one variable; a filled/solid
# colour means present, a contrasting colour would mark a missing cell.
def plot_missingness(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False,
                cmap=["#2c3e50", "#e74c3c"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Variable")
    fig.savefig(os.path.join(OUTPUT_DIR, "Figures", filename),
                dpi=300, bbox_inches="tight")
    plt.close()


plot_missingness(train_data, "Missingness Map — Training Data",
                  "missingness_map_train.png")
plot_missingness(test_data, "Missingness Map — Test Data",
                  "missingness_map_test.png")
print("Missingness maps saved.")

# --- 2b. Remove near-constant predictors -------------------------------------
# Sensors and operational settings are checked differently, because "no
# predictive information" means something different for each:
#
# - Sensors: physical measurements. A handful of FD001 sensors (documented
#   throughout the C-MAPSS literature, e.g. Saxena et al., 2008) have genuine
#   but extremely small physical variance under a single operating condition
#   -- they never take literally one exact value, so a distinct-value check
#   would not catch them, but their standard deviation is comfortably below
#   any level that could carry useful signal. These are removed using the
#   standard std < 0.01 threshold used throughout the literature.
#
# - Settings: setting_3 is a true structural constant (always exactly 100 in
#   FD001), which a distinct-value check catches directly and correctly.
#   setting_1 and setting_2, by contrast, have real row-to-row variation --
#   they are simply small in magnitude (values around +/-0.005) -- so testing
#   them against the same absolute std threshold used for sensors would
#   wrongly remove them too, even though they are not constant.
print("\n--- Removing Near-Constant Predictors ---")

sensor_cols = [col for col in train_data.columns if col.startswith("sensor_")]
setting_cols = [col for col in train_data.columns if col.startswith("setting_")]

sensor_std = train_data[sensor_cols].std().sort_values()
setting_nunique = train_data[setting_cols].nunique().sort_values()

# Save both variability tables as evidence for the removal decision
sensor_std.rename("Std_Dev").to_csv(
    os.path.join(OUTPUT_DIR, "Tables", "sensor_variability.csv"))
setting_nunique.rename("Unique_Values").to_csv(
    os.path.join(OUTPUT_DIR, "Tables", "setting_variability.csv"))
print("  Sensor and setting variability tables saved.")

STD_THRESHOLD = 0.01
constant_sensors = sensor_std[sensor_std < STD_THRESHOLD].index.tolist()
constant_settings = setting_nunique[setting_nunique == 1].index.tolist()
constant_predictors = constant_sensors + constant_settings

print(f"Sensors with std < {STD_THRESHOLD}: {constant_sensors}")
print(f"Settings with a single distinct value: {constant_settings}")
train_data = train_data.drop(columns=constant_predictors)
test_data = test_data.drop(columns=constant_predictors)

remaining_sensors = [col for col in train_data.columns if col.startswith("sensor_")]
remaining_settings = [col for col in train_data.columns if col.startswith("setting_")]
print(f"Remaining sensors: {len(remaining_sensors)}")
print(f"Remaining settings: {len(remaining_settings)}")
print(f"Training data shape after removal: {train_data.shape}")

print("\n--- After sensor removal: first 5 rows ---")
print(train_data.head())

# --- 2c. RUL target construction ---------------------------------------------
# Following Heimes (2008), RUL is capped at 125 cycles because sensors
# show little degradation early in engine life, and the flat early region
# would otherwise dominate the training signal.
RUL_CAP = 125

print("\n--- RUL Target Construction ---")
max_cycles = train_data.groupby("engine_id")["cycle"].max()
train_data = train_data.merge(max_cycles.rename("max_cycle"),
                               on="engine_id", how="left")
train_data["RUL"] = train_data["max_cycle"] - train_data["cycle"]
train_data["RUL"] = train_data["RUL"].clip(upper=RUL_CAP)
train_data = train_data.drop(columns=["max_cycle"])

print(f"RUL calculated and capped at {RUL_CAP} cycles.")
print("\n--- After RUL construction: first 5 rows ---")
print(train_data.head())
print("\n--- After RUL construction: last 5 rows ---")
print(train_data.tail())
print(f"\nTraining data shape: {train_data.shape}")
print(f"\nData types:\n{train_data.dtypes}")
print(f"\n--- Data info (after RUL construction) ---")
train_data.info()

# --- 2d. Confirm final predictor features ------------------------------------
print("\n--- Final Predictor Features ---")
feature_cols = [col for col in train_data.columns
                if col not in ["engine_id", "RUL"]]
print(f"Number of predictors: {len(feature_cols)}")
print(f"Features: {feature_cols}")

X = train_data[feature_cols]
y = train_data["RUL"]

# --- 2e. Feature normalization ------------------------------------------------
# Predictors are normalized (zero mean, unit variance) here, before any EDA is
# produced, so that every exploratory statistic and figure that follows is
# computed on the same scale the models will eventually see. This scaler is
# fitted on the full 100-engine training set purely to support EDA; it is
# NOT reused for model fitting. A separate scaler fitted only on the training
# split (Step 5 onward) is used for the models themselves, so that validation
# and test data never influence the scaling parameters used for prediction.
print("\n--- Feature Normalization (for EDA) ---")
eda_scaler = StandardScaler().fit(X)
X_scaled = pd.DataFrame(eda_scaler.transform(X), columns=feature_cols, index=X.index)

print("\n--- After normalization: first 5 rows ---")
print(X_scaled.head())
print("\n--- After normalization: last 5 rows ---")
print(X_scaled.tail())

scaled_check = X_scaled.describe().T[["mean", "std"]].round(3)
scaled_check.to_csv(os.path.join(OUTPUT_DIR, "Tables", "normalized_feature_summary.csv"))
print("\nPredictors normalized (StandardScaler). Mean/std after normalization:")
print(scaled_check)


# ==============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Summary statistics (normalized predictors + raw RUL target)
summary = pd.concat([X_scaled, y], axis=1).describe().T
summary["median"] = pd.concat([X_scaled, y], axis=1).median()
summary["skewness"] = pd.concat([X_scaled, y], axis=1).skew()
summary.to_csv(os.path.join(OUTPUT_DIR, "Tables", "summary_statistics.csv"))
print("Summary statistics saved.")

# RUL histogram
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(train_data["RUL"], bins=30, kde=True, ax=ax)
ax.set_title("Distribution of RUL (capped at 125)")
ax.set_xlabel("RUL")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "rul_histogram.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("RUL histogram saved.")

# RUL boxplot
fig, ax = plt.subplots(figsize=(7, 3))
sns.boxplot(x=train_data["RUL"], ax=ax)
ax.set_title("Boxplot of RUL")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "rul_boxplot.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("RUL boxplot saved.")

# QQ plot
fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(train_data["RUL"], dist="norm", plot=ax)
ax.set_title("QQ Plot of RUL")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "rul_qqplot.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("QQ plot saved.")

# Normality tests
shapiro_stat, shapiro_p = stats.shapiro(
    np.random.RandomState(SEED).choice(y.values, size=5000, replace=False))
ks_stat, ks_p = stats.kstest(y.values, "norm",
                              args=(y.mean(), y.std()))
print(f"Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.6f}")
print(f"KS test:      stat={ks_stat:.4f}, p={ks_p:.6f}")

normality_results = pd.DataFrame([
    {"Test": "Shapiro-Wilk", "Statistic": shapiro_stat, "p_value": shapiro_p},
    {"Test": "Kolmogorov-Smirnov", "Statistic": ks_stat, "p_value": ks_p},
])
normality_results.to_csv(os.path.join(OUTPUT_DIR, "Tables", "normality_tests.csv"),
                          index=False)

# Sensor trend plot — Engine 1, sensor_2 (first retained sensor)
# Plotted in raw physical units rather than normalized units, since the
# purpose here is to show how the sensor itself drifts as the engine
# degrades; normalized (unitless) values would obscure that physical
# interpretation. Normalization has already been applied above and is
# used for every other EDA output and for model training.
engine_1 = train_data[train_data["engine_id"] == 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(engine_1["cycle"], engine_1["sensor_2"])
ax.set_title("Sensor 2 Trend Over Time — Engine 1")
ax.set_xlabel("Cycle")
ax.set_ylabel("Sensor 2")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "sensor_trend_example.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Sensor trend plot saved.")

# Full correlation heatmap (computed on normalized predictors + RUL)
fig, ax = plt.subplots(figsize=(12, 9))
corr_matrix = pd.concat([X_scaled, y], axis=1).corr()
sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "correlation_heatmap.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Correlation heatmap saved.")

# Save the full correlation matrix as CSV for reference
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "Tables", "correlation_matrix.csv"))
print("  Correlation matrix saved.")

# Feature-to-RUL correlation bar chart
corr_with_rul = corr_matrix["RUL"].drop("RUL").sort_values()

# Save correlation values as CSV
pd.DataFrame({"Feature": corr_with_rul.index,
              "Pearson_r": corr_with_rul.values}).to_csv(
    os.path.join(OUTPUT_DIR, "Tables", "correlation_with_rul.csv"), index=False)
print("  Feature-to-RUL correlations saved.")

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["firebrick" if v < 0 else "steelblue" for v in corr_with_rul]
ax.barh(corr_with_rul.index, corr_with_rul.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Pearson Correlation of Each Feature with RUL")
ax.set_xlabel("Correlation")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "correlation_with_rul.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Correlation bar chart saved.")


# ==============================================================================
# STEP 4: ENGINE-LEVEL DEVELOPMENT SPLIT
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: ENGINE-LEVEL DEVELOPMENT SPLIT")
print("=" * 60)

# 80/20 split at engine level ensures all cycles from one engine stay
# in the same subset, preventing near-identical consecutive cycles
# from leaking across the training/validation boundary.
TEST_SIZE = 0.20

all_engine_ids = train_data["engine_id"].unique()
train_ids, val_ids = train_test_split(all_engine_ids, test_size=TEST_SIZE,
                                       random_state=SEED)

train_mask = train_data["engine_id"].isin(train_ids)
val_mask = train_data["engine_id"].isin(val_ids)

X_train, X_val = X[train_mask], X[val_mask]
y_train, y_val = y[train_mask], y[val_mask]
train_groups = train_data.loc[train_mask, "engine_id"].values
all_groups = train_data["engine_id"].values

print(f"Training engines ({len(train_ids)}): {sorted(train_ids)}")
print(f"Validation engines ({len(val_ids)}): {sorted(val_ids)}")
print(f"Training rows: {len(X_train)}, Validation rows: {len(X_val)}")

pd.DataFrame({"Training_Engine_IDs": sorted(train_ids)}).to_csv(
    os.path.join(OUTPUT_DIR, "Tables", "train_engine_ids.csv"), index=False)
pd.DataFrame({"Validation_Engine_IDs": sorted(val_ids)}).to_csv(
    os.path.join(OUTPUT_DIR, "Tables", "validation_engine_ids.csv"), index=False)
print("  Engine ID lists saved.")


# ==============================================================================
# STEP 5: MODEL DEVELOPMENT
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL DEVELOPMENT")
print("=" * 60)

# Note on normalization: features were already normalized once, globally,
# in Step 2e so that EDA could be produced on normalized data. That scaler
# (eda_scaler) is not reused here. For model fitting, scaling is instead
# refitted on each training fold/split as it is needed, so that validation
# and test engines never influence the scaling parameters a model is
# evaluated against (data leakage).
#
# Linear Regression and SVR use pipelines with StandardScaler inside,
# so scaling is refitted on each fold and never leaks validation data.
# Random Forest does not need scaling — tree splits are scale-invariant.
#
# The LSTM is scaled separately using a dedicated StandardScaler fitted
# only on the training engines (see Step 7).

lr_pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model", LinearRegression())])

svr_pipeline = Pipeline([("scaler", StandardScaler()),
                          ("model", SVR(kernel="rbf"))])

# n_jobs=-1 uses all available CPU cores for parallel tree training
rf_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)

print("Models defined:")
print("  - Mean RUL Baseline (no scaling needed)")
print("  - Linear Regression (StandardScaler inside pipeline)")
print("  - Random Forest (no scaling — tree-based, scale-invariant)")
print("  - SVR (StandardScaler inside pipeline)")
print("  - LSTM (scaled separately in Step 7)")


# ==============================================================================
# STEP 6: HYPERPARAMETER OPTIMISATION
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 6: HYPERPARAMETER OPTIMISATION")
print("=" * 60)

# GroupKFold ensures no engine's cycles appear in both the training
# and validation portions of any tuning fold, preventing data leakage.
N_FOLDS = 5
tuning_cv = GroupKFold(n_splits=N_FOLDS)

# Random Forest tuning
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=10,
                                cv=tuning_cv, scoring="neg_root_mean_squared_error",
                                random_state=SEED, n_jobs=-1)
rf_search.fit(X_train, y_train, groups=train_groups)
print(f"Best RF params: {rf_search.best_params_}")

# SVR tuning
svr_params = {
    "model__C": [1, 10, 50, 100, 200, 500, 1000],
    "model__gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.1],
    "model__epsilon": [0.01, 0.1, 0.5, 1.0],
}
svr_search = RandomizedSearchCV(svr_pipeline, svr_params, n_iter=20,
                                 cv=tuning_cv, scoring="neg_root_mean_squared_error",
                                 random_state=SEED, n_jobs=-1)
svr_search.fit(X_train, y_train, groups=train_groups)
print(f"Best SVR params: {svr_search.best_params_}")

# Save best parameters
best_params = pd.DataFrame([
    {"Model": "Random Forest", "Best_Params": str(rf_search.best_params_)},
    {"Model": "SVR", "Best_Params": str(svr_search.best_params_)},
])
best_params.to_csv(os.path.join(OUTPUT_DIR, "Tables", "best_hyperparameters.csv"),
                    index=False)


# ==============================================================================
# STEP 7: VALIDATION AND CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 7: VALIDATION AND CROSS-VALIDATION")
print("=" * 60)

# --- 7a. Held-out validation -------------------------------------------------
print("\n--- Held-Out Validation (20 engines) ---")

# Fit LR on training split
lr_pipeline.fit(X_train, y_train)

# Predictions on validation set
baseline_pred = np.full(len(y_val), y_train.mean())
lr_pred = lr_pipeline.predict(X_val)
rf_pred = rf_search.best_estimator_.predict(X_val)
svr_pred = svr_search.best_estimator_.predict(X_val)

val_results = []
for name, preds in [("Mean Baseline", baseline_pred),
                     ("Linear Regression", lr_pred),
                     ("Random Forest", rf_pred),
                     ("SVR", svr_pred)]:
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    val_results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

val_df = pd.DataFrame(val_results)
val_df.to_csv(os.path.join(OUTPUT_DIR, "Tables", "validation_results.csv"),
               index=False)

# --- 7b. Overfitting check ---------------------------------------------------
print("\n--- Train vs Validation (overfitting check) ---")
for name, model in [("Linear Regression", lr_pipeline),
                     ("Random Forest", rf_search.best_estimator_),
                     ("SVR", svr_search.best_estimator_)]:
    train_r2 = r2_score(y_train, model.predict(X_train))
    val_r2 = r2_score(y_val, model.predict(X_val))
    gap = train_r2 - val_r2
    print(f"  {name}: Train R2={train_r2:.4f}, Val R2={val_r2:.4f}, Gap={gap:.4f}")

# --- 7c. Grouped cross-validation (all 100 engines) -------------------------
print("\n--- 5-Fold Grouped Cross-Validation ---")

cv_folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups=all_groups))
cv_results = []

for fold_i, (tr_idx, va_idx) in enumerate(cv_folds, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    for name, estimator in [("Mean Baseline", None),
                             ("Linear Regression", clone(lr_pipeline)),
                             ("Random Forest", clone(rf_search.best_estimator_)),
                             ("SVR", clone(svr_search.best_estimator_))]:
        if estimator is None:
            preds = np.full(len(y_va), y_tr.mean())
        else:
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_va)

        rmse = np.sqrt(mean_squared_error(y_va, preds))
        r2 = r2_score(y_va, preds)
        cv_results.append({"Model": name, "Fold": fold_i,
                           "RMSE": rmse, "R2": r2})

cv_df = pd.DataFrame(cv_results)
cv_summary = cv_df.groupby("Model")[["RMSE", "R2"]].agg(["mean", "std"])
print(cv_summary)

cv_df.to_csv(os.path.join(OUTPUT_DIR, "Tables", "cv_fold_results.csv"), index=False)

# CV boxplot
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=cv_df, x="Model", y="RMSE", ax=ax)
ax.set_title("5-Fold Grouped Cross-Validation RMSE")
ax.tick_params(axis="x", rotation=15)
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "cv_rmse_boxplot.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# --- 7d. LSTM ----------------------------------------------------------------
print("\n--- LSTM Training ---")

# Number of past cycles used as input to the LSTM — the model sees a
# window of 30 consecutive sensor readings to predict the current RUL.
LSTM_WINDOW = 30

# Scale features for LSTM (fitted on training engines only)
dev_scaler = StandardScaler().fit(X_train)
X_scaled = pd.DataFrame(dev_scaler.transform(X), columns=X.columns, index=X.index)


def build_sequences(features, target, engine_ids, cycles, engines, window,
                    last_only=False):
    """Build sliding-window sequences for LSTM input."""
    seqs, targets, meta = [], [], []
    for eid in engines:
        mask = engine_ids == eid
        f = features[mask].values
        t = target[mask].values
        c = cycles[mask].values
        order = np.argsort(c)
        f, t, c = f[order], t[order], c[order]

        if len(f) < window:
            # Pad short engines by repeating first row
            pad = np.repeat(f[0:1], window - len(f), axis=0)
            f = np.vstack([pad, f])
            seqs.append(f[-window:])
            targets.append(t[-1])
            meta.append((eid, c[-1]))
        elif last_only:
            seqs.append(f[-window:])
            targets.append(t[-1])
            meta.append((eid, c[-1]))
        else:
            for end in range(window - 1, len(f)):
                seqs.append(f[end - window + 1: end + 1])
                targets.append(t[end])
                meta.append((eid, c[end]))

    return np.array(seqs), np.array(targets), pd.DataFrame(meta, columns=["engine_id", "cycle"])


X_seq_train, y_seq_train, _ = build_sequences(
    X_scaled, y, train_data["engine_id"], train_data["cycle"],
    train_ids, LSTM_WINDOW)

X_seq_val, y_seq_val, meta_val = build_sequences(
    X_scaled, y, train_data["engine_id"], train_data["cycle"],
    val_ids, LSTM_WINDOW)

print(f"Training sequences: {X_seq_train.shape}")
print(f"Validation sequences: {X_seq_val.shape}")


def build_lstm(n_features):
    """Build a simple 2-layer LSTM model."""
    model = Sequential([
        LSTM(64, input_shape=(LSTM_WINDOW, n_features), return_sequences=True,
             implementation=1),
        Dropout(0.2),
        LSTM(32, implementation=1),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


dev_lstm = build_lstm(X_seq_train.shape[2])
early_stop = EarlyStopping(monitor="val_loss", patience=5,
                            restore_best_weights=True)

history = dev_lstm.fit(X_seq_train, y_seq_train,
                       validation_data=(X_seq_val, y_seq_val),
                       epochs=50, batch_size=64,
                       callbacks=[early_stop], verbose=2)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"Best epoch: {best_epoch}")

epochs = np.arange(1, len(history.history["loss"]) + 1)

# LSTM training history plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, history.history["loss"], label="Training loss")
ax.plot(epochs, history.history["val_loss"], label="Validation loss")
ax.axvline(best_epoch, color="grey", linestyle="--",
           label=f"Best epoch ({best_epoch})")
ax.set_title("LSTM Training History")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_xticks(epochs)
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "lstm_training_history.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# LSTM validation score
lstm_val_pred = dev_lstm.predict(X_seq_val, verbose=0).flatten()
lstm_mae = mean_absolute_error(y_seq_val, lstm_val_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_seq_val, lstm_val_pred))
lstm_r2 = r2_score(y_seq_val, lstm_val_pred)
print(f"LSTM Validation: MAE={lstm_mae:.2f}, RMSE={lstm_rmse:.2f}, R2={lstm_r2:.4f}")


# ==============================================================================
# STEP 8: FINAL RETRAINING (ALL 100 ENGINES)
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 8: FINAL RETRAINING")
print("=" * 60)

# Retrain all models on the full 100 training engines
rf_best_params = rf_search.best_params_
svr_best_params = {k.replace("model__", ""): v
                   for k, v in svr_search.best_params_.items()}

start = time.time()
final_lr = Pipeline([("scaler", StandardScaler()),
                      ("model", LinearRegression())]).fit(X, y)
lr_time = time.time() - start

start = time.time()
final_rf = RandomForestRegressor(random_state=SEED, n_jobs=-1,
                                  **rf_best_params).fit(X, y)
rf_time = time.time() - start

start = time.time()
final_svr = Pipeline([("scaler", StandardScaler()),
                       ("model", SVR(kernel="rbf", **svr_best_params))]).fit(X, y)
svr_time = time.time() - start

# LSTM: retrain on all engines with fixed epoch count
final_scaler = StandardScaler().fit(X)
X_scaled_full = pd.DataFrame(final_scaler.transform(X),
                              columns=X.columns, index=X.index)

X_seq_full, y_seq_full, _ = build_sequences(
    X_scaled_full, y, train_data["engine_id"], train_data["cycle"],
    all_engine_ids, LSTM_WINDOW)

start = time.time()
final_lstm = build_lstm(X_seq_full.shape[2])
final_lstm.fit(X_seq_full, y_seq_full, epochs=best_epoch,
               batch_size=64, verbose=2)
lstm_time = time.time() - start

print(f"\nAll models retrained on {len(all_engine_ids)} engines.")
print(f"Training times — LR: {lr_time:.1f}s, RF: {rf_time:.1f}s, "
      f"SVR: {svr_time:.1f}s, LSTM: {lstm_time:.1f}s")


# ==============================================================================
# STEP 9: OFFICIAL NASA TEST EVALUATION
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 9: OFFICIAL NASA TEST EVALUATION")
print("=" * 60)


def nasa_phm_score(y_true, y_pred):
    """NASA PHM asymmetric scoring function (Saxena et al., 2008)."""
    error = np.array(y_pred) - np.array(y_true)
    scores = np.where(error < 0,
                      np.exp(-error / 13) - 1,
                      np.exp(error / 10) - 1)
    return float(scores.sum()), float(scores.mean())


# Get last cycle per test engine
test_last = test_data.groupby("engine_id").last().reset_index()
X_test = test_last[feature_cols]

# True RUL (capped for fair comparison)
true_rul = rul_data["true_rul"].values
true_rul_capped = np.clip(true_rul, 0, RUL_CAP)

# Build LSTM test sequences
test_engine_ids = np.sort(test_data["engine_id"].unique())
X_test_scaled = pd.DataFrame(final_scaler.transform(test_data[feature_cols]),
                              columns=feature_cols, index=test_data.index)
X_test_seq, _, _ = build_sequences(
    X_test_scaled, pd.Series(np.zeros(len(test_data)), index=test_data.index),
    test_data["engine_id"], test_data["cycle"],
    test_engine_ids, LSTM_WINDOW, last_only=True)

# Predictions
predictions = {
    "Mean Baseline": np.full(len(test_last), y.mean()),
    "Linear Regression": final_lr.predict(X_test),
    "Random Forest": final_rf.predict(X_test),
    "SVR": final_svr.predict(X_test),
    "LSTM": final_lstm.predict(X_test_seq, verbose=0).flatten(),
}

# Evaluate all models
print("\n--- Test Results (capped RUL) ---")
test_results = []
for name, preds in predictions.items():
    mae = mean_absolute_error(true_rul_capped, preds)
    rmse = np.sqrt(mean_squared_error(true_rul_capped, preds))
    r2 = r2_score(true_rul_capped, preds)
    nasa_total, nasa_mean = nasa_phm_score(true_rul_capped, preds)

    test_results.append({
        "Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
        "R2": round(r2, 4), "NASA_Score": round(nasa_total, 2),
    })
    print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
          f"R2={r2:.4f}, NASA={nasa_total:.2f}")

test_df = pd.DataFrame(test_results)
test_df.to_csv(os.path.join(OUTPUT_DIR, "Tables", "test_results.csv"), index=False)

# Diagnostic plots for each model
for name, preds in predictions.items():
    prefix = name.lower().replace(" ", "_")

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_rul_capped, preds, alpha=0.5)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "r--", label="Perfect prediction")
    ax.set_xlim(0, RUL_CAP)
    ax.set_ylim(0, RUL_CAP)
    ax.set_title(f"Actual vs Predicted — {name}")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figures", f"{prefix}_actual_vs_pred.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Residuals
    residuals = true_rul_capped - preds
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(preds, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Residuals — {name}")
    ax.set_xlabel("Predicted RUL")
    ax.set_ylabel("Residual")
    fig.savefig(os.path.join(OUTPUT_DIR, "Figures", f"{prefix}_residuals.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Residual histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    ax.set_title(f"Residual Distribution — {name}")
    fig.savefig(os.path.join(OUTPUT_DIR, "Figures", f"{prefix}_residual_hist.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

print("Diagnostic plots saved.")

# Save per-engine predictions
pred_table = pd.DataFrame({"engine_id": test_last["engine_id"].values,
                            "Actual_RUL": true_rul_capped})
for name, preds in predictions.items():
    pred_table[name.replace(" ", "_")] = preds
pred_table.to_csv(os.path.join(OUTPUT_DIR, "Tables", "test_predictions.csv"),
                   index=False)


# ==============================================================================
# STEP 10: MODEL INTERPRETATION
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 10: MODEL INTERPRETATION")
print("=" * 60)

# Impurity-based feature importance (Random Forest)
importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": final_rf.feature_importances_,
}).sort_values("Importance", ascending=False)
importance.to_csv(os.path.join(OUTPUT_DIR, "Tables", "rf_feature_importance.csv"),
                   index=False)
print("Top 5 features (impurity-based):")
print(importance.head())

# Permutation importance (on validation data)
perm = permutation_importance(rf_search.best_estimator_, X_val, y_val,
                               n_repeats=10, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": perm.importances_mean,
}).sort_values("Importance", ascending=False)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "Tables", "rf_permutation_importance.csv"),
                index=False)

# Feature importance plot (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
sns.barplot(data=importance, x="Importance", y="Feature", ax=ax1)
ax1.set_title("Impurity-Based Importance")
sns.barplot(data=perm_df, x="Importance", y="Feature", ax=ax2)
ax2.set_title("Permutation Importance")
fig.suptitle("Random Forest Feature Importance")
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "feature_importance.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Feature importance plots saved.")

# Top feature trend across multiple engines
top_feature = perm_df.iloc[0]["Feature"]
print(f"\nTop feature by permutation importance: {top_feature}")

fig, ax = plt.subplots(figsize=(8, 4))
for eid in [1, 20, 50, 80, 100]:
    subset = train_data[train_data["engine_id"] == eid]
    ax.plot(subset["cycle"], subset[top_feature], label=f"Engine {eid}")
ax.set_title(f"{top_feature} Trend Across Engines")
ax.set_xlabel("Cycle")
ax.set_ylabel(top_feature)
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, "Figures", "top_feature_trend.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# ==============================================================================
# STEP 11: SAVE OUTPUTS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 11: SAVE OUTPUTS")
print("=" * 60)

# Save trained models
joblib.dump(final_lr, os.path.join(OUTPUT_DIR, "Models", "final_lr.joblib"))
joblib.dump(final_rf, os.path.join(OUTPUT_DIR, "Models", "final_rf.joblib"))
joblib.dump(final_svr, os.path.join(OUTPUT_DIR, "Models", "final_svr.joblib"))
joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, "Models", "final_scaler.joblib"))
final_lstm.save(os.path.join(OUTPUT_DIR, "Models", "final_lstm.keras"))
print("All models saved.")

# Package versions
import sklearn, scipy
versions = pd.DataFrame([{
    "numpy": np.__version__, "pandas": pd.__version__,
    "sklearn": sklearn.__version__, "scipy": scipy.__version__,
    "tensorflow": tf.__version__,
}])
versions.to_csv(os.path.join(OUTPUT_DIR, "Tables", "package_versions.csv"),
                 index=False)
print("Package versions saved.")

# --- Final summary comparing all five models ---------------------------------
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON (Official NASA Test Set)")
print("=" * 60)
print(pd.DataFrame(test_results).to_string(index=False))

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"All outputs saved under '{OUTPUT_DIR}/'")
print("=" * 60)