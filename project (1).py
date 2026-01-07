# =====================================
# 1. Imports
# =====================================
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)


# =====================================
# 2. Load data
# =====================================
FILE_PATH = "Main Projectdataset_conflicto_deforestacion (1).xlsx"

df = pd.read_excel(FILE_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# =====================================
# 3. Drop rows with NaN target
# =====================================
TARGET = "deforestacion_año_siguiente"

df = df.dropna(subset=[TARGET])
print("Dataset shape after dropping NaN target:", df.shape)

print("Remaining NaNs in target:", df[TARGET].isna().sum())


# =====================================
# 4. Define features and target
# =====================================
X = df.drop(columns=[TARGET])
y = df[TARGET]


# =====================================
# 5. Identify feature types
# =====================================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))


# =====================================
# 6. Preprocessing pipelines
# =====================================
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# =====================================
# 7. Temporal train / test split
#    (predict future using the past)
# =====================================
TRAIN_END_YEAR = 2018

train_df = df[df["año"] <= TRAIN_END_YEAR]
test_df = df[df["año"] > TRAIN_END_YEAR]

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Train years: <= {TRAIN_END_YEAR}")
print(f"Test years:  > {TRAIN_END_YEAR}")
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# =====================================
# 8. Models
# =====================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "Support Vector Machine": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1,
    ),
}


# =====================================
# 9. Helper: MAE by deforestation range
# =====================================
def mae_by_bin(y_true, y_pred):
    dfm = pd.DataFrame({"y": y_true, "pred": y_pred})
    dfm["abs_err"] = (dfm["y"] - dfm["pred"]).abs()

    bins = [0, 20, 50, 150, 500, np.inf]
    labels = ["<=20", "20-50", "50-150", "150-500", ">500"]
    dfm["bin"] = pd.cut(
        dfm["y"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return (
        dfm.groupby("bin")
        .agg(
            n=("y", "size"),
            y_mean=("y", "mean"),
            MAE=("abs_err", "mean"),
            MedAE=("abs_err", "median"),
        )
        .reset_index()
    )


# =====================================
# 10. Training & evaluation
# =====================================
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MedAE": median_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    results[model_name] = {
        "metrics": metrics,
        "mae_by_bin": mae_by_bin(y_test, y_pred),
    }

    print("MAE :", metrics["MAE"])
    print("MedAE:", metrics["MedAE"])
    print("RMSE:", metrics["RMSE"])
    print("R2  :", metrics["R2"])


# =====================================
# 11. Model comparison summary
# =====================================
summary_df = (
    pd.DataFrame(
        {
            model: vals["metrics"]
            for model, vals in results.items()
        }
    )
    .T.sort_values(by="RMSE")
)

print("\nModel comparison:")
print(summary_df)


# =====================================
# 12. Detailed error analysis (key!)
# =====================================
for model_name, vals in results.items():
    print(f"\nMAE by deforestation range — {model_name}")
    print(vals["mae_by_bin"])


# =====================================
# 13. Visualizations for Random Forest
# =====================================
import os
import matplotlib.pyplot as plt

VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# Retrieve Random Forest results
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", models["Random Forest"]),
    ]
)

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# -------------------------------------
# 13.1 Predictions vs Actual
# -------------------------------------
plt.figure()
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)
plt.xlabel("Actual deforestation (t+1)")
plt.ylabel("Predicted deforestation (t+1)")
plt.title("Random Forest: Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "rf_predictions_vs_actual.png"))
plt.close()


# -------------------------------------
# 13.2 Residuals vs Actual
# -------------------------------------
residuals = y_test - y_pred_rf

plt.figure()
plt.scatter(y_test, residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("Actual deforestation (t+1)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Random Forest: Residuals vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "rf_residuals.png"))
plt.close()


# -------------------------------------
# 13.3 Feature importance
# -------------------------------------
# Get feature names after preprocessing
ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"] \
    .named_steps["encoder"]

categorical_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate(
    [numeric_features, categorical_feature_names]
)

importances = rf_pipeline.named_steps["model"].feature_importances_

imp_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    .sort_values("importance", ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest: Top 20 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "rf_feature_importance.png"))
plt.close()


# -------------------------------------
# 13.4 MAE by deforestation range (side-by-side with mean_y)
# -------------------------------------
mae_bins_rf = mae_by_bin(y_test, y_pred_rf)

bins = mae_bins_rf["bin"].astype(str)
mae_vals = mae_bins_rf["MAE"]
mean_y_vals = mae_bins_rf["y_mean"]

x = np.arange(len(bins))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    mae_vals,
    width,
    label="MAE",
)

plt.bar(
    x + width / 2,
    mean_y_vals,
    width,
    label="Mean actual deforestation",
)

plt.xticks(x, bins)
plt.xlabel("Deforestation range (actual)")
plt.ylabel("Value")
plt.title("Random Forest: MAE vs Mean Deforestation by Range")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "rf_mae_vs_mean_by_bin.png"))
plt.close()

