# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────
import warnings, os, textwrap
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, confusion_matrix,
                                      ConfusionMatrixDisplay, roc_curve, classification_report)
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import GridSearchCV

import xgboost  as xgb
import lightgbm as lgb
import shap

SEED       = 42
PLOT_DIR   = "plots"
OUTPUT_DIR = "outputs"
DATA_PATH  = "/mnt/user-data/uploads/TravelInsurancePrediction.csv"

os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"    : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
})
PALETTE = {"No": "#4A90D9", "Yes": "#E8534A", 0: "#4A90D9", 1: "#E8534A"}

print("=" * 68)
print(" TRAVEL INSURANCE PURCHASE PREDICTION — ML PIPELINE")
print("=" * 68)


# ─────────────────────────────────────────────────────────────────
# 1. DATASET UNDERSTANDING
# ─────────────────────────────────────────────────────────────────
print("\n[1] DATASET UNDERSTANDING")
print("-" * 50)

df_raw = pd.read_csv(DATA_PATH)
df     = df_raw.copy()

# Drop the index column that was saved with the CSV
df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"Shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Duplicates      : {df.duplicated().sum()}")
print(f"Missing values  : {df.isnull().sum().sum()}")
print("\nColumn overview:")
print(df.dtypes.rename("dtype").to_frame().join(
    df.isnull().sum().rename("nulls")).to_string())

TARGET  = "TravelInsurance"
NUMERIC = ["Age", "AnnualIncome", "FamilyMembers", "ChronicDiseases"]
CATEG   = ["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]

print(f"\nTarget          : {TARGET}")
print(f"Numeric features: {NUMERIC}")
print(f"Categorical feat: {CATEG}")
print("\nTarget class balance:")
vc = df[TARGET].value_counts()
print(f"  0 (No purchase) : {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  1 (Purchased)   : {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")
print("\nStatistical summary:")
print(df.describe(include="all").T.to_string())


# ─────────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────
print("\n[2] DATA PREPROCESSING")
print("-" * 50)

# 2a No missing values → nothing to impute
print("✓ No missing values detected — no imputation required.")

# 2b No duplicates → nothing to drop
print("✓ No duplicate rows detected — no removal required.")

# 2c Outlier analysis via IQR on numeric cols (report only; no removal
#    because the dataset is small and the range is business-reasonable)
print("\nOutlier check (IQR method):")
for col in ["Age", "AnnualIncome", "FamilyMembers"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:<20}: {n_out} potential outliers  (range [{df[col].min()}, {df[col].max()}])")

# 2d Categorical encoding: binary Yes/No → 1/0; Employment Type Label-Encoded
print("\nCategorical encoding:")
binary_cols = ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)
    print(f"  {col}: Yes→1, No→0")

le = LabelEncoder()
df["Employment Type"] = le.fit_transform(df["Employment Type"])
print(f"  Employment Type: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nPost-encoding dtypes:")
print(df.dtypes.to_string())


# ─────────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n[3] EXPLORATORY DATA ANALYSIS")
print("-" * 50)

# --- 3a Target distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vc_plot = df_raw[TARGET].value_counts()
axes[0].bar(["No Purchase (0)", "Purchased (1)"], vc_plot.values,
            color=["#4A90D9", "#E8534A"], width=0.5, edgecolor="white")
for i, v in enumerate(vc_plot.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")
axes[0].set_title("Target Variable Distribution")
axes[0].set_ylabel("Count")

axes[1].pie(vc_plot.values, labels=["No Purchase", "Purchased"],
            colors=["#4A90D9", "#E8534A"], autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Proportion")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_target_distribution.png")
plt.close()
print("✓ Saved: 01_target_distribution.png")

# --- 3b Correlation heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap (All Encoded Features)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png")
plt.close()
print("✓ Saved: 02_correlation_heatmap.png")

# --- 3c Numeric feature distributions by target ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
num_feats = ["Age", "AnnualIncome", "FamilyMembers", "ChronicDiseases"]
for ax, col in zip(axes.flat, num_feats):
    for val in [0, 1]:
        subset = df_raw[df_raw[TARGET] == val][col]
        ax.hist(subset, bins=20, alpha=0.65, label=f"{'Purchased' if val else 'Not Purchased'}",
                color=PALETTE[val], edgecolor="white")
    ax.set_title(f"{col} Distribution by Target")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
plt.suptitle("Numeric Feature Distributions by Insurance Purchase", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_numeric_distributions.png", bbox_inches="tight")
plt.close()
print("✓ Saved: 03_numeric_distributions.png")

# --- 3d Boxplots for outlier detection ---
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, col in zip(axes, ["Age", "AnnualIncome", "FamilyMembers"]):
    df_raw.boxplot(column=col, by=TARGET, ax=ax,
                   boxprops=dict(color="#4A90D9"),
                   medianprops=dict(color="#E8534A", linewidth=2),
                   whiskerprops=dict(color="#555"),
                   capprops=dict(color="#555"),
                   flierprops=dict(marker="o", markerfacecolor="#E8534A", markersize=4, alpha=0.5))
    ax.set_title(f"{col} by Target")
    ax.set_xlabel("TravelInsurance (0=No, 1=Yes)")
plt.suptitle("Boxplots: Numeric Features vs Target", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_boxplots.png")
plt.close()
print("✓ Saved: 04_boxplots.png")

# --- 3e Categorical feature analysis ---
cat_orig = ["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, cat_orig):
    ct = df_raw.groupby([col, TARGET]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax, color=["#4A90D9", "#E8534A"],
                edgecolor="white", width=0.6)
    ax.set_title(f"{col} — Purchase Rate")
    ax.set_ylabel("% of group")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(["No Purchase", "Purchased"], fontsize=9)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
plt.suptitle("Categorical Features — Purchase Rate by Category", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_categorical_analysis.png", bbox_inches="tight")
plt.close()
print("✓ Saved: 05_categorical_analysis.png")

# --- 3f Annual income vs age scatter ---
fig, ax = plt.subplots(figsize=(9, 5))
for val, label, color in [(0, "Not Purchased", "#4A90D9"), (1, "Purchased", "#E8534A")]:
    sub = df_raw[df_raw[TARGET] == val]
    ax.scatter(sub["Age"], sub["AnnualIncome"], alpha=0.4, s=18,
               color=color, label=label)
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (₹)")
ax.set_title("Age vs Annual Income — Coloured by Insurance Purchase")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_age_income_scatter.png")
plt.close()
print("✓ Saved: 06_age_income_scatter.png")

# --- 3g Income by EverTravelledAbroad ---
fig, ax = plt.subplots(figsize=(8, 5))
for val, label, color in [("No", "Never Abroad", "#4A90D9"), ("Yes", "Travelled Abroad", "#E8534A")]:
    sub = df_raw[df_raw["EverTravelledAbroad"] == val]["AnnualIncome"]
    ax.hist(sub, bins=20, alpha=0.65, label=label, color=color, edgecolor="white")
ax.set_title("Income Distribution by International Travel Experience")
ax.set_xlabel("Annual Income (₹)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_income_by_abroad.png")
plt.close()
print("✓ Saved: 07_income_by_abroad.png")

print("\nEDA complete — 7 visualizations saved.")


# ─────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n[4] FEATURE ENGINEERING")
print("-" * 50)

# Income Group: segment customers by wealth tier
income_bins   = [0, 600000, 900000, 1200000, float("inf")]
income_labels = [0, 1, 2, 3]   # Low, Mid, High, Premium
df["IncomeGroup"] = pd.cut(df["AnnualIncome"], bins=income_bins,
                           labels=income_labels).astype(int)
print("✓ IncomeGroup (0=Low <6L, 1=Mid 6-9L, 2=High 9-12L, 3=Premium >12L)")

# Income per family member: financial burden indicator
df["IncomePerMember"] = df["AnnualIncome"] / df["FamilyMembers"]
print("✓ IncomePerMember = AnnualIncome / FamilyMembers")

# Risk Score: composite risk indicator (ChronicDisease + large family + older age)
df["RiskScore"] = (df["ChronicDiseases"] +
                   (df["FamilyMembers"] >= 5).astype(int) +
                   (df["Age"] >= 30).astype(int))
print("✓ RiskScore = ChronicDisease + LargeFamily(≥5) + OlderAge(≥30)  [0-3]")

# Travel Propensity: already travelled abroad + frequent flyer = high propensity
df["TravelPropensity"] = df["FrequentFlyer"] + df["EverTravelledAbroad"]
print("✓ TravelPropensity = FrequentFlyer + EverTravelledAbroad  [0-2]")

# AgeGroup: discretise age
df["AgeGroup"] = pd.cut(df["Age"], bins=[24, 28, 31, 35],
                         labels=[0, 1, 2]).astype(int)
print("✓ AgeGroup (0=25-28, 1=29-31, 2=32-35)")

ALL_FEATURES = (["Age", "AnnualIncome", "FamilyMembers", "ChronicDiseases",
                  "Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad",
                  "IncomeGroup", "IncomePerMember", "RiskScore", "TravelPropensity", "AgeGroup"])

print(f"\nTotal features after engineering: {len(ALL_FEATURES)}")


# ─────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT  &  SCALING
# ─────────────────────────────────────────────────────────────────
print("\n[5] TRAIN/TEST SPLIT & SCALING")
print("-" * 50)

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"Training set  : {X_train.shape[0]:,} samples  ({y_train.mean()*100:.1f}% positive)")
print(f"Test set      : {X_test.shape[0]:,} samples  ({y_test.mean()*100:.1f}% positive)")

# Scale for models sensitive to magnitude (Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("✓ StandardScaler fitted on training data; applied to test data.")


# ─────────────────────────────────────────────────────────────────
# 6. MODEL BUILDING & EVALUATION
# ─────────────────────────────────────────────────────────────────
print("\n[6] MODEL BUILDING & EVALUATION")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    cv_auc  = cross_val_score(model, X_tr, y_tr, cv=cv,
                               scoring="roc_auc", n_jobs=-1).mean()
    metrics = {
        "Model"     : name,
        "Accuracy"  : round(accuracy_score(y_te, y_pred),  4),
        "Precision" : round(precision_score(y_te, y_pred), 4),
        "Recall"    : round(recall_score(y_te, y_pred),    4),
        "F1"        : round(f1_score(y_te, y_pred),        4),
        "ROC-AUC"   : round(roc_auc_score(y_te, y_proba),  4),
        "CV-AUC(5)" : round(cv_auc, 4),
    }
    print(f"  {name:<28} Acc={metrics['Accuracy']:.4f}  F1={metrics['F1']:.4f}  AUC={metrics['ROC-AUC']:.4f}")
    return metrics, model, y_pred, y_proba

results   = []
trained   = {}

# --- Logistic Regression (scaled data) ---
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
m, mdl, yp, yprob = evaluate_model("Logistic Regression", lr, X_train_sc, y_train, X_test_sc, y_test)
results.append(m); trained["Logistic Regression"] = (mdl, yp, yprob, True)

# --- Decision Tree (with light tuning) ---
dt_params = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
                     dt_params, cv=cv, scoring="roc_auc", n_jobs=-1)
m, mdl, yp, yprob = evaluate_model("Decision Tree", dt_gs, X_train, y_train, X_test, y_test)
results.append(m); trained["Decision Tree"] = (mdl, yp, yprob, False)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                             class_weight="balanced", random_state=SEED, n_jobs=-1)
m, mdl, yp, yprob = evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test)
results.append(m); trained["Random Forest"] = (mdl, yp, yprob, False)

# --- Gradient Boosting ---
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                  max_depth=4, subsample=0.8, random_state=SEED)
m, mdl, yp, yprob = evaluate_model("Gradient Boosting", gb, X_train, y_train, X_test, y_test)
results.append(m); trained["Gradient Boosting"] = (mdl, yp, yprob, False)

# --- XGBoost ---
xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              use_label_encoder=False, eval_metric="logloss",
                              scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1],
                              random_state=SEED, verbosity=0)
m, mdl, yp, yprob = evaluate_model("XGBoost", xgb_clf, X_train, y_train, X_test, y_test)
results.append(m); trained["XGBoost"] = (mdl, yp, yprob, False)

# --- LightGBM ---
lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               is_unbalance=True, random_state=SEED, verbose=-1)
m, mdl, yp, yprob = evaluate_model("LightGBM", lgb_clf, X_train, y_train, X_test, y_test)
results.append(m); trained["LightGBM"] = (mdl, yp, yprob, False)

# Comparison table
results_df = pd.DataFrame(results).set_index("Model").sort_values("ROC-AUC", ascending=False)
print("\nModel Comparison Table (sorted by ROC-AUC):")
print(results_df.to_string())

BEST_MODEL_NAME = results_df.index[0]
best_mdl, best_yp, best_yprob, best_scaled = trained[BEST_MODEL_NAME]
print(f"\n🏆 Best model: {BEST_MODEL_NAME}  (AUC={results_df.loc[BEST_MODEL_NAME,'ROC-AUC']:.4f})")
print(f"\nClassification Report — {BEST_MODEL_NAME}:")
print(classification_report(y_test, best_yp, target_names=["No Purchase", "Purchased"]))

# --- Visualise: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(results_df))
width = 0.15
colors = ["#4A90D9", "#5CBF85", "#E8534A", "#F5A623", "#9B59B6"]
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    ax.bar(x + i * width, results_df[metric], width, label=metric,
           color=color, edgecolor="white", alpha=0.9)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=15, ha="right")
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend(loc="lower right", ncol=5, fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_model_comparison.png")
plt.close()
print("✓ Saved: 08_model_comparison.png")

# --- Confusion matrices (all models) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, (mdl, yp, _, scaled)) in zip(axes.flat, trained.items()):
    cm = confusion_matrix(y_test, yp)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchased"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
plt.suptitle("Confusion Matrices — All Models", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_confusion_matrices.png")
plt.close()
print("✓ Saved: 09_confusion_matrices.png")

# --- ROC curves ---
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ["#4A90D9", "#5CBF85", "#E8534A", "#F5A623", "#9B59B6", "#1ABC9C"]
for (name, (_, _, yprob, _)), color in zip(trained.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    auc_val = roc_auc_score(y_test, yprob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/10_roc_curves.png")
plt.close()
print("✓ Saved: 10_roc_curves.png")


# ─────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (Best Model)
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] FEATURE IMPORTANCE — {BEST_MODEL_NAME}")
print("-" * 50)

_best_for_fi = best_mdl.best_estimator_ if isinstance(best_mdl, GridSearchCV) else best_mdl
if hasattr(_best_for_fi, "feature_importances_"):
    importances = _best_for_fi.feature_importances_
    fi_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(fi_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = ["#E8534A" if i == 0 else "#4A90D9" for i in range(len(fi_df))]
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
            color=colors_fi[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {BEST_MODEL_NAME}")
    for i, (val, feat) in enumerate(zip(fi_df["Importance"][::-1], fi_df["Feature"][::-1])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/11_feature_importance.png")
    plt.close()
    print("✓ Saved: 11_feature_importance.png")
elif hasattr(_best_for_fi, "coef_"):
    coefs = np.abs(_best_for_fi.coef_[0])
    fi_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": coefs})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(fi_df.to_string(index=False))
else:
    fi_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": np.zeros(len(ALL_FEATURES))})


# ─────────────────────────────────────────────────────────────────
# 8. EXPLAINABLE AI — SHAP
# ─────────────────────────────────────────────────────────────────
print("\n[8] EXPLAINABLE AI — SHAP VALUES")
print("-" * 50)

# Use the best tree-based model for SHAP
shap_model = best_mdl.best_estimator_ if isinstance(best_mdl, GridSearchCV) else best_mdl

try:
    explainer  = shap.TreeExplainer(shap_model)
    shap_vals  = explainer.shap_values(X_test)
    # For multi-output (RF), take class-1 SHAP
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # SHAP summary bar
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X_test, feature_names=ALL_FEATURES,
                      plot_type="bar", show=False, color="#4A90D9")
    plt.title(f"SHAP — Mean |SHAP| (Global Importance) — {BEST_MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/12_shap_bar.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved: 12_shap_bar.png")

    # SHAP beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=ALL_FEATURES, show=False)
    plt.title(f"SHAP Beeswarm — Feature Impact — {BEST_MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/13_shap_beeswarm.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved: 13_shap_beeswarm.png")
    shap_available = True

except Exception as e:
    print(f"  SHAP skipped: {e}")
    shap_available = False


# ─────────────────────────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────
print("\n[9] SAMPLE PREDICTIONS")
print("-" * 50)

sample_profiles = pd.DataFrame({
    "Age"                 : [25, 34, 30, 28],
    "AnnualIncome"        : [400000, 1500000, 900000, 700000],
    "FamilyMembers"       : [3, 5, 4, 8],
    "ChronicDiseases"     : [0, 0, 1, 1],
    "Employment Type"     : [1, 0, 1, 1],   # 1=Private, 0=Govt
    "GraduateOrNot"       : [1, 1, 1, 0],
    "FrequentFlyer"       : [0, 1, 1, 0],
    "EverTravelledAbroad" : [0, 1, 1, 0],
    "IncomeGroup"         : [0, 3, 1, 1],
    "IncomePerMember"     : [400000/3, 1500000/5, 900000/4, 700000/8],
    "RiskScore"           : [0, 2, 3, 2],
    "TravelPropensity"    : [0, 2, 2, 0],
    "AgeGroup"            : [0, 2, 1, 0],
})

X_samp = sample_profiles[ALL_FEATURES]
if best_scaled:
    X_samp_inp = scaler.transform(X_samp)
else:
    X_samp_inp = X_samp

preds  = best_mdl.predict(X_samp_inp)
probas = best_mdl.predict_proba(X_samp_inp)[:, 1]

print(f"{'Profile':<10} {'Prediction':<18} {'Prob(Buy)':<12} {'Interpretation'}")
print("-" * 75)
profiles_desc = [
    "Young low-income, no travel history",
    "Older high-income, frequent flyer, well-travelled",
    "Mid-income, chronic disease, travelled abroad",
    "Young low-income, large family, no travel"
]
for i, (pred, prob, desc) in enumerate(zip(preds, probas, profiles_desc)):
    label = "Will Purchase" if pred == 1 else "Will NOT Purchase"
    print(f"Profile {i+1:<3}  {label:<18}  {prob:.1%}   {desc}")


# ─────────────────────────────────────────────────────────────────
# 10. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
pd.DataFrame({"Profile": [f"P{i+1}" for i in range(4)],
              "Prediction": preds, "Prob_Purchase": probas.round(4),
              "Description": profiles_desc})\
  .to_csv(f"{OUTPUT_DIR}/sample_predictions.csv", index=False)

print("\n✓ model_comparison.csv, feature_importance.csv, sample_predictions.csv saved.")
print("\n" + "=" * 68)
print(" PIPELINE COMPLETE")
print("=" * 68)
