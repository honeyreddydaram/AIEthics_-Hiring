"""
AI Ethics – Hiring Bias Analysis
Full pipeline: descriptive stats → bias analysis → ML → fairness metrics
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "/Users/honey/Downloads/AI Ethics/archive_contents/final_bias_hiring_dataset.csv"
OUT_DIR    = "/Users/honey/Downloads/AI Ethics/hiring_bias_contents/hiring_bias_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
COLORS = {"Selected": "#2ecc71", "Rejected": "#e74c3c",
          "Male": "#3498db", "Female": "#e91e8c"}

DIVIDER = "=" * 65

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {name}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
print(DIVIDER)
print("1. LOADING DATASET")
print(DIVIDER)

df = pd.read_csv(DATA_PATH)
print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"Missing values :\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# encode binary target for ML
df["Hired"] = (df["Final_Decision"] == "Selected").astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("2. DESCRIPTIVE ANALYSIS")
print(DIVIDER)

total        = len(df)
total_hired  = df["Hired"].sum()
hire_rate    = total_hired / total * 100
avg_cgpa     = df["CGPA"].mean()
avg_exp      = df["Experience_Years"].mean()
gender_dist  = df["Gender"].value_counts()
edu_dist     = df["Education_Level"].value_counts()

print(f"Total Applicants   : {total}")
print(f"Total Selected     : {total_hired}")
print(f"Overall Hiring Rate: {hire_rate:.1f}%")
print(f"Avg CGPA           : {avg_cgpa:.2f}")
print(f"Avg Experience     : {avg_exp:.2f} years")
print(f"\nGender Distribution:\n{gender_dist.to_string()}")
print(f"\nEducation Distribution:\n{edu_dist.to_string()}")
print(f"\nNumerical Summary:\n{df[['CGPA','Experience_Years','Skills_Score','Interview_Score']].describe().round(2)}")

# Bar: overall outcome
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Dataset Overview", fontsize=15, fontweight="bold")

decision_counts = df["Final_Decision"].value_counts()
axes[0].bar(decision_counts.index, decision_counts.values,
            color=[COLORS[k] for k in decision_counts.index])
axes[0].set_title("Overall Hiring Decision")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, decision_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha="center", fontsize=11)

axes[1].bar(gender_dist.index, gender_dist.values,
            color=[COLORS[g] for g in gender_dist.index])
axes[1].set_title("Gender Distribution")
axes[1].set_ylabel("Count")

edu_counts = df["Education_Level"].value_counts()
axes[2].bar(edu_counts.index, edu_counts.values, color=sns.color_palette("pastel"))
axes[2].set_title("Education Level Distribution")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
save(fig, "01_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENDER BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("3. GENDER BIAS ANALYSIS")
print(DIVIDER)

gender_grp = df.groupby("Gender").agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
gender_grp["Hiring_Rate_%"] = (gender_grp["Hired"] / gender_grp["Applicants"] * 100).round(1)
print(gender_grp.to_string(index=False))

# Demographic Parity Difference
rates = gender_grp.set_index("Gender")["Hiring_Rate_%"]
dp_diff = abs(rates.get("Male", 0) - rates.get("Female", 0))

# Disparate Impact Ratio (female / male)
female_rate = rates.get("Female", 0) / 100
male_rate   = rates.get("Male",   0) / 100
di_ratio    = female_rate / male_rate if male_rate > 0 else float("nan")

print(f"\nDemographic Parity Difference : {dp_diff:.1f}%")
print(f"Disparate Impact Ratio (F/M)  : {di_ratio:.3f}  {'⚠ Possible discrimination (DI < 0.8)' if di_ratio < 0.8 else '✓ Within fair range'}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gender Bias Analysis", fontsize=15, fontweight="bold")

x = gender_grp["Gender"]
axes[0].bar(x, gender_grp["Applicants"], color=[COLORS[g] for g in x], alpha=0.6, label="Applicants")
axes[0].bar(x, gender_grp["Hired"],      color=[COLORS[g] for g in x], alpha=1.0, label="Hired")
axes[0].set_title("Applicants vs Hired by Gender")
axes[0].set_ylabel("Count")
axes[0].legend()

bars = axes[1].bar(x, gender_grp["Hiring_Rate_%"],
                   color=[COLORS[g] for g in x])
axes[1].set_title("Hiring Rate by Gender (%)")
axes[1].set_ylabel("Hiring Rate (%)")
axes[1].axhline(hire_rate, color="grey", linestyle="--", label=f"Overall avg {hire_rate:.1f}%")
axes[1].legend()
for bar, rate in zip(bars, gender_grp["Hiring_Rate_%"]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5, f"{rate}%", ha="center")

plt.tight_layout()
save(fig, "02_gender_bias.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIVERSITY BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("4. UNIVERSITY BIAS ANALYSIS")
print(DIVIDER)

uni_grp = df.groupby("University").agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
uni_grp["Hiring_Rate_%"] = (uni_grp["Hired"] / uni_grp["Applicants"] * 100).round(1)
# only universities with ≥5 applicants for reliable stats
uni_grp = uni_grp[uni_grp["Applicants"] >= 5].sort_values("Hiring_Rate_%", ascending=False)

print("Top 10 universities by hiring rate:")
print(uni_grp.head(10).to_string(index=False))
print("\nBottom 10 universities by hiring rate:")
print(uni_grp.tail(10).to_string(index=False))

top10    = uni_grp.head(10)
bottom10 = uni_grp.tail(10)

fig, axes = plt.subplots(2, 1, figsize=(14, 11))
fig.suptitle("University Hiring Rate Analysis", fontsize=15, fontweight="bold")

axes[0].barh(top10["University"], top10["Hiring_Rate_%"], color="#2980b9")
axes[0].set_title("Top 10 Universities — Highest Hiring Rate")
axes[0].set_xlabel("Hiring Rate (%)")
axes[0].invert_yaxis()

axes[1].barh(bottom10["University"], bottom10["Hiring_Rate_%"], color="#e74c3c")
axes[1].set_title("Bottom 10 Universities — Lowest Hiring Rate")
axes[1].set_xlabel("Hiring Rate (%)")
axes[1].invert_yaxis()

plt.tight_layout()
save(fig, "03_university_bias.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. EDUCATION LEVEL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("5. EDUCATION LEVEL IMPACT")
print(DIVIDER)

edu_grp = df.groupby("Education_Level").agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
edu_grp["Hiring_Rate_%"] = (edu_grp["Hired"] / edu_grp["Applicants"] * 100).round(1)
print(edu_grp.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(edu_grp["Education_Level"], edu_grp["Hiring_Rate_%"],
              color=sns.color_palette("Blues_d", len(edu_grp)))
ax.set_title("Hiring Rate by Education Level", fontsize=14, fontweight="bold")
ax.set_ylabel("Hiring Rate (%)")
ax.axhline(hire_rate, color="grey", linestyle="--", label=f"Overall avg {hire_rate:.1f}%")
ax.legend()
for bar, rate in zip(bars, edu_grp["Hiring_Rate_%"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3, f"{rate}%", ha="center")
plt.tight_layout()
save(fig, "04_education_level.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SKILLS SCORE vs HIRING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("6. SKILLS SCORE vs HIRING")
print(DIVIDER)

df["Skills_Bucket"] = pd.cut(df["Skills_Score"],
                              bins=[0, 40, 70, 100],
                              labels=["Low (0–40)", "Mid (40–70)", "High (70–100)"])
skills_grp = df.groupby("Skills_Bucket", observed=True).agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
skills_grp["Hiring_Rate_%"] = (skills_grp["Hired"] / skills_grp["Applicants"] * 100).round(1)
print(skills_grp.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Skills Score Analysis", fontsize=15, fontweight="bold")

axes[0].bar(skills_grp["Skills_Bucket"].astype(str), skills_grp["Hiring_Rate_%"],
            color=["#e74c3c", "#f39c12", "#2ecc71"])
axes[0].set_title("Hiring Rate by Skills Score Bucket")
axes[0].set_ylabel("Hiring Rate (%)")
for bar, rate in zip(axes[0].patches, skills_grp["Hiring_Rate_%"]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5, f"{rate}%", ha="center")

hired_colors = df["Final_Decision"].map(COLORS)
axes[1].scatter(df["Skills_Score"], df["Interview_Score"],
                c=hired_colors, alpha=0.4, s=20)
axes[1].set_title("Skills vs Interview Score\n(green=Selected, red=Rejected)")
axes[1].set_xlabel("Skills Score")
axes[1].set_ylabel("Interview Score")
patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()
           if k in ("Selected", "Rejected")]
axes[1].legend(handles=patches)

plt.tight_layout()
save(fig, "05_skills_score.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. INTERVIEW SCORE IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("7. INTERVIEW SCORE IMPACT")
print(DIVIDER)

corr_interview = df[["Interview_Score", "Hired"]].corr().loc["Interview_Score", "Hired"]
print(f"Correlation (Interview_Score vs Hired): {corr_interview:.4f}")

df["Interview_Bucket"] = pd.cut(df["Interview_Score"],
                                 bins=[0, 40, 70, 100],
                                 labels=["Low (0–40)", "Mid (40–70)", "High (70–100)"])
intv_grp = df.groupby("Interview_Bucket", observed=True).agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
intv_grp["Hiring_Rate_%"] = (intv_grp["Hired"] / intv_grp["Applicants"] * 100).round(1)
print(intv_grp.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Interview Score Analysis", fontsize=15, fontweight="bold")

axes[0].bar(intv_grp["Interview_Bucket"].astype(str), intv_grp["Hiring_Rate_%"],
            color=["#e74c3c", "#f39c12", "#2ecc71"])
axes[0].set_title("Hiring Rate by Interview Score Bucket")
axes[0].set_ylabel("Hiring Rate (%)")
for bar, rate in zip(axes[0].patches, intv_grp["Hiring_Rate_%"]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5, f"{rate}%", ha="center")

sns.boxplot(data=df, x="Final_Decision", y="Interview_Score",
            palette={"Selected": "#2ecc71", "Rejected": "#e74c3c"}, ax=axes[1])
axes[1].set_title("Interview Score Distribution by Decision")
axes[1].set_xlabel("Final Decision")
axes[1].set_ylabel("Interview Score")

plt.tight_layout()
save(fig, "06_interview_score.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPERIENCE vs HIRING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("8. EXPERIENCE vs HIRING")
print(DIVIDER)

df["Exp_Bucket"] = pd.cut(df["Experience_Years"],
                           bins=[-1, 1, 4, 100],
                           labels=["0–1 yrs", "2–4 yrs", "5+ yrs"])
exp_grp = df.groupby("Exp_Bucket", observed=True).agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
exp_grp["Hiring_Rate_%"] = (exp_grp["Hired"] / exp_grp["Applicants"] * 100).round(1)
print(exp_grp.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(exp_grp["Exp_Bucket"].astype(str), exp_grp["Hiring_Rate_%"],
        marker="o", linewidth=2.5, color="#8e44ad", markersize=9)
ax.fill_between(exp_grp["Exp_Bucket"].astype(str), exp_grp["Hiring_Rate_%"],
                alpha=0.15, color="#8e44ad")
ax.set_title("Hiring Rate by Experience Level", fontsize=14, fontweight="bold")
ax.set_ylabel("Hiring Rate (%)")
ax.set_xlabel("Experience Bracket")
for x_val, y_val in zip(exp_grp["Exp_Bucket"].astype(str), exp_grp["Hiring_Rate_%"]):
    ax.annotate(f"{y_val}%", (x_val, y_val), textcoords="offset points",
                xytext=(0, 10), ha="center")
plt.tight_layout()
save(fig, "07_experience.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. JOB ROLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("9. JOB ROLE ANALYSIS")
print(DIVIDER)

role_grp = df.groupby("Job_Applied").agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
role_grp["Hiring_Rate_%"] = (role_grp["Hired"] / role_grp["Applicants"] * 100).round(1)
role_grp = role_grp.sort_values("Hiring_Rate_%", ascending=False)
print(role_grp.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("coolwarm", len(role_grp))
ax.barh(role_grp["Job_Applied"], role_grp["Hiring_Rate_%"], color=colors)
ax.set_title("Hiring Rate by Job Role", fontsize=14, fontweight="bold")
ax.set_xlabel("Hiring Rate (%)")
ax.axvline(hire_rate, color="grey", linestyle="--", label=f"Overall avg {hire_rate:.1f}%")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
save(fig, "08_job_role.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. COMPANY-WISE HIRING BIAS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("10. COMPANY-WISE HIRING BIAS")
print(DIVIDER)

comp_grp = df.groupby("Company").agg(
    Applicants=("Hired", "count"),
    Hired=("Hired", "sum")
).reset_index()
comp_grp["Hiring_Rate_%"] = (comp_grp["Hired"] / comp_grp["Applicants"] * 100).round(1)
comp_grp = comp_grp[comp_grp["Applicants"] >= 5].sort_values("Hiring_Rate_%", ascending=False)
print(comp_grp.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 7))
norm = plt.Normalize(comp_grp["Hiring_Rate_%"].min(), comp_grp["Hiring_Rate_%"].max())
colors = plt.cm.RdYlGn(norm(comp_grp["Hiring_Rate_%"]))
ax.barh(comp_grp["Company"], comp_grp["Hiring_Rate_%"], color=colors)
ax.set_title("Company-wise Hiring Rate (color = rate intensity)", fontsize=14, fontweight="bold")
ax.set_xlabel("Hiring Rate (%)")
ax.axvline(hire_rate, color="grey", linestyle="--", label=f"Overall avg {hire_rate:.1f}%")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
save(fig, "09_company_bias.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. MACHINE LEARNING PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("11. MACHINE LEARNING PREDICTION")
print(DIVIDER)

# Encode categoricals
le_gender = LabelEncoder()
le_edu    = LabelEncoder()

df_ml = df.copy()
df_ml["Gender_enc"]    = le_gender.fit_transform(df_ml["Gender"])
df_ml["Education_enc"] = le_edu.fit_transform(df_ml["Education_Level"])

FEATURES = ["Gender_enc", "Education_enc", "CGPA",
            "Experience_Years", "Skills_Score", "Interview_Score"]
TARGET   = "Hired"

X = df_ml[FEATURES]
y = df_ml[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":             xgb.XGBClassifier(n_estimators=200, random_state=42,
                                              eval_metric="logloss", verbosity=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "Accuracy":  accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "F1":        f1_score(y_test, preds, zero_division=0),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["Rejected", "Selected"]))

results_df = pd.DataFrame(results).T.round(4)
print("\nModel Comparison:")
print(results_df)

# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(results_df))
width = 0.2
metrics = ["Accuracy", "Precision", "Recall", "F1"]
palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
for i, (metric, color) in enumerate(zip(metrics, palette)):
    ax.bar(x_pos + i * width, results_df[metric], width, label=metric, color=color, alpha=0.85)
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(results_df.index, rotation=10)
ax.set_ylim(0, 1.1)
ax.set_title("ML Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
save(fig, "10_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("12. FEATURE IMPORTANCE ANALYSIS")
print(DIVIDER)

feature_labels = ["Gender", "Education", "CGPA",
                  "Experience", "Skills Score", "Interview Score"]

rf_model  = models["Random Forest"]
xgb_model = models["XGBoost"]

rf_imp  = rf_model.feature_importances_
xgb_imp = xgb_model.feature_importances_

fi_df = pd.DataFrame({
    "Feature":           feature_labels,
    "Random Forest":     rf_imp,
    "XGBoost":           xgb_imp,
    "Average":           (rf_imp + xgb_imp) / 2,
}).sort_values("Average", ascending=False)
print(fi_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Feature Importance Analysis", fontsize=15, fontweight="bold")

for ax, col, color, title in zip(
        axes,
        ["Random Forest", "XGBoost", "Average"],
        ["#2980b9", "#e67e22", "#8e44ad"],
        ["Random Forest", "XGBoost", "Average (RF + XGB)"]):
    sub = fi_df.sort_values(col, ascending=True)
    ax.barh(sub["Feature"], sub[col], color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Importance")

plt.tight_layout()
save(fig, "11_feature_importance.png")

# Gender specifically
gender_importance_rf  = rf_imp[0]
gender_importance_xgb = xgb_imp[0]
print(f"\nGender importance — Random Forest : {gender_importance_rf:.4f}")
print(f"Gender importance — XGBoost       : {gender_importance_xgb:.4f}")
if (gender_importance_rf + gender_importance_xgb) / 2 > 0.05:
    print("⚠ Gender has notable predictive weight → potential encoded bias.")
else:
    print("✓ Gender importance is low — direct gender signal may be minimal.")

# ══════════════════════════════════════════════════════════════════════════════
# 13. FAIRNESS METRICS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("13. FAIRNESS METRICS")
print(DIVIDER)

# Use RF predictions on full dataset for fairness audit
df_ml["RF_Pred"] = rf_model.predict(X)

for group_col in ["Gender", "Education_Level"]:
    print(f"\n  --- {group_col} ---")
    grp = df_ml.groupby(group_col).agg(
        Actual_Rate   =("Hired",   "mean"),
        Predicted_Rate=("RF_Pred", "mean"),
        Count         =("Hired",   "count"),
    ).reset_index()
    grp["Actual_Rate_%"]    = (grp["Actual_Rate"]    * 100).round(1)
    grp["Predicted_Rate_%"] = (grp["Predicted_Rate"] * 100).round(1)
    print(grp[[group_col, "Count", "Actual_Rate_%", "Predicted_Rate_%"]].to_string(index=False))

# Demographic parity for Gender using model predictions
gender_pred_rates = df_ml.groupby("Gender")["RF_Pred"].mean()
if "Male" in gender_pred_rates and "Female" in gender_pred_rates:
    dp_diff_pred  = abs(gender_pred_rates["Male"] - gender_pred_rates["Female"]) * 100
    di_ratio_pred = gender_pred_rates["Female"] / gender_pred_rates["Male"]
    print(f"\n  Model Fairness (RF Predictions):")
    print(f"  Demographic Parity Difference : {dp_diff_pred:.1f}%")
    print(f"  Disparate Impact Ratio (F/M)  : {di_ratio_pred:.3f}  "
          f"{'⚠ Possible discrimination (DI < 0.8)' if di_ratio_pred < 0.8 else '✓ Within fair range'}")

# Fairness summary chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fairness Metrics — Actual vs Predicted Hiring Rate", fontsize=14, fontweight="bold")

grp_g = df_ml.groupby("Gender").agg(
    Actual=("Hired", "mean"),
    Predicted=("RF_Pred", "mean")
).reset_index()

x_pos = np.arange(len(grp_g))
w = 0.3
axes[0].bar(x_pos - w/2, grp_g["Actual"] * 100,    w, label="Actual",    color="#3498db", alpha=0.85)
axes[0].bar(x_pos + w/2, grp_g["Predicted"] * 100, w, label="Predicted", color="#e67e22", alpha=0.85)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(grp_g["Gender"])
axes[0].set_title("Gender: Actual vs Predicted Hiring Rate")
axes[0].set_ylabel("Rate (%)")
axes[0].legend()

grp_e = df_ml.groupby("Education_Level").agg(
    Actual=("Hired", "mean"),
    Predicted=("RF_Pred", "mean")
).reset_index()

x_pos2 = np.arange(len(grp_e))
axes[1].bar(x_pos2 - w/2, grp_e["Actual"] * 100,    w, label="Actual",    color="#3498db", alpha=0.85)
axes[1].bar(x_pos2 + w/2, grp_e["Predicted"] * 100, w, label="Predicted", color="#e67e22", alpha=0.85)
axes[1].set_xticks(x_pos2)
axes[1].set_xticklabels(grp_e["Education_Level"], rotation=10)
axes[1].set_title("Education: Actual vs Predicted Hiring Rate")
axes[1].set_ylabel("Rate (%)")
axes[1].legend()

plt.tight_layout()
save(fig, "12_fairness_metrics.png")

# ══════════════════════════════════════════════════════════════════════════════
# 14. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("14. CORRELATION HEATMAP")
print(DIVIDER)

numeric_cols = ["CGPA", "Experience_Years", "Skills_Score", "Interview_Score", "Hired"]
corr_matrix  = df[numeric_cols].corr()
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap (Numeric Features vs Hired)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "13_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("ANALYSIS COMPLETE — SUMMARY")
print(DIVIDER)
print(f"Output directory : {OUT_DIR}")
print(f"Charts saved     : 13 PNG files")
print()
print("Key Findings:")
print(f"  • Overall hiring rate          : {hire_rate:.1f}%")
print(f"  • Gender DP Difference (actual): {dp_diff:.1f}%  "
      f"({'Bias detected' if dp_diff > 5 else 'Low disparity'})")
print(f"  • Disparate Impact Ratio       : {di_ratio:.3f}  "
      f"({'⚠ Discriminatory' if di_ratio < 0.8 else '✓ Fair'})")
print(f"  • Best ML model                : {results_df['F1'].idxmax()}  "
      f"(F1={results_df['F1'].max():.3f})")
print(f"  • Top feature (RF)             : {feature_labels[np.argmax(rf_imp)]}")
print(f"  • Gender importance (avg)      : {(gender_importance_rf + gender_importance_xgb)/2:.4f}")
print(DIVIDER)
