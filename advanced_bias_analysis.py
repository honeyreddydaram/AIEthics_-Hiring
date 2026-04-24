"""
Advanced AI Ethics – Hiring Bias Analysis
Covers: Fairness Metrics, SHAP, LIME, Counterfactual, Proxy Discrimination,
        Structural Bias, Model Bias Audit, Adversarial Debiasing
"""

import os, warnings, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/honey/Downloads/AI Ethics/archive_contents/final_bias_hiring_dataset.csv"
OUT_DIR   = "/Users/honey/Downloads/AI Ethics/hiring_bias_contents/hiring_bias_output"
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG   = "#0f1117"
CARD_BG   = "#1a1a2e"
sns.set_theme(style="darkgrid", font_scale=1.1)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [saved] {name}")

DIVIDER = "=" * 65

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Hired"] = (df["Final_Decision"] == "Selected").astype(int)

le_g = LabelEncoder(); le_e = LabelEncoder(); le_u = LabelEncoder()
df["Gender_enc"]    = le_g.fit_transform(df["Gender"])
df["Education_enc"] = le_e.fit_transform(df["Education_Level"])
df["University_enc"]= le_u.fit_transform(df["University"])

FEATURES = ["Gender_enc","Education_enc","CGPA","Experience_Years","Skills_Score","Interview_Score"]
LABELS   = ["Gender","Education","CGPA","Experience","Skills Score","Interview Score"]
X  = df[FEATURES]
y  = df["Hired"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf  = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
xgb_m = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0).fit(X_train, y_train)
lr  = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)

results_store = {}  # collect all numbers for MD report

# ══════════════════════════════════════════════════════════════════════════════
# A1. INTERSECTIONAL BIAS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("A1. INTERSECTIONAL BIAS")
print(DIVIDER)

# Gender × Education
ge = df.groupby(["Gender","Education_Level"]).agg(
    Count=("Hired","count"), Hired=("Hired","sum")).reset_index()
ge["Rate_%"] = (ge["Hired"]/ge["Count"]*100).round(1)
print("Gender × Education:\n", ge.to_string(index=False))
results_store["intersectional_gender_edu"] = ge.to_dict(orient="records")

# Gender × Top-5 Universities by volume
top_unis = df["University"].value_counts().head(8).index.tolist()
df_top   = df[df["University"].isin(top_unis)]
gu = df_top.groupby(["Gender","University"]).agg(
    Count=("Hired","count"), Hired=("Hired","sum")).reset_index()
gu["Rate_%"] = (gu["Hired"]/gu["Count"]*100).round(1)
print("\nGender × University (top 8):\n", gu.to_string(index=False))
results_store["intersectional_gender_uni"] = gu.to_dict(orient="records")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Intersectional Bias Analysis", color="white", fontsize=15, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

pivot_ge = ge.pivot(index="Education_Level", columns="Gender", values="Rate_%")
pivot_ge.plot(kind="bar", ax=axes[0], color=["#e91e8c","#3498db"], alpha=0.85)
axes[0].set_title("Hiring Rate: Gender × Education", color="white")
axes[0].set_ylabel("Hiring Rate (%)", color="white")
axes[0].set_xlabel("Education Level", color="white")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(facecolor=CARD_BG, labelcolor="white")

pivot_gu = gu.pivot(index="University", columns="Gender", values="Rate_%")
pivot_gu.plot(kind="bar", ax=axes[1], color=["#e91e8c","#3498db"], alpha=0.85)
axes[1].set_title("Hiring Rate: Gender × University (top 8)", color="white")
axes[1].set_ylabel("Hiring Rate (%)", color="white")
axes[1].set_xlabel("", color="white")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(facecolor=CARD_BG, labelcolor="white")

plt.tight_layout()
save(fig, "A1_intersectional_bias.png")

# ══════════════════════════════════════════════════════════════════════════════
# A2. EQUAL OPPORTUNITY DIFFERENCE & EQUALIZED ODDS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("A2. EQUAL OPPORTUNITY DIFFERENCE & EQUALIZED ODDS")
print(DIVIDER)

df_test = X_test.copy()
df_test["Gender"]  = df["Gender"].iloc[X_test.index].values
df_test["Hired"]   = y_test.values
df_test["RF_Pred"] = rf.predict(X_test)

def fairness_metrics(group_col, pos_label=1):
    rows = []
    for grp, sub in df_test.groupby(group_col):
        tn, fp, fn, tp = confusion_matrix(sub["Hired"], sub["RF_Pred"], labels=[0,1]).ravel()
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0   # True Positive Rate (Equal Opportunity)
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0   # False Positive Rate
        fnr = fn/(fn+tp) if (fn+tp)>0 else 0   # False Negative Rate
        rows.append({"Group":grp,"TPR":round(tpr,4),"FPR":round(fpr,4),"FNR":round(fnr,4),
                     "TP":tp,"FP":fp,"FN":fn,"TN":tn})
    return pd.DataFrame(rows)

fm = fairness_metrics("Gender")
print(fm.to_string(index=False))

if len(fm) == 2:
    tpr_diff = abs(fm["TPR"].iloc[0] - fm["TPR"].iloc[1])
    fpr_diff = abs(fm["FPR"].iloc[0] - fm["FPR"].iloc[1])
    fnr_diff = abs(fm["FNR"].iloc[0] - fm["FNR"].iloc[1])
    print(f"\nEqual Opportunity Difference (TPR gap) : {tpr_diff:.4f}  {'⚠ Unequal' if tpr_diff>0.05 else '✓ Fair'}")
    print(f"Equalized Odds — FPR gap               : {fpr_diff:.4f}  {'⚠ Unequal' if fpr_diff>0.05 else '✓ Fair'}")
    print(f"Equalized Odds — FNR gap               : {fnr_diff:.4f}  {'⚠ Unequal' if fnr_diff>0.05 else '✓ Fair'}")
    results_store["equal_opportunity"] = {
        "TPR_Female": fm[fm.Group=="Female"]["TPR"].values[0],
        "TPR_Male":   fm[fm.Group=="Male"]["TPR"].values[0],
        "TPR_diff":   round(tpr_diff,4),
        "FPR_diff":   round(fpr_diff,4),
        "FNR_diff":   round(fnr_diff,4),
    }

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Equal Opportunity & Equalized Odds by Gender", color="white", fontsize=14, fontweight="bold")

metrics_plot = ["TPR","FPR","FNR"]
titles = ["True Positive Rate\n(Equal Opportunity)",
          "False Positive Rate\n(Equalized Odds)",
          "False Negative Rate\n(Equalized Odds)"]
colors_g = {"Female":"#e91e8c","Male":"#3498db"}

for ax, metric, title in zip(axes, metrics_plot, titles):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    bars = ax.bar(fm["Group"], fm[metric],
                  color=[colors_g[g] for g in fm["Group"]], alpha=0.85)
    ax.set_title(title, color="white", fontsize=11)
    ax.set_ylabel("Rate", color="white")
    for bar, val in zip(bars, fm[metric]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", color="white", fontsize=11)

plt.tight_layout()
save(fig, "A2_equal_opportunity_equalized_odds.png")

# ══════════════════════════════════════════════════════════════════════════════
# A3. CALIBRATION TEST
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("A3. CALIBRATION TEST")
print(DIVIDER)

rf_proba = rf.predict_proba(X_test)[:,1]
df_cal   = X_test.copy()
df_cal["Gender"] = df["Gender"].iloc[X_test.index].values
df_cal["Hired"]  = y_test.values
df_cal["Proba"]  = rf_proba

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Calibration Test — Is Predicted Probability Reliable Per Gender?",
             color="white", fontsize=14, fontweight="bold")

cal_results = {}
for ax, gender, color in zip(axes, ["Female","Male"], ["#e91e8c","#3498db"]):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    sub = df_cal[df_cal["Gender"]==gender]
    frac_pos, mean_pred = calibration_curve(sub["Hired"], sub["Proba"], n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2.5, markersize=8, label=gender)
    ax.plot([0,1],[0,1],"--", color="grey", linewidth=1.5, label="Perfect calibration")
    ax.set_title(f"Calibration — {gender}", color="white", fontsize=12)
    ax.set_xlabel("Mean Predicted Probability", color="white")
    ax.set_ylabel("Fraction Actually Hired", color="white")
    ax.legend(facecolor=CARD_BG, labelcolor="white")
    cal_results[gender] = {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}
    print(f"{gender}: mean_pred={np.round(mean_pred,2)}, frac_pos={np.round(frac_pos,2)}")

results_store["calibration"] = cal_results
plt.tight_layout()
save(fig, "A3_calibration_test.png")

# ══════════════════════════════════════════════════════════════════════════════
# B1. SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("B1. SHAP VALUES")
print(DIVIDER)

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# For binary: shap_values is list [class0, class1] or shape (n,f,2)
if isinstance(shap_values, list):
    sv = shap_values[1]          # class=1 (Selected)
elif shap_values.ndim == 3:
    sv = shap_values[:,:,1]
else:
    sv = shap_values

mean_abs_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({"Feature": LABELS, "Mean_SHAP": mean_abs_shap}).sort_values("Mean_SHAP", ascending=False)
print(shap_df.to_string(index=False))
results_store["shap_mean"] = shap_df.to_dict(orient="records")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("SHAP Values — Feature Impact on Hiring Prediction", color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

# Bar summary
colors_shap = ["#e74c3c" if f=="Gender" else "#3498db" for f in shap_df["Feature"]]
axes[0].barh(shap_df["Feature"], shap_df["Mean_SHAP"], color=colors_shap, alpha=0.85)
axes[0].set_title("Mean |SHAP| per Feature\n(red = protected attribute)", color="white")
axes[0].set_xlabel("Mean |SHAP Value|", color="white")
axes[0].invert_yaxis()
for bar, val in zip(axes[0].patches, shap_df["Mean_SHAP"]):
    axes[0].text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", color="white", fontsize=10)

# Beeswarm-style scatter (manual)
feature_order = shap_df["Feature"].tolist()
feat_idx      = [LABELS.index(f) for f in feature_order]
for i, (fi, fname) in enumerate(zip(feat_idx, feature_order)):
    jitter   = np.random.uniform(-0.2, 0.2, len(sv))
    col_vals = X_test.iloc[:, fi].values
    norm     = (col_vals - col_vals.min()) / (np.ptp(col_vals) + 1e-9)
    scatter_c = plt.cm.RdBu(norm)
    axes[1].scatter(sv[:, fi], i + jitter, c=scatter_c, alpha=0.35, s=12)
axes[1].set_yticks(range(len(feature_order)))
axes[1].set_yticklabels(feature_order, color="white")
axes[1].set_xlabel("SHAP Value (impact on prediction)", color="white")
axes[1].set_title("SHAP Distribution per Feature\n(blue=low value, red=high value)", color="white")
axes[1].axvline(0, color="white", linewidth=0.8, linestyle="--")

plt.tight_layout()
save(fig, "B1_shap_values.png")

# ══════════════════════════════════════════════════════════════════════════════
# B2. LIME — Local Explainability
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("B2. LIME — LOCAL EXPLAINABILITY")
print(DIVIDER)

lime_explainer = LimeTabularExplainer(
    X_train.values, feature_names=LABELS,
    class_names=["Rejected","Selected"], mode="classification", random_state=42)

# Pick 2 borderline cases (predicted proba near 0.5)
proba_all = rf.predict_proba(X_test)[:,1]
borderline_idx = np.argsort(np.abs(proba_all - 0.5))[:2]

lime_results = []
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("LIME — Local Explanations for Borderline Decisions", color="white", fontsize=14, fontweight="bold")

for ax, b_idx in zip(axes, borderline_idx):
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    instance  = X_test.iloc[b_idx].values
    exp       = lime_explainer.explain_instance(instance, rf.predict_proba, num_features=6)
    exp_list  = exp.as_list()
    feat_names= [e[0] for e in exp_list]
    feat_vals = [e[1] for e in exp_list]
    bar_cols  = ["#2ecc71" if v > 0 else "#e74c3c" for v in feat_vals]
    ax.barh(feat_names, feat_vals, color=bar_cols, alpha=0.85)
    actual    = df["Final_Decision"].iloc[X_test.index[b_idx]]
    pred_prob = proba_all[b_idx]
    ax.set_title(f"Borderline Case #{b_idx}\nActual: {actual} | Pred Prob: {pred_prob:.2f}",
                 color="white", fontsize=11)
    ax.set_xlabel("LIME Weight (green=towards Selected)", color="white")
    ax.axvline(0, color="white", linewidth=0.8)
    lime_results.append({"case_idx": int(b_idx), "actual": actual,
                         "pred_prob": round(float(pred_prob),3), "features": exp_list})
    print(f"Case {b_idx}: Actual={actual}, Prob={pred_prob:.3f}")
    for e in exp_list:
        print(f"   {e[0]:35s} → {e[1]:+.4f}")

results_store["lime"] = lime_results
plt.tight_layout()
save(fig, "B2_lime_explanations.png")

# ══════════════════════════════════════════════════════════════════════════════
# B3. COUNTERFACTUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("B3. COUNTERFACTUAL ANALYSIS — Gender Flip")
print(DIVIDER)

df_cf = X_test.copy()
df_cf["Original_Gender"]  = df["Gender"].iloc[X_test.index].values
df_cf["Original_Pred"]    = rf.predict(X_test)
df_cf["Original_Prob"]    = rf.predict_proba(X_test)[:,1]

# Flip gender (0→1, 1→0)
X_flipped = X_test.copy()
X_flipped["Gender_enc"] = 1 - X_test["Gender_enc"]
df_cf["Flipped_Pred"]   = rf.predict(X_flipped)
df_cf["Flipped_Prob"]   = rf.predict_proba(X_flipped)[:,1]
df_cf["Decision_Changed"] = (df_cf["Original_Pred"] != df_cf["Flipped_Pred"]).astype(int)
df_cf["Prob_Delta"]       = (df_cf["Flipped_Prob"] - df_cf["Original_Prob"]).round(4)

total_changed     = df_cf["Decision_Changed"].sum()
pct_changed       = total_changed / len(df_cf) * 100
female_df         = df_cf[df_cf["Original_Gender"]=="Female"]
male_df           = df_cf[df_cf["Original_Gender"]=="Male"]
female_changed    = female_df["Decision_Changed"].mean() * 100
male_changed      = male_df["Decision_Changed"].mean() * 100

print(f"Total decisions changed on gender flip : {total_changed} / {len(df_cf)} ({pct_changed:.1f}%)")
print(f"% Female→Male decisions changed        : {female_changed:.1f}%")
print(f"% Male→Female decisions changed        : {male_changed:.1f}%")
print(f"Avg prob delta (Female→Male)           : {female_df['Prob_Delta'].mean():.4f}")
print(f"Avg prob delta (Male→Female)           : {male_df['Prob_Delta'].mean():.4f}")
results_store["counterfactual"] = {
    "total_changed": int(total_changed), "pct_changed": round(pct_changed,2),
    "female_pct_changed": round(female_changed,2), "male_pct_changed": round(male_changed,2),
    "avg_prob_delta_female": round(float(female_df["Prob_Delta"].mean()),4),
    "avg_prob_delta_male":   round(float(male_df["Prob_Delta"].mean()),4),
}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Counterfactual Analysis — What If Gender Was Flipped?", color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

# Changed vs unchanged
changed_counts = df_cf["Decision_Changed"].value_counts().sort_index()
axes[0].bar(["No Change","Decision\nChanged"], changed_counts.values,
            color=["#2ecc71","#e74c3c"], alpha=0.85)
axes[0].set_title("Decision Stability on Gender Flip", color="white")
axes[0].set_ylabel("Count", color="white")
for bar, val in zip(axes[0].patches, changed_counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 str(val), ha="center", color="white", fontsize=12)

# Per gender
axes[1].bar(["Female→Male","Male→Female"], [female_changed, male_changed],
            color=["#e91e8c","#3498db"], alpha=0.85)
axes[1].set_title("% Changed by Original Gender", color="white")
axes[1].set_ylabel("% Decisions Changed", color="white")
for bar, val in zip(axes[1].patches, [female_changed, male_changed]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%", ha="center", color="white", fontsize=12)

# Probability delta distribution
axes[2].hist(female_df["Prob_Delta"], bins=25, alpha=0.6, color="#e91e8c", label="Female→Male")
axes[2].hist(male_df["Prob_Delta"],   bins=25, alpha=0.6, color="#3498db", label="Male→Female")
axes[2].axvline(0, color="white", linestyle="--")
axes[2].set_title("Probability Change Distribution", color="white")
axes[2].set_xlabel("Δ Predicted Probability", color="white")
axes[2].set_ylabel("Frequency", color="white")
axes[2].legend(facecolor=CARD_BG, labelcolor="white")

plt.tight_layout()
save(fig, "B3_counterfactual_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# B4. THRESHOLD SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("B4. THRESHOLD SENSITIVITY")
print(DIVIDER)

thresholds  = np.arange(0.3, 0.75, 0.05)
thresh_data = []
for t in thresholds:
    preds = (rf_proba >= t).astype(int)
    for gender in ["Female","Male"]:
        mask = df_cf["Original_Gender"] == gender
        sub_y    = y_test.values[mask]
        sub_pred = preds[mask]
        hr = sub_pred.mean() * 100
        f1 = f1_score(sub_y, sub_pred, zero_division=0)
        thresh_data.append({"Threshold":round(t,2),"Gender":gender,
                             "Hiring_Rate":hr,"F1":f1})

thresh_df = pd.DataFrame(thresh_data)
print(thresh_df.to_string(index=False))
results_store["threshold_sensitivity"] = thresh_data

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Threshold Sensitivity — How Cutoff Affects Gender Parity", color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

for gender, color in [("Female","#e91e8c"),("Male","#3498db")]:
    sub = thresh_df[thresh_df["Gender"]==gender]
    axes[0].plot(sub["Threshold"], sub["Hiring_Rate"], "o-", color=color, linewidth=2.5, label=gender)
    axes[1].plot(sub["Threshold"], sub["F1"], "o-", color=color, linewidth=2.5, label=gender)

axes[0].set_title("Hiring Rate vs Threshold", color="white")
axes[0].set_xlabel("Decision Threshold", color="white"); axes[0].set_ylabel("Hiring Rate (%)", color="white")
axes[0].legend(facecolor=CARD_BG, labelcolor="white"); axes[0].axvline(0.5, color="grey", linestyle="--")

axes[1].set_title("F1 Score vs Threshold", color="white")
axes[1].set_xlabel("Decision Threshold", color="white"); axes[1].set_ylabel("F1 Score", color="white")
axes[1].legend(facecolor=CARD_BG, labelcolor="white"); axes[1].axvline(0.5, color="grey", linestyle="--")

plt.tight_layout()
save(fig, "B4_threshold_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════════
# C1. PROXY DISCRIMINATION — University ↔ Gender
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("C1. PROXY DISCRIMINATION — University & Gender")
print(DIVIDER)

uni_gender = df.groupby("University")["Gender_enc"].mean().reset_index()
uni_gender.columns = ["University","Pct_Male"]
uni_gender["Pct_Female"] = 1 - uni_gender["Pct_Male"]
uni_gender = uni_gender.merge(
    df.groupby("University").agg(Count=("Hired","count"), HireRate=("Hired","mean")).reset_index(),
    on="University")
uni_gender = uni_gender[uni_gender["Count"] >= 5]
corr_univ_gender = uni_gender["Pct_Male"].corr(uni_gender["HireRate"])
print(f"Correlation — % Male in University vs Hiring Rate: {corr_univ_gender:.4f}")
results_store["proxy_uni_gender_corr"] = round(corr_univ_gender,4)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Proxy Discrimination — University as Privilege Proxy", color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

axes[0].scatter(uni_gender["Pct_Male"]*100, uni_gender["HireRate"]*100,
                alpha=0.6, color="#f39c12", s=40)
axes[0].set_xlabel("% Male Students in University", color="white")
axes[0].set_ylabel("Hiring Rate (%)", color="white")
axes[0].set_title(f"University Gender Composition vs Hiring Rate\n(corr={corr_univ_gender:.3f})", color="white")

# CGPA vs hiring by gender
for gender, color in [("Female","#e91e8c"),("Male","#3498db")]:
    sub = df[df["Gender"]==gender]
    bins = pd.cut(sub["CGPA"], bins=5)
    cgpa_hr = sub.groupby(bins, observed=True)["Hired"].mean()*100
    axes[1].plot(range(len(cgpa_hr)), cgpa_hr.values, "o-", color=color, linewidth=2.5, label=gender)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels([f"Q{i+1}" for i in range(5)], color="white")
axes[1].set_title("CGPA vs Hiring Rate by Gender\n(do women need higher CGPA?)", color="white")
axes[1].set_xlabel("CGPA Quintile (Q1=lowest)", color="white")
axes[1].set_ylabel("Hiring Rate (%)", color="white")
axes[1].legend(facecolor=CARD_BG, labelcolor="white")

plt.tight_layout()
save(fig, "C1_proxy_discrimination.png")

# ══════════════════════════════════════════════════════════════════════════════
# C2. SKILLS SCORE PARITY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("C2. SKILLS SCORE PARITY")
print(DIVIDER)

df["Skills_Q"] = pd.qcut(df["Skills_Score"], q=4, labels=["Q1 Low","Q2","Q3","Q4 High"])
sp = df.groupby(["Skills_Q","Gender"], observed=True).agg(
    Count=("Hired","count"), Hired=("Hired","sum")).reset_index()
sp["Rate_%"] = (sp["Hired"]/sp["Count"]*100).round(1)
print(sp.to_string(index=False))
results_store["skills_parity"] = sp.to_dict(orient="records")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
pivot_sp = sp.pivot(index="Skills_Q", columns="Gender", values="Rate_%")
pivot_sp.plot(kind="bar", ax=ax, color=["#e91e8c","#3498db"], alpha=0.85)
ax.set_title("Skills Score Parity — Equal Skills, Equal Chances?", color="white", fontsize=13, fontweight="bold")
ax.set_xlabel("Skills Score Quartile", color="white"); ax.set_ylabel("Hiring Rate (%)", color="white")
ax.tick_params(axis="x", rotation=0)
ax.legend(facecolor=CARD_BG, labelcolor="white")
plt.tight_layout()
save(fig, "C2_skills_parity.png")

# ══════════════════════════════════════════════════════════════════════════════
# D1. QUALIFIED-BUT-REJECTED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("D1. QUALIFIED-BUT-REJECTED ANALYSIS")
print(DIVIDER)

qualified = df[(df["Skills_Score"] > 70) & (df["Interview_Score"] > 70)]
qbr = qualified.groupby("Gender").agg(
    Total=("Hired","count"), Hired=("Hired","sum")).reset_index()
qbr["Rejected"]    = qbr["Total"] - qbr["Hired"]
qbr["Rejected_%"]  = (qbr["Rejected"]/qbr["Total"]*100).round(1)
qbr["Hired_%"]     = (qbr["Hired"]/qbr["Total"]*100).round(1)
print(f"Qualified candidates (Skills>70 AND Interview>70): {len(qualified)}")
print(qbr.to_string(index=False))
results_store["qualified_but_rejected"] = qbr.to_dict(orient="records")
results_store["qualified_total"] = len(qualified)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Qualified-but-Rejected Analysis\n(Skills > 70 AND Interview > 70)",
             color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

x = np.arange(len(qbr)); w = 0.35
axes[0].bar(x-w/2, qbr["Hired"],    w, label="Hired",    color="#2ecc71", alpha=0.85)
axes[0].bar(x+w/2, qbr["Rejected"], w, label="Rejected", color="#e74c3c", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(qbr["Gender"].values, color="white")
axes[0].set_title("Qualified Candidates — Hired vs Rejected", color="white")
axes[0].set_ylabel("Count", color="white")
axes[0].legend(facecolor=CARD_BG, labelcolor="white")

axes[1].bar(qbr["Gender"], qbr["Rejected_%"],
            color=["#e91e8c","#3498db"], alpha=0.85)
axes[1].set_title("% Rejected Among Qualified Candidates", color="white")
axes[1].set_ylabel("Rejection Rate (%)", color="white")
for bar, val in zip(axes[1].patches, qbr["Rejected_%"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val}%", ha="center", color="white", fontsize=12)

plt.tight_layout()
save(fig, "D1_qualified_but_rejected.png")

# ══════════════════════════════════════════════════════════════════════════════
# D2. OVER-QUALIFICATION PENALTY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("D2. OVER-QUALIFICATION PENALTY")
print(DIVIDER)

df["CGPA_Q"] = pd.qcut(df["CGPA"], q=4, labels=["Q1 Low","Q2","Q3","Q4 High"])
oq = df.groupby(["CGPA_Q","Education_Level"], observed=True).agg(
    Count=("Hired","count"), Hired=("Hired","sum")).reset_index()
oq["Rate_%"] = (oq["Hired"]/oq["Count"]*100).round(1)
print(oq.to_string(index=False))
results_store["overqualification"] = oq.to_dict(orient="records")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
pivot_oq = oq.pivot(index="CGPA_Q", columns="Education_Level", values="Rate_%")
pivot_oq.plot(kind="bar", ax=ax, color=["#f39c12","#9b59b6"], alpha=0.85)
ax.set_title("Over-Qualification Penalty — High CGPA Doesn't Always Help",
             color="white", fontsize=13, fontweight="bold")
ax.set_xlabel("CGPA Quartile", color="white"); ax.set_ylabel("Hiring Rate (%)", color="white")
ax.tick_params(axis="x", rotation=0)
ax.legend(facecolor=CARD_BG, labelcolor="white")
plt.tight_layout()
save(fig, "D2_overqualification_penalty.png")

# ══════════════════════════════════════════════════════════════════════════════
# D3. JOB ROLE SEGREGATION BY GENDER
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("D3. JOB ROLE SEGREGATION BY GENDER")
print(DIVIDER)

role_gender = df.groupby(["Job_Applied","Gender"]).size().unstack(fill_value=0)
role_gender["Total"]    = role_gender.sum(axis=1)
role_gender["Pct_Female"] = (role_gender["Female"]/role_gender["Total"]*100).round(1)
role_gender["Pct_Male"]   = (role_gender["Male"]/role_gender["Total"]*100).round(1)
role_gender = role_gender.sort_values("Pct_Female", ascending=False)
print(role_gender[["Female","Male","Total","Pct_Female","Pct_Male"]].to_string())
results_store["job_role_segregation"] = role_gender.reset_index().to_dict(orient="records")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
roles = role_gender.index.tolist()
y_pos = np.arange(len(roles))
ax.barh(y_pos, role_gender["Pct_Female"], color="#e91e8c", alpha=0.85, label="Female")
ax.barh(y_pos, -role_gender["Pct_Male"],  color="#3498db", alpha=0.85, label="Male")
ax.set_yticks(y_pos); ax.set_yticklabels(roles, color="white", fontsize=10)
ax.set_xlabel("← Male %  |  Female % →", color="white")
ax.set_title("Job Role Gender Segregation (Butterfly Chart)", color="white", fontsize=13, fontweight="bold")
ax.axvline(0, color="white", linewidth=1)
ax.legend(facecolor=CARD_BG, labelcolor="white")
# fix x-tick labels to show absolute values
xticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(x))}%" for x in xticks], color="white")
plt.tight_layout()
save(fig, "D3_job_role_segregation.png")

# ══════════════════════════════════════════════════════════════════════════════
# E1. BIASED VS DEBIASED MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("E1. BIASED vs DEBIASED MODEL")
print(DIVIDER)

# Biased = includes Gender_enc
# Debiased = remove Gender_enc
FEATURES_DEBIASED = ["Education_enc","CGPA","Experience_Years","Skills_Score","Interview_Score"]
LABELS_DEBIASED   = ["Education","CGPA","Experience","Skills Score","Interview Score"]

X_db        = df[FEATURES_DEBIASED]
Xdb_train, Xdb_test = X_db.iloc[X_train.index], X_db.iloc[X_test.index]

rf_debiased = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xdb_train, y_train)

def get_metrics(model, Xt, yt):
    preds = model.predict(Xt)
    return {
        "Accuracy":  round(accuracy_score(yt, preds),4),
        "Precision": round(precision_score(yt, preds, zero_division=0),4),
        "Recall":    round(recall_score(yt, preds, zero_division=0),4),
        "F1":        round(f1_score(yt, preds, zero_division=0),4),
        "AUC":       round(roc_auc_score(yt, model.predict_proba(Xt)[:,1]),4),
    }

biased_metrics   = get_metrics(rf,          X_test,  y_test)
debiased_metrics = get_metrics(rf_debiased, Xdb_test, y_test)

print("Biased model (with Gender):")
print(biased_metrics)
print("\nDebiased model (Gender removed):")
print(debiased_metrics)

# Compute gender disparity for each model
def hiring_rate_by_gender(model, Xt, gender_series):
    preds = model.predict(Xt)
    rows  = []
    for g in ["Female","Male"]:
        mask = (gender_series == g).values
        rows.append({"Gender":g, "Hire_Rate_%": round(preds[mask].mean()*100,1)})
    return pd.DataFrame(rows)

gender_test = df["Gender"].iloc[X_test.index].reset_index(drop=True)
bias_disp   = hiring_rate_by_gender(rf,          X_test,  gender_test)
debias_disp = hiring_rate_by_gender(rf_debiased, Xdb_test, gender_test)
print("\nGender disparity — Biased model:"); print(bias_disp.to_string(index=False))
print("\nGender disparity — Debiased model:"); print(debias_disp.to_string(index=False))
results_store["biased_vs_debiased"] = {
    "biased": biased_metrics, "debiased": debiased_metrics,
    "bias_disparity": bias_disp.to_dict(orient="records"),
    "debias_disparity": debias_disp.to_dict(orient="records"),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Biased vs Debiased Model Comparison", color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

metric_keys = ["Accuracy","Precision","Recall","F1","AUC"]
x_pos = np.arange(len(metric_keys)); w = 0.3
axes[0].bar(x_pos-w/2, [biased_metrics[m]   for m in metric_keys], w, label="Biased",   color="#e74c3c", alpha=0.85)
axes[0].bar(x_pos+w/2, [debiased_metrics[m] for m in metric_keys], w, label="Debiased", color="#2ecc71", alpha=0.85)
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(metric_keys, color="white", rotation=15)
axes[0].set_ylim(0,1.1); axes[0].set_ylabel("Score", color="white")
axes[0].set_title("Performance: Biased vs Debiased", color="white")
axes[0].legend(facecolor=CARD_BG, labelcolor="white")

# Side-by-side hiring disparity
for i, (disp_df, label, color) in enumerate([
        (bias_disp,"Biased","#e74c3c"), (debias_disp,"Debiased","#2ecc71")]):
    x = np.arange(len(disp_df)) + i*0.05
    axes[1].bar(np.arange(len(disp_df))*2 + i*0.7, disp_df["Hire_Rate_%"],
                0.6, label=label, color=color, alpha=0.85)
axes[1].set_xticks([0.35, 2.35]); axes[1].set_xticklabels(["Female","Male"], color="white")
axes[1].set_title("Gender Hiring Rate Disparity\nBiased vs Debiased", color="white")
axes[1].set_ylabel("Hiring Rate (%)", color="white")
axes[1].legend(facecolor=CARD_BG, labelcolor="white")

plt.tight_layout()
save(fig, "E1_biased_vs_debiased.png")

# ══════════════════════════════════════════════════════════════════════════════
# E2. FAIRNESS–ACCURACY TRADEOFF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("E2. FAIRNESS–ACCURACY TRADEOFF")
print(DIVIDER)

tradeoff = []
for alpha in np.arange(0, 1.05, 0.1):
    # Blend predictions: alpha=0 pure biased, alpha=1 pure debiased
    prob_biased   = rf.predict_proba(X_test)[:,1]
    prob_debiased = rf_debiased.predict_proba(Xdb_test)[:,1]
    blended_prob  = (1-alpha)*prob_biased + alpha*prob_debiased
    preds         = (blended_prob >= 0.5).astype(int)
    acc           = accuracy_score(y_test, preds)
    f1            = f1_score(y_test, preds, zero_division=0)
    female_mask   = (gender_test == "Female").values
    male_mask     = (gender_test == "Male").values
    hr_f  = preds[female_mask].mean()*100
    hr_m  = preds[male_mask].mean()*100
    dp    = abs(hr_f - hr_m)
    di    = (hr_f/hr_m) if hr_m > 0 else 1
    tradeoff.append({"Alpha":round(alpha,2),"Accuracy":round(acc,4),"F1":round(f1,4),
                     "DP_Diff":round(dp,2),"DI_Ratio":round(di,4)})

tradeoff_df = pd.DataFrame(tradeoff)
print(tradeoff_df.to_string(index=False))
results_store["fairness_accuracy_tradeoff"] = tradeoff

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Fairness–Accuracy Tradeoff\n(α=0 → Biased Model, α=1 → Debiased Model)",
             color="white", fontsize=14, fontweight="bold")

for ax in axes:
    ax.set_facecolor(CARD_BG); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

axes[0].plot(tradeoff_df["Alpha"], tradeoff_df["Accuracy"]*100, "o-", color="#3498db", linewidth=2.5, label="Accuracy")
axes[0].plot(tradeoff_df["Alpha"], tradeoff_df["F1"]*100,       "s-", color="#2ecc71", linewidth=2.5, label="F1 Score")
axes[0].set_xlabel("Debiasing Strength (α)", color="white"); axes[0].set_ylabel("Score (%)", color="white")
axes[0].set_title("Model Performance vs Debiasing Strength", color="white")
axes[0].legend(facecolor=CARD_BG, labelcolor="white")

axes[1].plot(tradeoff_df["Alpha"], tradeoff_df["DP_Diff"], "o-", color="#e74c3c",  linewidth=2.5, label="DP Difference (%)")
axes[1].plot(tradeoff_df["Alpha"], tradeoff_df["DI_Ratio"], "s-", color="#f39c12", linewidth=2.5, label="DI Ratio")
axes[1].axhline(0.8, color="grey", linestyle="--", label="DI=0.8 fairness threshold")
axes[1].set_xlabel("Debiasing Strength (α)", color="white"); axes[1].set_ylabel("Fairness Metric", color="white")
axes[1].set_title("Fairness Metrics vs Debiasing Strength", color="white")
axes[1].legend(facecolor=CARD_BG, labelcolor="white")

plt.tight_layout()
save(fig, "E2_fairness_accuracy_tradeoff.png")

# ══════════════════════════════════════════════════════════════════════════════
# E3. ADVERSARIAL DEBIASING (Approx: Gender Prediction Residual)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("E3. ADVERSARIAL DEBIASING")
print(DIVIDER)

# Step 1: Train hiring predictor WITHOUT gender
# Step 2: Train gender predictor on the hiring model's probabilities
# Step 3: If gender can be predicted from output → model encodes gender
# Step 4: Show that debiased model leaks less gender info

FEAT_NO_G = ["Education_enc","CGPA","Experience_Years","Skills_Score","Interview_Score"]
rf_nog = RandomForestClassifier(n_estimators=200, random_state=42).fit(
    df[FEAT_NO_G], y)

# Probabilities from both models on full dataset
prob_biased_all   = rf.predict_proba(df[FEATURES])[:,1]
prob_debiased_all = rf_nog.predict_proba(df[FEAT_NO_G])[:,1]
gender_y          = df["Gender_enc"]

# Can we predict gender from model output?
adv_biased   = LogisticRegression(max_iter=500).fit(
    prob_biased_all.reshape(-1,1), gender_y)
adv_debiased = LogisticRegression(max_iter=500).fit(
    prob_debiased_all.reshape(-1,1), gender_y)

adv_acc_biased   = cross_val_score(adv_biased,   prob_biased_all.reshape(-1,1),   gender_y, cv=5).mean()
adv_acc_debiased = cross_val_score(adv_debiased, prob_debiased_all.reshape(-1,1), gender_y, cv=5).mean()

print(f"Gender prediction accuracy from BIASED model output   : {adv_acc_biased:.4f}")
print(f"Gender prediction accuracy from DEBIASED model output : {adv_acc_debiased:.4f}")
print(f"Random baseline                                        : 0.5000")
results_store["adversarial_debiasing"] = {
    "biased_gender_leak": round(float(adv_acc_biased),4),
    "debiased_gender_leak": round(float(adv_acc_debiased),4),
    "random_baseline": 0.5,
}

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
labels_adv = ["Biased Model\n(with Gender)", "Debiased Model\n(no Gender)", "Random\nBaseline"]
vals_adv   = [adv_acc_biased, adv_acc_debiased, 0.5]
cols_adv   = ["#e74c3c","#2ecc71","#95a5a6"]
bars = ax.bar(labels_adv, vals_adv, color=cols_adv, alpha=0.85)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Gender Prediction Accuracy from Model Output", color="white")
ax.set_title("Adversarial Debiasing Test\nCan the model output reveal the applicant's gender?",
             color="white", fontsize=13, fontweight="bold")
ax.axhline(0.5, color="grey", linestyle="--", label="Random baseline")
ax.legend(facecolor=CARD_BG, labelcolor="white")
for bar, val in zip(bars, vals_adv):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.4f}", ha="center", color="white", fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "E3_adversarial_debiasing.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
json_path = os.path.join(OUT_DIR, "advanced_analysis_results.json")
with open(json_path, "w") as f:
    json.dump(results_store, f, indent=2, default=str)
print(f"\n[saved] advanced_analysis_results.json")

print(f"\n{DIVIDER}")
print("ALL ANALYSES COMPLETE")
print(DIVIDER)
print(f"Output: {OUT_DIR}")
print(f"Charts : A1–A3, B1–B4, C1–C2, D1–D3, E1–E3  (14 new PNGs)")
