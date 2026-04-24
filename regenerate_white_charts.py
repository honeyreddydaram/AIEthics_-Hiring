"""
Regenerates B3 and E2 charts with white background for the paper.
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/honey/Downloads/AI Ethics/archive_contents/final_bias_hiring_dataset.csv"
OUT_DIR   = "/Users/honey/Downloads/AI Ethics/hiring_bias_contents/hiring_bias_output"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  [saved] {name}")

# ── Load & prepare ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Hired"] = (df["Final_Decision"] == "Selected").astype(int)

le_g = LabelEncoder(); le_e = LabelEncoder()
df["Gender_enc"]    = le_g.fit_transform(df["Gender"])
df["Education_enc"] = le_e.fit_transform(df["Education_Level"])

FEATURES = ["Gender_enc","Education_enc","CGPA","Experience_Years","Skills_Score","Interview_Score"]
FEATURES_DEBIASED = ["Education_enc","CGPA","Experience_Years","Skills_Score","Interview_Score"]

X = df[FEATURES]
y = df["Hired"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

gender_test = df["Gender"].iloc[X_test.index].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# B3. COUNTERFACTUAL ANALYSIS — white background
# ══════════════════════════════════════════════════════════════════════════════
print("Generating B3_counterfactual_analysis.png ...")

df_cf = X_test.copy()
df_cf["Original_Gender"] = df["Gender"].iloc[X_test.index].values
df_cf["Original_Pred"]   = rf.predict(X_test)
df_cf["Original_Prob"]   = rf.predict_proba(X_test)[:,1]

X_flipped = X_test.copy()
X_flipped["Gender_enc"]   = 1 - X_test["Gender_enc"]
df_cf["Flipped_Pred"]     = rf.predict(X_flipped)
df_cf["Flipped_Prob"]     = rf.predict_proba(X_flipped)[:,1]
df_cf["Decision_Changed"] = (df_cf["Original_Pred"] != df_cf["Flipped_Pred"]).astype(int)
df_cf["Prob_Delta"]       = (df_cf["Flipped_Prob"] - df_cf["Original_Prob"]).round(4)

female_df      = df_cf[df_cf["Original_Gender"]=="Female"]
male_df        = df_cf[df_cf["Original_Gender"]=="Male"]
female_changed = female_df["Decision_Changed"].mean() * 100
male_changed   = male_df["Decision_Changed"].mean() * 100
total_changed  = df_cf["Decision_Changed"].sum()

fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")
fig.suptitle("Counterfactual Analysis: What If Gender Was Flipped?", fontsize=14, fontweight="bold")

# Panel 1: Changed vs unchanged
changed_counts = df_cf["Decision_Changed"].value_counts().sort_index()
axes[0].bar(["No Change", "Decision\nChanged"], changed_counts.values,
            color=["#2ecc71","#e74c3c"], alpha=0.85, edgecolor="white")
axes[0].set_title("Decision Stability on Gender Flip")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, changed_counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 str(val), ha="center", fontsize=12, fontweight="bold")

# Panel 2: Per gender change rate
axes[1].bar(["Female→Male","Male→Female"], [female_changed, male_changed],
            color=["#e91e8c","#3498db"], alpha=0.85, edgecolor="white")
axes[1].set_title("% Decisions Changed by Original Gender")
axes[1].set_ylabel("% Decisions Changed")
for bar, val in zip(axes[1].patches, [female_changed, male_changed]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

# Panel 3: Probability delta distribution
axes[2].hist(female_df["Prob_Delta"], bins=25, alpha=0.6, color="#e91e8c", label="Female→Male")
axes[2].hist(male_df["Prob_Delta"],   bins=25, alpha=0.6, color="#3498db", label="Male→Female")
axes[2].axvline(0, color="black", linestyle="--", linewidth=1.2)
axes[2].set_title("Probability Change Distribution")
axes[2].set_xlabel("Δ Predicted Probability")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
save(fig, "B3_counterfactual_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# E2. FAIRNESS–ACCURACY TRADEOFF — white background
# ══════════════════════════════════════════════════════════════════════════════
print("Generating E2_fairness_accuracy_tradeoff.png ...")

X_db       = df[FEATURES_DEBIASED]
Xdb_train  = X_db.iloc[X_train.index]
Xdb_test   = X_db.iloc[X_test.index]
rf_debiased = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xdb_train, y_train)

tradeoff = []
for alpha in np.arange(0, 1.05, 0.1):
    prob_b  = rf.predict_proba(X_test)[:,1]
    prob_db = rf_debiased.predict_proba(Xdb_test)[:,1]
    blended = (1-alpha)*prob_b + alpha*prob_db
    preds   = (blended >= 0.5).astype(int)
    acc     = accuracy_score(y_test, preds) * 100
    f1      = f1_score(y_test, preds, zero_division=0) * 100
    f_mask  = (gender_test == "Female").values
    m_mask  = (gender_test == "Male").values
    hr_f    = preds[f_mask].mean() * 100
    hr_m    = preds[m_mask].mean() * 100
    dp      = abs(hr_f - hr_m)
    di      = hr_f / hr_m if hr_m > 0 else 1
    tradeoff.append({"Alpha": round(alpha,2), "Accuracy": acc, "F1": f1,
                     "DP_Diff": dp, "DI_Ratio": di})

td = pd.DataFrame(tradeoff)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
fig.suptitle("Fairness-Accuracy Tradeoff\n(α=0: Biased Model, α=1: Debiased Model)",
             fontsize=14, fontweight="bold")

axes[0].plot(td["Alpha"], td["Accuracy"], "o-", color="#3498db", linewidth=2.5, label="Accuracy (%)")
axes[0].plot(td["Alpha"], td["F1"],       "s-", color="#2ecc71", linewidth=2.5, label="F1 Score (%)")
axes[0].set_xlabel("Debiasing Strength (α)")
axes[0].set_ylabel("Score (%)")
axes[0].set_title("Model Performance vs Debiasing Strength")
axes[0].legend()

axes[1].plot(td["Alpha"], td["DP_Diff"],  "o-", color="#e74c3c",  linewidth=2.5, label="DP Difference (%)")
axes[1].plot(td["Alpha"], td["DI_Ratio"], "s-", color="#f39c12",  linewidth=2.5, label="DI Ratio")
axes[1].axhline(0.8, color="grey", linestyle="--", linewidth=1.2, label="DI=0.8 fairness threshold")
axes[1].set_xlabel("Debiasing Strength (α)")
axes[1].set_ylabel("Fairness Metric")
axes[1].set_title("Fairness Metrics vs Debiasing Strength")
axes[1].legend()

plt.tight_layout()
save(fig, "E2_fairness_accuracy_tradeoff.png")

print("\nDone. Both charts saved with white background.")
