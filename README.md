# Ethical Challenges of Artificial Intelligence in Hiring and Recruitment Systems
### A Comprehensive Bias Audit Study

**Authors:** Honey Reddy Daram · Lekha Dakshinamurthy · Sathyadharini Srinivasan  
**Institution:** AI Systems, University of Florida, Gainesville, USA  
**Course:** AI Systems — Final Research Paper

---

## Overview

This project presents a comprehensive empirical bias audit of an AI-driven hiring system. While AI-powered recruitment tools promise efficiency and objectivity, they risk encoding and amplifying historical biases embedded in training data. This study investigates whether — and how — gender bias manifests in a machine learning hiring model, and evaluates the findings through three ethical frameworks: Utilitarian, Kantian, and Rawlsian.

The central finding of this research is a critical gap between surface-level fairness and hidden bias. The system satisfies current regulatory thresholds (Disparate Impact Ratio = 1.019), yet 24.8% of hiring decisions change when only gender is flipped. Removing gender from the model costs only 1.3% F1 score, demonstrating that fairness and accuracy are not in conflict.

---

## Research Questions

| # | Research Question |
|---|---|
| **RQ1** | Do AI hiring systems exhibit measurable gender bias at both surface and deeper analytical levels? |
| **RQ2** | How can explainability techniques (SHAP, LIME, counterfactual analysis) reveal hidden bias that standard fairness metrics miss? |
| **RQ3** | What approaches can improve fairness while maintaining predictive performance in AI recruitment systems? |

---

## Dataset

- **Size:** 2,000 synthetic job applicants
- **Roles:** 10 AI-related job categories across 67 companies
- **Features:** Gender, Education Level, University, CGPA, Experience Years, Skills Score, Interview Score
- **Target:** Binary hiring decision (Selected / Rejected)
- **Overall hiring rate:** 57.8% (Female: 58.3%, Male: 57.2%)

> **Why synthetic data?** No publicly available real-world dataset exists that combines interview scores, skills assessment scores, and final hiring decisions together. Companies do not release this data publicly due to privacy concerns and legal liability. Synthetic data is a standard tool in AI hiring bias research, allowing controlled isolation of bias mechanisms while simulating realistic hiring scenarios.

---

## Methodology

Analysis was structured across five phases:

```
Phase 1 — Descriptive Analysis
         Hiring rates by gender, education, university, job role, company

Phase 2 — Fairness Metrics
         Demographic Parity · Equal Opportunity · Equalized Odds · Calibration

Phase 3 — Explainability
         SHAP Values · LIME Local Explanations · Counterfactual Analysis · Threshold Sensitivity

Phase 4 — Proxy & Structural Bias
         University proxy test · Skills parity · Qualified-but-rejected · Job role segregation

Phase 5 — Model Bias Audit
         Biased vs Debiased model · Fairness-accuracy tradeoff · Adversarial debiasing
```

**Models used:** Logistic Regression · Random Forest (200 estimators) · XGBoost (200 estimators)  
**Split:** 80/20 stratified train-test  
**Tools:** Python 3.13 · scikit-learn · XGBoost · SHAP 0.51.0 · LIME 0.2.0.1 · pandas · matplotlib · seaborn

---

## Key Results

### Surface-Level Fairness (Misleading)

| Metric | Value | Verdict |
|--------|-------|---------|
| Female Hiring Rate | 58.3% | — |
| Male Hiring Rate | 57.2% | — |
| Demographic Parity Difference | 1.1% | ✅ Low |
| Disparate Impact Ratio (F/M) | 1.019 | ✅ Above EEOC threshold of 0.8 |

At the surface level, the system appears gender-fair and would pass current regulatory audits.

---

### Hidden Bias (Revealed by Deep Analysis)

| Analysis | Finding | Severity |
|----------|---------|----------|
| XGBoost feature importance | Gender = 16.5% (4th most important feature) | ⚠️ Concerning |
| Counterfactual flip | 24.8% of decisions change when only gender is flipped | 🔴 Significant |
| Asymmetry | Female→Male raises probability by +0.020; Male→Female lowers by only -0.005 | 🔴 Asymmetric |
| LIME (borderline cases) | Gender tips decision toward "Selected" for male candidates in both borderline cases | 🔴 Active bias |
| Equalized Odds (FPR gap) | 0.057 — exceeds the 0.05 fairness threshold | ⚠️ Unequal |
| Calibration | Model underestimates female candidates by up to 32 percentage points | ⚠️ Miscalibrated |

---

### Structural Bias

| Finding | Detail |
|---------|--------|
| University hiring rate range | 0% (HKUST) to 94.7% (ETH Zurich) — a 94.7pp gap driven by institutional prestige |
| Intersectional gap | Female Stanford applicant: 90.9% vs Female University of Adelaide: 31.2% (59.7pp gap) |
| Job role segregation | Women cluster in AI Ethics & Compliance (55.0%); men dominate Data Architecture (57.3%) |

---

### Debiasing Results

| Metric | Biased Model | Debiased Model | Change |
|--------|-------------|----------------|--------|
| F1 Score | 0.664 | 0.651 | -1.3% only |
| Male Hire Rate | 72.3% | 69.1% | -3.2% |
| Female Hire Rate | 72.8% | 72.2% | -0.6% |

**Key insight:** Removing gender costs only 1.3% F1 while eliminating a 3.2 percentage point invisible male advantage. Fairness and accuracy are demonstrably not in conflict.

---

## Ethical Analysis

This study evaluates its findings through three foundational ethical frameworks. The literature on AI ethics highlights a fundamental gap between technical fairness metrics and genuine ethical accountability — Barocas and Selbst (2016) warn that satisfying statistical parity does not resolve the underlying ethical problem, and Doshi-Velez and Kim (2017) argue that accountability requires not just measurable outcomes but explainable processes.

---

### 1. Utilitarian Analysis

**Framework:** An action is ethical if it produces the greatest good for the greatest number.

**Application to findings:**  
Classical utilitarianism might appear to justify this system: aggregate hiring rates are nearly equal, suggesting near-equal population-level welfare. However, this utilitarian defence collapses under scrutiny. The 24.8% of candidates whose decisions change on gender flip represent real individuals experiencing real harm — lost employment, foregone income, diminished career prospects — hidden within population statistics.

The debiasing experiment provides the decisive utilitarian evidence:
- Removing gender costs **1.3% F1** (negligible welfare loss)
- Removing gender eliminates a **3.2pp** invisible male hiring advantage (substantial welfare gain)

The welfare cost of maintaining bias vastly exceeds the welfare cost of removing it. The industry argument that "fairness costs performance" is not only empirically false in this dataset but represents a utilitarian miscalculation — it counts only one side of the ledger. A genuine utilitarian analysis unambiguously supports debiasing.

---

### 2. Kantian (Deontological) Analysis

**Framework:** Persons must be treated as ends in themselves, never merely as means. Actions must be universalisable.

**Application to findings:**  
Kant's categorical imperative applies in two formulations:

**First formulation (universalisability):** The maxim "use gender to influence hiring outcomes" cannot be universalised. No rational agent would consent to a world in which employment depends on gender rather than competence. The maxim contradicts the meritocratic basis that makes hiring decisions legitimate.

**Second formulation (Formula of Humanity):** The LIME analysis reveals the paradigmatic Kantian wrong. At the decision boundary — where a candidate's qualifications are neither clearly sufficient nor insufficient — the model uses gender as a tiebreaker. The individual is not evaluated as a person with unique qualities but is processed as an instance of a demographic category. This is treating persons as means to statistical optimisation.

**Critically**, this violation is not mitigated by aggregate fairness. Kant's ethics are individual, not statistical. Each gender-influenced decision is an independent moral failure, regardless of what population-level statistics show.

---

### 3. Rawlsian Analysis

**Framework:** Just institutions are those rational agents would design behind a "veil of ignorance," unaware of their own gender, class, or background. Inequalities are permissible only if they benefit the least advantaged.

**Application to findings:**  
Behind the veil of ignorance — not knowing whether you would be male or female, from an elite or non-elite university — no rational agent would design a system where:
- Stanford female applicants achieve 90.9% hiring rates vs 31.2% for University of Adelaide female applicants
- Women are channelled into AI Ethics & Compliance roles (55.0% female) while men dominate Data Architecture (57.3%)
- University prestige creates a 94.7 percentage point hiring rate range based on institutional affiliation rather than individual merit

Rawls's **difference principle** holds that inequalities are justified only if they benefit the least advantaged. Both the university prestige effect and occupational segregation systematically disadvantage those from non-elite institutions and women seeking technical roles, without producing any compensating benefit. There is no mechanism by which these inequalities make the least advantaged better off.

The Rawlsian critique is especially powerful because it addresses the **structural dimension** of bias that individual fairness metrics ignore entirely. Justice, for Rawls, is a property of social structures — not only of individual acts.

---

### Ethical Summary

| Framework | Verdict | Key Argument |
|-----------|---------|--------------|
| Utilitarian | ❌ Fails genuine test | Aggregate parity masks individual harm; debiasing costs nothing meaningful |
| Kantian | ❌ Violates categorical imperative | Gender used as tiebreaker reduces persons to statistical categories |
| Rawlsian | ❌ Violates difference principle | University prestige and job segregation disadvantage the least advantaged |
| Regulatory (EEOC / GDPR) | ⚠️ Technically compliant | Passes legal thresholds but violates the spirit of anti-discrimination law |

> **Core ethical finding:** A system can be legally compliant and statistically fair at the aggregate level while simultaneously violating the ethical rights of individual candidates. Current regulatory frameworks are insufficient to catch this class of harm.

---

## The Invisible Bias Problem

Our adversarial debiasing test reveals a critical governance failure: the biased model's gender effect is undetectable through output-layer audits (only 0.35% above random gender prediction accuracy), yet causes 24.8% of decisions to change on gender flip.

This means:
- Standard fairness audits would **pass** this system as compliant
- The EEOC four-fifths rule would **pass** this system as compliant  
- GDPR Article 22's right to explanation is **meaningless** if the bias cannot be surfaced by explanation tools

Mandatory counterfactual and SHAP-based auditing must become regulatory requirements, not optional best practices.

---

## The IDEAL Framework

Based on our findings, we propose **IDEAL** — a five-pillar framework for ethical AI recruitment:

| Pillar | Action |
|--------|--------|
| **I** — Inclusive Data | Remove protected attributes; audit remaining features for proxy correlations |
| **D** — Debiased Models | Mandatory feature exclusion; fairness-constrained training (Fairlearn / AIF360) |
| **E** — Explainability by Default | SHAP reports and counterfactual explanations for all rejected candidates |
| **A** — Auditing and Monitoring | Quarterly counterfactual audits; flag if >5% decisions change on gender flip |
| **L** — Legal Compliance | Align with EU AI Act; model cards; candidate appeal processes |

---

## Repository Structure

```
AIEthics_-Hiring/
│
├── hiring_bias_analysis.py          # Phase 1-3: Descriptive stats, ML models, fairness metrics
├── advanced_bias_analysis.py        # Phase 4-5: SHAP, LIME, counterfactual, debiasing
├── regenerate_white_charts.py       # Regenerates key charts with white background
│
├── archive_contents/
│   └── final_bias_hiring_dataset.csv   # 2,000 applicant dataset
│
└── hiring_bias_contents/hiring_bias_output/
    ├── 01_dataset_overview.png          # Dataset distribution
    ├── 02_gender_bias.png               # Hiring rate by gender
    ├── 03_university_bias.png           # Top/bottom universities
    ├── 04_education_level.png           # Education level impact
    ├── 05_skills_score.png              # Skills score analysis
    ├── 06_interview_score.png           # Interview score impact
    ├── 07_experience.png                # Experience vs hiring
    ├── 08_job_role.png                  # Job role hiring rates
    ├── 09_company_bias.png              # Company-wise hiring rates
    ├── 10_model_comparison.png          # ML model performance
    ├── 11_feature_importance.png        # Feature importance (RF + XGB)
    ├── 12_fairness_metrics.png          # Actual vs predicted fairness
    ├── 13_correlation_heatmap.png       # Feature correlations
    ├── A1_intersectional_bias.png       # Gender × Education/University
    ├── A2_equal_opportunity_equalized_odds.png  # TPR/FPR by gender
    ├── A3_calibration_test.png          # Calibration curves
    ├── B1_shap_values.png               # SHAP feature attribution
    ├── B2_lime_explanations.png         # LIME local explanations
    ├── B3_counterfactual_analysis.png   # Gender flip counterfactuals
    ├── B4_threshold_sensitivity.png     # Fairness vs decision threshold
    ├── C1_proxy_discrimination.png      # University as privilege proxy
    ├── C2_skills_parity.png             # Skills parity by gender
    ├── D1_qualified_but_rejected.png    # High-skill rejection analysis
    ├── D2_overqualification_penalty.png # CGPA × Education effects
    ├── D3_job_role_segregation.png      # Gender distribution by role
    ├── E1_biased_vs_debiased.png        # Model with vs without gender
    ├── E2_fairness_accuracy_tradeoff.png # Debiasing tradeoff curve
    ├── E3_adversarial_debiasing.png     # Gender leakage from outputs
    ├── feature_importance_detailed.png  # Detailed feature importance
    └── advanced_analysis_results.json  # All numerical results
```

---

## How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime
```

**2. Update file paths** in both scripts (lines 26-27 in `hiring_bias_analysis.py` and lines 28-29 in `advanced_bias_analysis.py`) to match your local path.

**3. Run the basic analysis:**
```bash
python hiring_bias_analysis.py
```

**4. Run the advanced analysis:**
```bash
python advanced_bias_analysis.py
```

**5. Regenerate white-background charts for the paper:**
```bash
python regenerate_white_charts.py
```

All charts are saved to the `hiring_bias_output/` directory.

---

## Key References

1. Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review*, 104(3), 671–732.
2. Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2), 153–163.
3. Dastin, J. (2018). Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*.
4. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv:1702.08608*.
5. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
6. Kant, I. (1785). *Groundwork of the Metaphysics of Morals*. (Translated by M. Gregor, Cambridge University Press, 1998).
7. Köchling, A., & Wehner, M. C. (2020). Discriminated by an algorithm. *Business Research*, 13, 795–848.
8. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
9. Raghavan, M., Barocas, S., Kleinberg, J., & Levy, K. (2020). Mitigating bias in algorithmic hiring. *ACM FAccT*.
10. Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.
11. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" *KDD*.
12. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box. *Harvard Journal of Law & Technology*, 31(2).
13. EU AI Act. (2024). *Official Journal of the European Union*.
14. EEOC. (1978). *Uniform Guidelines on Employee Selection Procedures*. 29 C.F.R. § 1607.

---

*University of Florida — AI Systems Course — April 2026*
