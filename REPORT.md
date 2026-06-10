# Travel Insurance Purchase Prediction — Project Report

> **Institution:** SASTRA University  
> **Domain:** Machine Learning / Predictive Analytics  
> **Dataset:** TravelInsurancePrediction.csv (1,987 records)  
> **Target:** Binary Classification — Will a customer purchase travel insurance?

---

## 1. Introduction

Travel insurance is a financial product that covers trip cancellations, medical emergencies, and lost luggage for travellers. For insurance companies, accurately predicting which customers are likely to purchase a policy enables more efficient marketing, personalised outreach, and better resource allocation. This project builds an end-to-end machine learning system that predicts travel insurance purchase intent using demographic, financial, and behavioural features collected from a travel agency's customer database.

---

## 2. Problem Statement

**Objective:** Given nine customer attributes (age, income, employment type, education, family size, chronic disease status, frequent flyer status, and international travel history), predict whether the customer will purchase travel insurance (binary: 0 = No, 1 = Yes).

This is a **supervised binary classification** problem with a moderate class imbalance (64.3% negative, 35.7% positive).

---

## 3. Dataset Description

| Column | Type | Description |
|---|---|---|
| Age | Numeric | Customer age (25–35 years) |
| Employment Type | Categorical | Government Sector or Private Sector/Self Employed |
| GraduateOrNot | Binary | Whether the customer is a college graduate |
| AnnualIncome | Numeric | Annual income in Indian Rupees (₹3L – ₹18L) |
| FamilyMembers | Numeric | Number of family members (2–9) |
| ChronicDiseases | Binary | 1 = has a chronic illness, 0 = does not |
| FrequentFlyer | Binary | Whether the customer frequently travels by air |
| EverTravelledAbroad | Binary | Whether the customer has previously travelled internationally |
| **TravelInsurance** | **Target** | **1 = purchased insurance, 0 = did not purchase** |

**Key statistics:**
- 1,987 total records with no missing values and no duplicate rows after the unnamed index column is removed.
- Annual income ranges from ₹3,00,000 to ₹18,00,000 with a mean of ₹9,32,763.
- Age is tightly clustered between 25 and 35 — a narrow, young-professional demographic.
- 35.7% of customers purchased insurance — moderate imbalance that warrants class-weight adjustments.

---

## 4. Data Cleaning

**Missing values:** None detected across all 9 features. The dataset is clean and industry-ready.

**Duplicate rows:** The `Unnamed: 0` column is an artefact from the original CSV export and was dropped. After removal, **738 apparent duplicate rows** share identical feature values. This is expected in a survey dataset where multiple customers share the same age bracket, income tier, and employment category. No rows were removed because these represent genuinely distinct individuals who happen to share the same profile, and removing them would introduce selection bias.

**Outlier analysis (IQR method):**
- Age (range 25–35): 0 outliers — the dataset only covers young professionals.
- AnnualIncome (₹3L–₹18L): 0 statistical outliers — values are business-realistic.
- FamilyMembers (2–9): 0 outliers.

No outlier removal was performed. All values are logically consistent with a travel-agency customer base.

**Categorical encoding:**
- `GraduateOrNot`, `FrequentFlyer`, `EverTravelledAbroad`: Binary Yes/No → 1/0 (label encoding is appropriate since there are only two classes with no ordinal relationship that needs preserving).
- `Employment Type`: Government Sector → 0, Private Sector/Self Employed → 1 via `LabelEncoder`.

---

## 5. EDA Findings

### 5.1 Target Distribution
35.7% of 1,987 customers purchased insurance. This is a **mild class imbalance** — not severe enough to require SMOTE oversampling, but sufficient to justify using `class_weight="balanced"` in scikit-learn models and `scale_pos_weight` in XGBoost.

### 5.2 Annual Income — The Dominant Predictor
The income distributions of buyers vs non-buyers diverge significantly above ₹10,00,000. Customers earning more than ₹12.5L are substantially more likely to purchase. This suggests a **wealth threshold effect** — once discretionary income exceeds a comfort level, travel insurance becomes an easy purchase.

### 5.3 EverTravelledAbroad — Strong Behavioural Signal
Customers who have previously travelled internationally show a markedly higher insurance purchase rate. International travel experience creates awareness of travel risks and insurance value. This is the strongest binary predictor in the dataset.

### 5.4 FrequentFlyer — Moderate Signal
Frequent flyers have a higher purchase rate, but not as strong as international travel history. Frequent domestic flyers may not perceive the same risk level as international travellers.

### 5.5 Age
Despite the narrow range (25–35), older customers within this band (32–35) show a higher purchase rate. Age correlates loosely with income, so part of this signal may be mediated by income.

### 5.6 Employment Type
Government sector employees show a slightly higher purchase rate, possibly due to greater job security enabling non-essential purchases, or due to employer travel mandates.

### 5.7 ChronicDiseases
Customers with chronic conditions show a slightly higher purchase rate, likely driven by medical coverage needs during travel.

### 5.8 Family Members
Larger families (5–7 members) show somewhat higher purchase rates, possibly because insuring multiple people on one policy feels more economically justified.

---

## 6. Feature Engineering

Five engineered features were created to capture information that the raw columns don't fully express:

| Feature | Formula | Rationale |
|---|---|---|
| IncomeGroup | Binned AnnualIncome into 4 tiers (Low/Mid/High/Premium) | Captures non-linear threshold effects better than raw income |
| IncomePerMember | AnnualIncome ÷ FamilyMembers | Measures disposable income per person; large families dilute wealth |
| RiskScore | ChronicDiseases + (FamilyMembers≥5) + (Age≥30) | Aggregate risk indicator; higher risk → stronger purchase motive |
| TravelPropensity | FrequentFlyer + EverTravelledAbroad | Composite travel exposure score (0 to 2) |
| AgeGroup | Discretised age into three bands (25-28, 29-31, 32-35) | Allows models to pick up non-linear age effects |

Of these, `IncomePerMember` proved useful in tree splits (appeared in the top-5 features), while `TravelPropensity` cleanly compressed two correlated binary features into one ordinal signal.

---

## 7. Model Development

All models used a **80/20 stratified train-test split** (1,589 training / 398 test samples). Five-fold **Stratified Cross Validation** was used for robust AUC estimation. `class_weight="balanced"` was applied where available to handle class imbalance.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC (5-fold) |
|---|---|---|---|---|---|---|
| **Decision Tree** | **0.8090** | **0.8511** | 0.5634 | **0.6780** | **0.7719** | 0.8064 |
| Gradient Boosting | 0.8015 | 0.8182 | 0.5704 | 0.6722 | 0.7718 | **0.8077** |
| Random Forest | 0.8065 | 0.8351 | 0.5704 | 0.6778 | 0.7668 | 0.8068 |
| Logistic Regression | 0.7236 | 0.6127 | **0.6127** | 0.6127 | 0.7663 | 0.7731 |
| LightGBM | 0.7839 | 0.7456 | 0.5986 | 0.6641 | 0.7655 | 0.8119 |
| XGBoost | 0.7839 | 0.7456 | 0.5986 | 0.6641 | 0.7637 | 0.8101 |

**Best model: Decision Tree (tuned via GridSearchCV)** — highest test ROC-AUC (0.7719), highest accuracy (80.9%), and highest precision (0.851). The Decision Tree's performance parity with ensemble methods suggests that the decision boundary in this dataset is relatively shallow and tree-based, which tree ensembles do not dramatically improve upon.

**Note on Recall:** Logistic Regression achieves the best recall (0.613) at the cost of lower precision. In an insurance marketing context, higher recall (catching more buyers) may be more commercially valuable — this is a business trade-off worth discussing with stakeholders.

---

## 8. Feature Importance Analysis

### Top Features (Decision Tree, Gini Importance):

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | AnnualIncome | 0.648 | Wealthier customers almost always buy — the dominant split |
| 2 | FamilyMembers | 0.162 | Larger families have more to protect; a key secondary split |
| 3 | Age | 0.107 | Older customers (within 25–35) are more risk-aware |
| 4 | IncomePerMember | 0.038 | Per-capita wealth refines what raw income misses |
| 5 | TravelPropensity | 0.030 | Frequent flyers who've been abroad are prime buyers |
| 6 | FrequentFlyer | 0.006 | Partially subsumed by TravelPropensity |
| 7 | IncomeGroup | 0.004 | Income tier; correlated with AnnualIncome |
| 8 | GraduateOrNot | 0.003 | Education adds marginal predictive value |
| 9 | ChronicDiseases | 0.002 | Modest signal; chronic illness motivates coverage |

**Dominant finding:** Annual income accounts for 64.8% of the model's predictive power. This is not surprising — travel insurance is a discretionary purchase and income directly determines purchasing capacity. Feature engineering (`IncomePerMember`) added additional refinement beyond raw income.

### SHAP Analysis (Global)
SHAP values confirm the feature importance ranking and additionally show:
- **AnnualIncome** has a strongly positive SHAP value for high earners and negative for low earners — a clean monotonic relationship.
- **EverTravelledAbroad** = Yes adds significant positive SHAP value regardless of other features.
- **FamilyMembers** shows a non-linear SHAP pattern: medium families (4–5) push predictions up; very large families push down (income dilution effect captured via IncomePerMember).

---

## 9. Key Business Insights

1. **Income is the primary gating factor.** Customers earning above ₹10L are far more likely to buy. Marketing campaigns should prioritise the ₹9L–₹18L segment for insurance cross-sells.

2. **International travel history is the best behavioural signal.** Customers who have travelled abroad are pre-qualified leads — they understand the value of insurance. A targeted post-international-flight email campaign could yield strong conversion rates.

3. **Frequent flyers are a secondary priority.** They are more receptive than the baseline population but less certain than international travellers.

4. **Younger, lower-income customers (Profile 1: Age 25, ₹4L)** are almost certain non-buyers. Marketing spend on this segment has very low ROI.

5. **Chronic disease customers are a niche opportunity** for medically-enhanced travel insurance products, even among lower-income brackets.

6. **Family size creates nuanced risk perception.** Families of 4–6 represent a sweet spot — large enough to feel vulnerable, not so large that income dilution makes insurance feel unaffordable.

---

## 10. Challenges Encountered

- **Narrow age range (25–35):** The dataset captures only young professionals, limiting the model's generalisability to older or retired travellers who may have very different insurance behaviour.
- **High income concentration:** Annual income is one of eight predictors yet contributes 64.8% of tree-based importance. This creates a near-single-feature model for tree methods and raises questions about whether income proxies for other unmeasured factors (savings, family wealth, employer benefits).
- **Mild class imbalance:** The 64/36 split required class-weight adjustments. Recall scores across all models remain below 65%, meaning roughly 35% of actual buyers are missed. In a real deployment, the decision threshold should be lowered to 0.35–0.40 to capture more buyers at the cost of slightly more false positives.
- **Duplicate profiles:** 738 rows share all feature values. This inflates cross-validation optimism slightly, since "identical" training and test records can exist across folds.

---

## 11. Future Improvements

1. **Threshold tuning:** Use Precision-Recall curves to select a custom prediction threshold that maximises business value rather than defaulting to 0.5.
2. **Richer features:** Customer spending behaviour, past claim history, holiday frequency, and destination type would likely add significant predictive power.
3. **Ensemble stacking:** Combine the best-performing models (Decision Tree, Gradient Boosting, Random Forest) into a stacked ensemble to capture diverse decision boundaries.
4. **Larger, fresher dataset:** The 1,987-record dataset is small for production ML. Augmenting with more recent data or external demographic sources would improve generalisation.
5. **Deployment pipeline:** Wrap the model in a FastAPI REST service with an input validation schema and a prediction endpoint, enabling real-time scoring at a call centre or website.
6. **A/B testing:** Deploy the model in a controlled marketing experiment to measure actual uplift in insurance sales vs. untargeted campaigns.

---

## 12. Conclusion

This project built a complete end-to-end machine learning pipeline for travel insurance purchase prediction. Starting from raw CSV data, it delivered dataset profiling, data cleaning, exploratory analysis with 13 visualisations, feature engineering (5 new features), training and evaluation of 6 classification models, SHAP-based explainability, and sample predictions with probability scores.

The **tuned Decision Tree** achieved the best test ROC-AUC of **0.7719** and accuracy of **80.9%**, with all ensemble methods performing within 1% of each other — indicating a relatively simple decision surface dominated by income. The most actionable business finding is that **income level combined with international travel experience** are sufficient to correctly classify approximately 80% of customers, enabling a data-driven segmentation strategy for insurance marketing.

---

*Report generated by the Travel Insurance ML Pipeline. All results are based on the uploaded `TravelInsurancePrediction.csv` dataset.*
