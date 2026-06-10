# 🛡️ Travel Insurance Purchase Prediction

> **End-to-End Machine Learning Classification Project**  
> Predict whether a customer will purchase travel insurance using demographic, financial, and behavioural features.

---

## 📌 Project Overview

This project builds a complete ML pipeline — from raw data to explainable predictions — on a real-world travel insurance dataset. It covers data cleaning, exploratory analysis, feature engineering, multi-model training, SHAP-based explainability, and a sample prediction system.

**Best result:** Decision Tree — ROC-AUC **0.7719**, Accuracy **80.9%**

---

## 📂 Repository Structure

```
travel-insurance-prediction/
│
├── travel_insurance_pipeline.py   # Full end-to-end ML pipeline
├── REPORT.md                      # Professional project report
├── README.md                      # This file
│
├── plots/                         # All EDA and model visualisations
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_numeric_distributions.png
│   ├── 04_boxplots.png
│   ├── 05_categorical_analysis.png
│   ├── 06_age_income_scatter.png
│   ├── 07_income_by_abroad.png
│   ├── 08_model_comparison.png
│   ├── 09_confusion_matrices.png
│   ├── 10_roc_curves.png
│   ├── 11_feature_importance.png
│   ├── 12_shap_bar.png
│   └── 13_shap_beeswarm.png
│
└── outputs/
    ├── model_comparison.csv       # All model metrics
    ├── feature_importance.csv     # Feature ranking
    └── sample_predictions.csv     # Sample customer predictions
```

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| Source | Travel agency customer database |
| Records | 1,987 customers |
| Features | 8 input features + 1 target |
| Target | `TravelInsurance` (0 = No, 1 = Yes) |
| Class balance | 64.3% No / 35.7% Yes |
| Missing values | None |

### Features

| Feature | Type | Description |
|---|---|---|
| Age | Integer | Customer age (25–35) |
| Employment Type | Categorical | Government / Private Sector |
| GraduateOrNot | Binary | College graduate (Yes/No) |
| AnnualIncome | Integer | Annual income in ₹ |
| FamilyMembers | Integer | Number of family members |
| ChronicDiseases | Binary | Has a chronic illness (1/0) |
| FrequentFlyer | Binary | Frequent air traveller (Yes/No) |
| EverTravelledAbroad | Binary | Has travelled internationally (Yes/No) |
| **TravelInsurance** | **Target** | **Purchased insurance (1/0)** |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/travel-insurance-prediction.git
cd travel-insurance-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap

# 4. Place the dataset
# Put TravelInsurancePrediction.csv in the project root
# Update DATA_PATH in travel_insurance_pipeline.py if needed

# 5. Run the pipeline
python travel_insurance_pipeline.py
```

---

## 🚀 Usage

### Run the full pipeline

```bash
python travel_insurance_pipeline.py
```

This will:
1. Load and inspect the dataset
2. Preprocess and encode features
3. Generate 7 EDA plots (saved to `plots/`)
4. Engineer 5 new features
5. Train 6 classification models
6. Generate 6 evaluation plots (confusion matrices, ROC curves, feature importance, SHAP)
7. Print model comparison and sample predictions
8. Save results CSVs to `outputs/`

### Predict for a new customer (Python)

```python
import pandas as pd
import joblib  # after saving the model with joblib.dump(best_mdl, 'model.pkl')

# Load model
model = joblib.load('model.pkl')

# Create customer profile
customer = pd.DataFrame([{
    "Age": 31,
    "AnnualIncome": 1200000,
    "FamilyMembers": 4,
    "ChronicDiseases": 0,
    "Employment Type": 1,       # 1 = Private
    "GraduateOrNot": 1,
    "FrequentFlyer": 1,
    "EverTravelledAbroad": 1,
    "IncomeGroup": 2,
    "IncomePerMember": 300000,
    "RiskScore": 1,
    "TravelPropensity": 2,
    "AgeGroup": 1
}])

prediction  = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]
print(f"Purchase: {'Yes' if prediction else 'No'} ({probability:.1%} probability)")
```

---

## 📈 Results

### Model Performance

| Model | Accuracy | F1 Score | ROC-AUC | CV-AUC (5-fold) |
|---|---|---|---|---|
| **Decision Tree ✅** | **80.9%** | **0.678** | **0.772** | 0.806 |
| Gradient Boosting | 80.2% | 0.672 | 0.772 | 0.808 |
| Random Forest | 80.7% | 0.678 | 0.767 | 0.807 |
| Logistic Regression | 72.4% | 0.613 | 0.766 | 0.773 |
| LightGBM | 78.4% | 0.664 | 0.766 | 0.812 |
| XGBoost | 78.4% | 0.664 | 0.764 | 0.810 |

### Top Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | AnnualIncome | 64.8% |
| 2 | FamilyMembers | 16.2% |
| 3 | Age | 10.7% |
| 4 | IncomePerMember *(engineered)* | 3.8% |
| 5 | TravelPropensity *(engineered)* | 3.0% |

---

## 🖼️ Key Visualisations

| Plot | Description |
|---|---|
| `01_target_distribution.png` | Class balance bar chart and pie chart |
| `02_correlation_heatmap.png` | Pairwise correlations across all encoded features |
| `03_numeric_distributions.png` | Histograms by insurance purchase status |
| `05_categorical_analysis.png` | Purchase rate per category for all categorical features |
| `08_model_comparison.png` | Side-by-side accuracy/F1/AUC bar chart for all models |
| `10_roc_curves.png` | ROC curves for all 6 models on the test set |
| `11_feature_importance.png` | Ranked feature importance from the best model |
| `12_shap_bar.png` | SHAP global mean importance (Explainable AI) |
| `13_shap_beeswarm.png` | SHAP beeswarm — directional feature impact |

---

## 💡 Key Business Insights

- **Annual income is the single most powerful predictor** (64.8% importance). Customers earning above ₹10L are disproportionately likely to buy.
- **International travel experience** is the best behavioural signal — it creates insurance awareness and willingness to pay.
- **Frequent flyers** are a secondary high-conversion segment.
- **Young, low-income customers** with no travel history are unlikely buyers — marketing resources should be redirected away from this segment.
- An **80/20 model** (income + travel propensity) can explain most of the variance, suggesting the dataset could be enriched with additional behavioural features for production deployment.

---

## 🔮 Future Enhancements

- [ ] Probability threshold tuning via Precision-Recall curve optimisation
- [ ] SMOTE / class-balanced resampling for improved recall
- [ ] Stacking ensemble (Decision Tree + Gradient Boosting + Logistic Regression)
- [ ] REST API deployment using FastAPI + Docker
- [ ] Integration with a CRM for real-time lead scoring
- [ ] Richer features: travel frequency, past claims, destination type, trip duration
- [ ] Streamlit dashboard for business-user interactive predictions

---

## 🛠️ Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=1.7
lightgbm>=4.0
shap>=0.42
```

---

## 📄 License

This project is developed for academic and portfolio purposes. Dataset credit to the original source (Kaggle / travel agency anonymised data).

---

*Built with Python · scikit-learn · XGBoost · LightGBM · SHAP*
