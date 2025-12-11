# CRRT Clot Formation Prediction: Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project develops a machine learning-powered clinical decision support system to predict clot formation in Continuous Renal Replacement Therapy (CRRT) circuits. The system combines XGBoost prediction with SHAP explainability and LLM-generated clinical recommendations to create a trustworthy tool that healthcare providers will actually use.

### 🎯 Key Objectives

1. **Predict clotting risk** using lab values and CRRT machine parameters
2. **Provide explainable predictions** via SHAP feature contributions that clinicians can validate
3. **Generate actionable recommendations** using LLM integration for clinical context
4. **Deploy as a web application** with FastAPI backend and React frontend

---

## 🥇 Key Achievements

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.74 |
| **Patient Observations** | 125,611 |
| **Features Engineered** | 60 |
| **Deployment-Ready Features** | 20 (99.55% performance retention) |

---

## 🥏 Clinical Context

**Problem**: CRRT circuit clotting causes:
- Treatment interruptions (2+ hour downtime per event)
- Increased healthcare costs (~$600-1000 per circuit replacement)
- Potential patient harm from therapy gaps
- Increased nursing workload

**Current Limitation**: Many predictive models in clinical settings suffer from an explainability gap. Providers routinely ask "Why is this patient flagged?" without receiving satisfying answers beyond "the model says so." This project addresses that trust deficit by combining prediction with SHAP-based explanations and LLM-generated clinical context.

---

## 📊 Dataset

- **Source**: MIMIC-IV database (deidentified ICU data)
- **Cohort**: Adult patients receiving CRRT
- **Observations**: 125,611 time points
- **Features**: 60 numeric features after preprocessing
  - Lab values (platelets, creatinine, BUN, phosphate, PTT, fibrinogen, etc.)
  - CRRT machine parameters (blood flow, citrate dose, filter pressure, effluent pressure)
  - Anticoagulation mode (one-hot encoded: heparin, citrate, none)
  - Temporal features (rate of change for key labs)
- **Target**: Binary clot formation (clots_corrected: 0=no clot, 1=clot)
- **Class Balance**: ~14% clot rate (addressed via class weighting)

### Data Preprocessing
- Removed features with >80% missing data
- Applied median imputation for remaining missing values
- Standardized features using StandardScaler
- **Grouped train/test split by circuit ID** to prevent temporal data leakage
- Removed `clots_increasing` feature (contained future information)

---

## 🧠 Methodology

### Addressing Data Leakage

A critical methodological improvement was implementing grouped train/test splits by circuit ID. This prevents the model from "seeing" future observations from the same CRRT circuit during training, which previously inflated performance metrics.

```python
# Grouped split prevents data leakage
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=circuit_ids))
```

### Model Comparison

| Model | Best Strategy | ROC-AUC | Avg Precision |
|-------|---------------|---------|---------------|
| **XGBoost** ⭐ | Original (Class Weighted) | **0.74** | **0.94** |
| Random Forest | SMOTE | 0.71 | 0.93 |
| Logistic Regression | SMOTE | 0.69 | 0.92 |

### Feature Importance (Top 20 by Gain)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | blood_flow | 0.047 |
| 2 | phosphate | 0.031 |
| 3 | filter_pressure | 0.029 |
| 4 | prefilter_replacement_rate | 0.028 |
| 5 | effluent_pressure | 0.027 |
| 6 | creatinine | 0.026 |
| 7 | mode_citrate | 0.024 |
| 8 | mode_heparin | 0.024 |
| 9 | postfilter_replacement_rate | 0.024 |
| 10 | effluent_bloodflow_ratio | 0.024 |

### Deployment Feasibility Analysis

Testing whether a reduced feature set maintains performance for practical deployment:

| Features | ROC-AUC | Performance Retention |
|----------|---------|----------------------|
| 60 (All) | 0.737 | — (baseline) |
| 20 (Top) | 0.734 | **99.55%** ✅ |

**Conclusion**: Top 20 features achieve near-identical performance, making manual entry deployment feasible.

---

## 🎨 Explainability: SHAP Integration

The system uses SHAP (SHapley Additive exPlanations) TreeExplainer to provide patient-level feature contributions:

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(patient_data)

# Visualize top contributing factors
shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_values[0]
}).sort_values(by='shap_value', key=abs, ascending=False)
```

### Example Output
```
Patient Risk Score: 67% (HIGH RISK)

Top Contributing Factors:
  filter_pressure ↑ (285 mmHg)    ████████████  +0.18
  mode_citrate (active)           ████████      +0.12
  phosphate ↑ (6.8 mg/dL)         ██████        +0.09
  blood_flow ↓ (180 mL/min)       ████          +0.06
  PTT ↓ (42 sec)                  ███           -0.04
```

---

## 🚀 Deployment Architecture

### Backend (FastAPI on Render)
- RESTful API for predictions
- SHAP explanations endpoint
- OpenAI API integration for clinical recommendations
- Model served via pickle/joblib

### Frontend (React on Vercel)
- Risk gauge visualization
- SHAP waterfall chart for feature contributions
- LLM-generated clinical recommendations panel
- Responsive design for bedside use

### LLM Integration
Natural language synthesis of predictions using OpenAI API:
```
"This patient shows elevated clotting risk (67%) primarily driven by 
high filter pressure and active citrate anticoagulation. Consider 
evaluating circuit patency and adjusting replacement fluid rates. 
The elevated phosphate level may indicate metabolic disturbance 
requiring attention."
```

---

## 📈 Key Results

### Model Performance (After Leakage Correction)
- **ROC-AUC**: 0.74 (realistic clinical performance)
- **Average Precision**: 0.94
- **Recall at 50% threshold**: 0.87 (catches 87% of clots)

### Confusion Matrix Insights
- Balanced sensitivity/specificity trade-off
- Tunable threshold for clinical preference (sensitivity vs. alert fatigue)

---

## 🛠️ Technology Stack

### Core ML Libraries
- **scikit-learn** (1.7.2): Preprocessing, evaluation
- **XGBoost** (2.0.0): Gradient boosting classifier
- **SHAP** (0.50.0): Model explainability
- **pandas** (2.3.3): Data manipulation
- **imbalanced-learn** (0.11.0): SMOTE resampling

### Web Application
- **FastAPI**: Backend API framework
- **React**: Frontend framework
- **Render**: Backend hosting
- **Vercel**: Frontend hosting
- **OpenAI API**: LLM recommendations

### Development Environment
- **Python**: 3.13+
- **Node.js**: 18+
- **MIMIC-IV**: Clinical database

---

## 📁 Repository Structure

```
crrt-clotting-prediction/
├── README.md
├── requirements.txt
├── prediction_model.ipynb        # Main analysis notebook
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── model/
│   │   ├── xgb_model.pkl         # Trained XGBoost model
│   │   └── scaler.pkl            # Feature scaler
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RiskGauge.jsx
│   │   │   ├── SHAPChart.jsx
│   │   │   └── Recommendations.jsx
│   │   └── App.jsx
│   └── package.json
└── outputs/
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── shap_summary.png
    └── feature_importance.png
```

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt
```
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.50.0
matplotlib>=3.7.2
seaborn>=0.12.2
imbalanced-learn>=0.11.0
fastapi>=0.100.0
uvicorn>=0.23.0
openai>=1.0.0
```

### Running the Analysis
```bash
# Launch Jupyter
jupyter notebook

# Open the main notebook
prediction_model.ipynb
```

### Running the Web Application
```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📊 Reproducing Results

### Train the XGBoost Model
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit

# Grouped split to prevent leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=circuit_ids))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)
```

### Generate SHAP Explanations
```python
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
```

---

## 🎓 Academic Context

This project was completed as part of **BME 580: Biomedical Data Science** and **DTI 530: Technology Core** at Duke University's Master of Engineering in Design, Technology, & Innovation program.

### Presented At
- **Duke Health Data Science Poster Showcase** (December 2025)

### Learning Objectives Addressed
1. ✅ Applied supervised learning (classification) to real clinical data
2. ✅ Addressed temporal data leakage through grouped splitting
3. ✅ Implemented SHAP explainability for clinical trust
4. ✅ Deployed full-stack web application (FastAPI + React)
5. ✅ Integrated LLM for clinical recommendation generation

---

## 🔮 Future Work

- [ ] Prospective validation with real-time Duke Health data
- [ ] Epic EHR integration for automated feature extraction
- [ ] A/B testing: explainable interface vs. traditional alerts
- [ ] Uncertainty quantification for model confidence
- [ ] Multi-site validation study

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The MIMIC-IV data is subject to separate data use agreements and cannot be redistributed.

---

## 📚 References

1. Johnson, A.E.W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

3. Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.

4. Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.

---

## 📄 Version History

- **v2.0.0** (December 2025)
  - Deployed web application (FastAPI + React)
  - SHAP explainability integration
  - LLM-generated clinical recommendations
  - Addressed temporal data leakage
  - Feature selection for deployment feasibility

- **v1.0.0** (November 2025)
  - Initial XGBoost model training
  - PCA and K-means analysis
  - Interface design sketches