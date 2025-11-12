# CRRT Clot Formation Prediction: Clinical Decision Support Interface

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project develops a machine learning-powered clinical decision support interface to predict clot formation in Continuous Renal Replacement Therapy (CRRT) circuits. The goal is to create a trustworthy, explainable tool that healthcare providers will actually use—avoiding the pitfalls of "black box" AI systems that get dismissed in clinical settings.

### 🎯 Key Objectives

1. **Predict clotting risk** using lab values and CRRT machine parameters
2. **Provide explainable predictions** that clinicians can validate against their judgment
3. **Design a practical interface** that balances algorithmic accuracy with real-world usability
4. **Address deployment feasibility** for clinical implementation

---

## 🏥 Clinical Context

**Problem**: CRRT circuit clotting causes:
- Treatment interruptions (2+ hour downtime per event)
- Increased healthcare costs (~$600-1000 per circuit replacement)
- Potential patient harm from therapy gaps
- Increased nursing workload

**Current Limitation**: Our hospital's Sepsis Watch model, while technically successful, suffers from an explainability gap. Providers routinely ask "Why is this patient flagged?" without receiving satisfying answers beyond "the model says so." This project aims to solve that trust deficit in a different dimension with CRRT. 

---

## 📊 Dataset

- **Source**: MIMIC-IV database (deidentified ICU data)
- **Cohort**: Adult patients receiving CRRT
- **Observations**: 125,611 time points
- **Features**: 57 numeric features after preprocessing
  - Lab values (platelets, creatinine, BUN, phosphate, etc.)
  - CRRT machine parameters (blood flow, citrate dose, filter pressure)
  - Temporal features (prior clot history, rate of change)
- **Target**: Binary clot formation (clots_corrected: 0=no clot, 1=clot)
- **Class Balance**: 9.54% clot rate (imbalanced but workable)

### Data Preprocessing
- Removed 14 features with >80% missing data
- Excluded 6 non-numeric categorical features
- Applied median imputation for remaining missing values
- Standardized features using StandardScaler
- Train/test split: 80/20 (stratified)

---

## 🧠 Methodology

### Supervised Learning Models

Tested four algorithms with four resampling strategies each:

| Model | Best Strategy | ROC-AUC | Avg Precision |
|-------|---------------|---------|---------------|
| Logistic Regressio | |
| **XGBoost** ⭐ | Original (Imbalanced) | **0.9904** | **0.8891** |
| Random Forest | SMOTE | 0.9507 | 0.7437 |
| Logistic Regression | SMOTE | 0.9165 | 0.7249 |

**Winner**: XGBoost with original imbalanced data achieved near-perfect discrimination.

### Unsupervised Learning Analysis

#### PCA (Principal Component Analysis)
- **Finding**: High-dimensional problem
- PC1 explains only 6% of variance (no dominant factor)
- 30 components needed for 80% variance
- 37 components needed for 90% variance
- **Implication**: Cannot simplify to 10-15 features without performance loss

#### K-means Clustering
- **Finding**: Two distinct patient phenotypes
- **Cluster 0 (Low-Risk)**: 7.4% clot rate, 69% of patients
- **Cluster 1 (High-Risk)**: 14.4% clot rate, 31% of patients
- Statistical significance: χ² = 1212.29, p < 0.0001
- **Separation pattern**: Kidney dysfunction (elevated creatinine, BUN)
- **Implication**: Pattern recognition can validate predictions

### Feature Importance (Top 10)

1. **prior_clots_count** (33.0%) - History of clotting events
2. **phosphate** (5.6%) - Electrolyte imbalance indicator
3. **creatinine** (3.9%) - Kidney function marker
4. **bun** (3.2%) - Kidney function marker
5. **platelet** (2.8%) - Coagulation factor
6. **effluent_pressure** (2.5%) - Circuit mechanical stress
7. **ldh** (2.3%) - Cell damage marker
8. **heparin_dose** (2.1%) - Anticoagulation level
9. **potassium** (1.9%) - Electrolyte balance
10. **ptt** (1.8%) - Coagulation time

---

## 🎨 Interface Design

### Visualization 1: Risk Gauge
- **Purpose**: Immediate situational awareness
- **Display**: "87% - HIGH RISK (95th percentile)"
- **Benefit**: Quick assessment without cognitive overload

### Visualization 2: Feature Contribution Bar Chart
- **Purpose**: Explainability and trust-building
- **Display**: 
  ```
  Prior Clots (3 events)    ████████████████  45%
  Phosphate ↑ (7.2 mg/dL)   ████████          18%
  Creatinine ↑ (3.1 mg/dL)  ██████            12%
  Platelets ↓ (89 K/µL)     ████              8%
  Filter Pressure ↑ (185)   ███               6%
  ```
- **Benefit**: Providers can validate against clinical judgment

### LLM Integration
- Natural language synthesis of predictions
- Explanation of feature contributions in clinical terms

---

## 📈 Key Results

### Model Performance
- **ROC-AUC**: 0.9904 (near-perfect discrimination)
- **Precision**: 0.899 (90% of alerts are true positives)
- **Recall**: 0.917 (catches 92% of clots)
- **F1-Score**: 0.908

### Confusion Matrix (Test Set: 25,123 patients)
|                | Predicted: No Clot | Predicted: Clot |
|----------------|-------------------|-----------------|
| **Actual: No Clot** | 22,477 (TN) | 248 (FP) |
| **Actual: Clot**    | 199 (FN) | 2,199 (TP) |

- **False Positive Rate**: 1.1% (minimal alert fatigue)
- **False Negative Rate**: 8.3% (misses ~199 clots, but catches majority)

### PCA vs XGBoost Comparison
Testing if dimensionality reduction maintains performance:

| Features | ROC-AUC | Performance Loss |
|----------|---------|------------------|
| 57 (Original) | 0.9904 | — (baseline) |
| 37 (PCA 90%) | 0.9273 | -6.31% ❌ |
| 30 (PCA 80%) | 0.9190 | -7.14% ❌ |

**Conclusion**: All 57 features necessary for optimal performance. CRRT clotting is genuinely high-dimensional.

---

## 🚧 Open Questions & Future Work

### 1. First-Time Patient Performance
**Issue**: `prior_clots_count` contributes 45% to risk scores. Does performance degrade for first-time CRRT patients without clotting history?

**Next Step**: Subgroup analysis comparing model performance on patients with vs. without prior clots.

### 2. Deployment Feasibility
**Challenge**: Model requires 57 features, but manual entry of 57 values is clinically unrealistic.

**Options**:
- **MDCalc-style calculator**: Requires feature selection to 10-15 most important features
- **Lightweight Epic integration**: Auto-pull labs/CRRT parameters (requires IT resources)
- **Full integration**: Requires Sepsis Watch-level institutional backing (unlikely for smaller patient population)

**Next Step**: Systematic feature selection testing to identify minimum viable feature set (target: 15 features with <5% AUC loss).

### 3. Alert Threshold Calibration
**Question**: At what risk score (75%? 85%? 95th percentile?) should the system escalate from passive monitoring to active provider notification?

**Consideration**: Balance sensitivity (catch clots) vs. alert fatigue (don't overwhelm providers).

**Next Step**: Stakeholder interviews with nephrologists and ICU nurses.

### 4. Phenotype-Specific Recommendations
**Question**: Should the LLM provide different clinical recommendations for patients matching the "kidney injury phenotype" vs. "stable phenotype"?

**Example**: "This patient exhibits kidney injury pattern—focus on volume status and nephrotoxin avoidance."

### 5. Missing Lab Handling
**Question**: How should the interface handle incomplete lab data in time-sensitive scenarios?

**Options**:
- Display "Model confidence: LOW" warning
- Attempt prediction with degraded accuracy
- Refuse prediction until critical labs available

---

## 🛠️ Technology Stack

### Core Libraries
- **scikit-learn** (1.3.0): Machine learning algorithms
- **XGBoost** (2.0.0): Gradient boosting (best model)
- **pandas** (2.0.3): Data manipulation
- **numpy** (1.24.3): Numerical computing
- **matplotlib** (3.7.2): Visualization
- **seaborn** (0.12.2): Statistical visualization
- **imbalanced-learn** (0.11.0): SMOTE resampling

### Development Environment
- **Python**: 3.8+
- **Jupyter Notebook**: Interactive analysis
- **MIMIC-IV**: Clinical database

---

## 📁 Repository Structure

```
crrt-clotting-prediction/
├── README.md                               
├── requirements.txt                        
├── part2_crrt_clots_prediction_v3.ipynb    # Main analysis notebook
├── outputs/
│   ├── confusion_matrices_best_models.png
│   ├── roc_curves_best_models.png
│   ├── pca_vs_kmeans_comparison.png
│   ├── kmeans_feature_profiles.png
│   ├── xgboost_original_vs_pca.png
│   └── feature_importance_top20.png
├── projectproposal/
│   ├── LowFidelity_Sketch.pdf
│   ├── projectproposal_sketches.pdf
│   └──  projectproposal.docx
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
matplotlib>=3.7.2
seaborn>=0.12.2
imbalanced-learn>=0.11.0
jupyter>=1.0.0
scipy>=1.10.1
```

### Data Access
This project uses the MIMIC-IV database, which requires:
1. Completion of CITI training
2. Signed data use agreement
3. PhysioNet credentialed access

### Running the Analysis
```bash
# Launch Jupyter
jupyter notebook

# Open the main notebook
notebooks/part2_crrt_clots_prediction_v3.ipynb

# Run all cells
```

---

## 📊 Reproducing Results

### Train the XGBoost Model
```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load preprocessed data (see notebook)
# X_train_scaled, X_test_scaled, y_train, y_test

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)

# Evaluate
y_prob = xgb.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")  # Expected: 0.9904
```

### Run K-means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Reduce dimensionality for clustering
pca = PCA(n_components=25, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

# Fit K-means (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Analyze clot rates by cluster
print(pd.crosstab(clusters, y_train, normalize='index'))
```

---

## 🎓 Academic Context

This project was completed as part of DTI530: Technology Core, part of Duke University's Master of Engineering in Design, Technology, & Innovation program.

### Learning Objectives Addressed
1. ✅ Applied supervised learning (classification) to real clinical data
2. ✅ Performed unsupervised learning (PCA, K-means) for pattern discovery
3. ✅ Evaluated model performance using appropriate metrics for imbalanced data
4. ✅ Designed user-centered interface based on clinical workflow needs
5. ✅ Addressed deployment challenges (explainability, feasibility, trust)

### Related Work
This project builds on lessons learned from our hospital's Sepsis Watch implementation, which demonstrates that importance of explainable AI for full clinical trust. While the sepsis model is successfully used every day, frequent provider questions about alert justification reveal an explainability gap that undermines confidence.

---

## 🤝 Acknowledgments

- **MIMIC-IV Database**

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The MIMIC-IV data is subject to separate data use agreements and cannot be redistributed. See PhysioNet for access requirements.

---

## 📚 References

1. Johnson, A.E.W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

3. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

4. Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.

5. Sendak, M.P., Gao, M., Brajer, N., & Balu, S. (2020). Presenting machine learning model information to clinical end users with model facts labels. *NPJ Digital Medicine*, 3(1), 41.

---

## 🔄 Version History

- **v1.0.0** (November 2025) - Initial release
  - XGBoost model training and evaluation
  - LR and Random Forest analysis
  - PCA and K-means analysis
  - Interface design sketches

---

## 🌟 Future Enhancements

- [ ] Prospective validation study with real-time clinical data
- [ ] Feature selection analysis for MDCalc-style deployment
- [ ] LLM integration for natural language explanations
- [ ] A/B testing: explainable interface vs. traditional alerts

---
