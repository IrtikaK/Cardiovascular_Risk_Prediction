# Cardiovascular_Risk_Prediction
# Prediction of Cardiovascular Disease using Classical Machine Learning Algorithms

**MSc Dissertation Project** | Northumbria University | MSc Artificial Intelligence Technology

Predicting cardiovascular disease (CVD) risk using classical machine learning algorithms on the Cleveland Heart Disease Dataset, with comprehensive evaluation including cross-validation, calibration assessment, subgroup analysis, and SHAP-based interpretability.

---

## 📊 Project Overview

| Aspect | Details |
|--------|---------|
| **Dataset** | Cleveland Heart Disease (UCI ML Repository) |
| **Samples** | 303 patients |
| **Features** | 13 clinical variables |
| **Task** | Binary Classification (CVD present/absent) |
| **Best Model** | Random Forest |
| **Best AUC-ROC** | 0.964 (95% CI: 0.911-0.995) |

---

## 🎯 Research Objectives

- Implement and compare five classical ML algorithms for CVD prediction
- Establish robust methodology: preprocessing, stratified cross-validation, hyperparameter optimisation
- Evaluate using multiple metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC, Calibration
- Examine model robustness across demographic subgroups (sex, age)
- Enhance interpretability using SHAP analysis

---

## 🔬 Methodology

### Data Preprocessing
- Missing value imputation (mode imputation for categorical variables)
- Target variable binarisation (0 = no disease, 1-4 = disease present)
- Feature standardisation (StandardScaler for numerical features)
- One-hot encoding for categorical features

### Model Training
- **Train-Test Split**: 80:20 stratified sampling
- **Cross-Validation**: 5-fold stratified CV
- **Hyperparameter Tuning**: RandomizedSearchCV (50 iterations)

### Evaluation Framework
- Discrimination metrics (AUC-ROC, Accuracy, Precision, Recall, F1)
- Calibration assessment (Brier Score, Calibration curves)
- Bootstrap confidence intervals (300 iterations)
- Subgroup performance analysis (sex, age groups)
- SHAP-based feature importance

---

## 🤖 Models Evaluated

| Algorithm | Description |
|-----------|-------------|
| **Logistic Regression** | Linear baseline model with regularisation |
| **Random Forest** | Ensemble of decision trees (100 estimators) |
| **Support Vector Machine** | RBF kernel with optimised C and gamma |
| **K-Nearest Neighbours** | Instance-based learning |
| **Decision Tree** | Single tree classifier |

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.918** | **0.871** | **0.964** | **0.915** | **0.964** |
| Logistic Regression | 0.869 | 0.813 | 0.929 | 0.867 | 0.958 |
| SVM | 0.852 | 0.800 | 0.893 | 0.844 | 0.946 |
| KNN | 0.820 | 0.774 | 0.857 | 0.814 | 0.872 |
| Decision Tree | 0.787 | 0.742 | 0.821 | 0.780 | 0.708 |

### Key Findings
```
✓ RANDOM FOREST demonstrates SUPERIOR PERFORMANCE
  - AUC-ROC: 0.964 (95% CI: 0.911-0.995)
  - Accuracy: 91.8%
  - Sensitivity: 96.4% (critical for detecting disease)
  - Specificity: 87.9%
  - Brier Score: 0.093 (good calibration)
```

### Confusion Matrix (Random Forest)

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 29 (TN) | 4 (FP) |
| **Actual Positive** | 1 (FN) | 27 (TP) |

### Subgroup Performance

| Subgroup | N | AUC-ROC |
|----------|---|---------|
| Female | 20 | 1.000 |
| Male | 41 | 0.926 |
| Age 40-49 | 13 | 1.000 |
| Age 50-59 | 28 | 0.969 |
| Age 60-69 | 14 | 0.756 |

---

## 🔍 Feature Importance (SHAP Analysis)

Top predictive features identified:

1. **Number of major vessels** (fluoroscopy) - Mean |SHAP|: 0.42
2. **Chest pain type** - Mean |SHAP|: 0.38
3. **Thalassemia status** - Mean |SHAP|: 0.31
4. **Maximum heart rate achieved**
5. **ST depression (oldpeak)**
6. **Exercise-induced angina**

These features align with established cardiovascular pathophysiology, supporting clinical validity.

---

## 🛠️ Technologies Used

- **Python 3.9**
- **scikit-learn 1.3.0** - ML algorithms, preprocessing, evaluation
- **SHAP** - Model interpretability
- **pandas / numpy** - Data manipulation
- **matplotlib / seaborn** - Visualisations
- **statsmodels** - Statistical analysis

---

## 📁 Project Structure
```
├── Cardiovascular_Risk_Prediction.ipynb    # Main analysis notebook
├── README.md                                # Project documentation
└── data/
    └── heart.csv                            # Cleveland Heart Disease Dataset
```

---

## 🚀 How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/IrtikaK/Cardiovascular_Risk_Prediction.git
```

2. **Install dependencies**
```bash
   pip install pandas numpy scikit-learn shap matplotlib seaborn
```

3. **Run the notebook**
   - Open in Jupyter Notebook / JupyterLab, or
   - Upload to [Google Colab](https://colab.research.google.com/)

---

## 📊 Visualisations Included

- ROC Curves for all models
- Precision-Recall Curves
- Confusion Matrix
- Calibration Curves
- SHAP Summary Plots
- SHAP Interaction Plots
- Feature Importance Bar Charts
- Subgroup Performance Analysis

---

## 📚 Clinical Significance

- **High Sensitivity (96.4%)**: Minimises missed diagnoses of CVD
- **Good Calibration**: Predicted probabilities align with actual outcomes
- **Interpretable**: SHAP analysis provides clinically meaningful explanations
- **Robust**: Consistent performance across demographic subgroups

---

## ⚠️ Limitations

- Historical single-centre dataset (Cleveland Clinic)
- Relatively small sample size (n=303)
- Relies on some invasive diagnostic features
- Requires external validation on modern datasets

---

## 📌 Future Improvements

- [ ] External validation on contemporary datasets 
- [ ] Develop models using non-invasive variables only
- [ ] Explore deep learning architectures
- [ ] Prospective clinical validation

---

## 📄 Dataset Source

[Heart Disease Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

## 👤 Author

**Irtika Khan**  
MSc Artificial Intelligence Technology  
Northumbria University  

GitHub: [@IrtikaK](https://github.com/IrtikaK)

---

## 📜 License

This project was completed as part of MSc dissertation requirements at Northumbria University.  
For educational and research purposes.
