# Medical Insurance Cost Analysis & Prediction

## Executive Summary
Built an end-to-end analytical framework to estimate U.S. medical insurance costs using statistical inference, regression modelling, and machine learning.

**Key Outcome:**
* **Smoker status** is the dominant cost driver.
* Introducing the interaction term `smoker × bmi` boosted the Baseline Linear Model $R^2$ from **0.805 to 0.883**, outperforming Random Forest (**0.872**) while retaining full model interpretability.

---

## Business Problem
Quantify demographic and behavioural cost drivers, and evaluate the trade-off between **model interpretability** and **predictive accuracy** for insurance pricing.

---

## Dataset
* **Source:** Public Kaggle – Medical Insurance Cost Dataset
* **Size:** 1,337 observations
* **Target:** `charges` (medical insurance cost)

---

## Enterprise Data Context & Systems
In a production insurance environment, data is ingested across core systems:
* **Application Form / PAS:** Primary intake for demographics and medical health disclosures (`smoker`, `bmi`, etc.).
* **Underwriting System:** Integrates actuarial tables to calculate base premium charges.
* **Enterprise Data Warehouse (EDW):** Centralises cross-system data (Claims, Billing, CRM) for risk modelling.

---

## Approach & Diagnostics

### 1. Data Preprocessing & Statistical Inference
* Verified zero missing values; applied one-hot encoding.
* Validated relationships via Pearson correlation, t-tests, and ANOVA.
* **Finding:** Smoking status showed the strongest impact ($p < 0.001$).

### 2. Predictive Modeling & Diagnostic Evolution
* **Baseline Linear Model :** High interpretability, but failed to capture joint non-linear risk.
* **Random Forest :** Captured complex non-linear feature interactions, but lacks transparent logic.
* **Refined LM with Interaction :** Manually adding `smoker * bmi` captured the primary risk structure, beating Random Forest while keeping full interpretability.

---

## Model Comparison

| Model | $R^2$ | MAE | RMSE | Key Performance Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Linear Model** | 0.8069 | 4,177.65 | 5,957.53 | Interpretable baseline; misses joint risk factors |
| **Random Forest** | 0.8815 | **2,570.25** | 4,666.49 | Lower MAE on typical cases; black-box structure |
| **Refined LM (`smoker × bmi`)** | **0.8869** | 2,794.21 | **4,559.75** | **Best overall $R^2$ & RMSE; directly converts to pricing rules** |

*Note: Metrics evaluated on an out-of-sample test set.*

---

## Key Insights
1. **Compound Risk:** Smoking alone increases costs, but **Smoking + High BMI** creates a sharp non-linear cost surge.
2. **Interpretability Wins:** Domain-informed feature engineering allowed a linear model with interaction terms to outperform black-box machine learning algorithms.
3. **Actionable Business Rule:** Translates directly into automated underwriting rules (e.g., immediate premium surcharge multiplier for high-BMI smokers).

---

## Model Limitations
Residual diagnostics of the Refined LM revealed persistent heteroscedasticity among higher-charge cohorts.

---

## Tech Stack
* **Python:** Pandas, NumPy
* **Modelling & Diagnostics:** Statsmodels, Scikit-learn
* **Visualisation:** Matplotlib, Seaborn

---

## Future Work
* Apply **Log transformation (log(Charges))** to handle skewed insurance claim distributions and mitigate remaining heteroscedasticity.
* Deploy the model via **FastAPI** with automated underwriting surcharge logic.

