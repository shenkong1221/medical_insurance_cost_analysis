# Medical Insurance Cost Analysis & Prediction

## Executive Summary
Built an end-to-end analytical and predictive framework to understand and estimate U.S. medical insurance costs using statistical inference, regression modeling, and machine learning.

Key outcome:
- Insurance costs are primarily driven by smoking behavior
- Nonlinear models significantly outperform linear regression
- Random Forest improves predictive accuracy by ~16% R² uplift

## Business Problem
Medical insurance pricing is influenced by multiple demographic and behavioral factors.

The goal of this project is to:
- Identify the most impactful cost drivers
- Quantify their statistical significance
- Build predictive models for insurance cost estimation
- Compare interpretability vs predictive accuracy

## Dataset
- Source: Kaggle – Medical Cost Personal Dataset
- Size: 1337 observations
- Target variable: charges (medical insurance cost)

## Approach
### Data Processing
- Removed duplicates
- Verified missing values
- One-hot encoding for categorical variables

### Statistical Inference
Performed hypothesis testing to validate relationships:
- Pearson correlation → continuous variables
- t-test → binary variables
- ANOVA → categorical variables

Key finding:
- Smoking status shows the strongest statistical impact on insurance charges (p < 0.001)

### Predictive Modeling
Baseline Model: OLS Regression
- R² = 0.75
- Strong interpretability
- Limited ability to capture nonlinear effects

Improved Model: Log-Linear Regression
- Addressed heteroscedasticity
- R² = 0.76
- Improved stability but limited structural gain

Machine Learning Model: Random Forest
- Captures nonlinear relationships and feature interactions
- R² = 0.87
- Best predictive performance among all models

## Model Comparison
| Model | R-squared | MAE | RMSE | Strength |
|------|-----------|-----|------|------------------|
| OLS Regression | 0.750 | 4181.335 | 6058.642 | High interpretability |
| Log-Linear Regression | 0.762 | 4227.538 | 8281.830 | Stable statistical model |
| Random Forest | 0.873 | 2703.819 | 4839.950 | Best predictive accuracy |

## Key Insights
- Smoking is the dominant cost driver
- Age and BMI have moderate but consistent effects
- Medical insurance costs exhibit nonlinear behavior
- Linear models underperform for high-cost individuals
- Tree-based models better capture risk stratification

## Interpretation Trade-off
| Model Type | Use Case |
|------|------------------|
| Linear Models | Explainability & statistical inference |
| Random Forest | Prediction & risk modeling |

## Tech Stack
- Python
- Pandas / NumPy
- Statsmodels
- Scikit-learn
- Matplotlib

## Key Takeaway
Insurance cost prediction is not purely linear.
While linear regression provides interpretable insights, machine learning models reveal hidden nonlinear risk structures, significantly improving predictive accuracy.

## Future Work
- Add SHAP explainability for model interpretability
- Explore Gradient Boosting / XGBoost
- Introduce interaction terms in regression models
- Deploy as a REST API for real-world usage
