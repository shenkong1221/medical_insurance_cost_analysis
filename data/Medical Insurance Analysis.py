from tkinter.font import BOLD

import pandas as pd
import numpy as np
import statsmodels.api as sm


# Step 1: Data Cleaning
insurance_df = pd.read_csv("/Users/shenkong/Desktop/个人简历/项目/01 Medical_insurance_cost_analysis/data/insurance.csv")
print(insurance_df.head())

insurance_df.drop_duplicates(inplace=True)
insurance_df.isnull().sum()
insurance_df["age"].value_counts().sort_index()
insurance_df["age"].describe()
insurance_df["smoker"].value_counts()

ins_cleaned_df = insurance_df.copy()
ins_cleaned_df.to_csv("/Users/shenkong/Desktop/个人简历/项目/01 Medical_insurance_cost_analysis/data/ins_cleaned.csv", index=False, encoding='utf-8')  # Output the cleaned data

print(ins_cleaned_df.head())

# Step 2: Hypothesis Testing
# 1. Continuous Variables: age/bmi/children vs charges
# Use correlation & p-value
from scipy.stats import pearsonr

corr_age, p_age = pearsonr(ins_cleaned_df["age"], ins_cleaned_df["charges"])
corr_bmi, p_bmi = pearsonr(ins_cleaned_df["bmi"], ins_cleaned_df["charges"])
corr_children, p_children = pearsonr(ins_cleaned_df["children"], ins_cleaned_df["charges"])

cp1_df = pd.DataFrame(
    {
        "Correlation": [corr_age, corr_bmi, corr_children],
        "p-value": [p_age, p_bmi, p_children]
    },
    index = ["age", "bmi", "children"]
)
print(cp1_df)

# 2. Categorical Variables: smoker/sex/region vs charges
# 2-1. Use t-test(Yes or No)
from scipy.stats import ttest_ind

yes = ins_cleaned_df[ins_cleaned_df["smoker"] == "yes"]["charges"]
no = ins_cleaned_df[ins_cleaned_df["smoker"] == "no"]["charges"]

t_stat, p_smoker = ttest_ind(yes, no)

# There is a strongly significant difference in medical charges between smokers and non-smokers

male = ins_cleaned_df[ins_cleaned_df["sex"] == "male"]["charges"]
female = ins_cleaned_df[ins_cleaned_df["sex"] == "female"]["charges"]

t_stat, p_sex = ttest_ind(male, female)
# There is a slightly significant difference between male and female
# 看个mini注释答案！！！！！！！！

# 2-2. Use ANOVA -- tests whether there are statistically significant differences among all groups
print(ins_cleaned_df["region"].unique())
# Use ANOVA instead of t-test, there are more than 2 values in this column

from scipy.stats import f_oneway

four_regions = [
    ins_cleaned_df[ins_cleaned_df["region"] == r]["charges"]
    for r in ins_cleaned_df["region"].unique()
]

f_stat, p_region = f_oneway(*four_regions)

# Summary table:
summary_table = pd.DataFrame({
    "Test":[
        "Pearson",
        "Pearson",
        "Pearson",
        "t-test",
        "t-test",
        "ANOVA"],
    "p-value":[
        p_age,
        p_bmi,
        p_children,
        p_smoker,
        p_sex,
        p_region]
},
    index = [
    "age",
    "bmi",
    "children",
    "smoker",
    "sex",
    "region"]
)

# Create a significant label function:
def sig_label(p):
    if p < 0.001:
        return "Yes (Strong)"
    elif p < 0.05:
        return "Yes"
    else:
        return "No"

# Add a new column:
summary_table["Significant?"] = summary_table["p-value"].apply(sig_label)
summary_table["p-value"] = summary_table["p-value"].apply(lambda x: round(x, 4)) # Round p-values to 4 decimal places

print(summary_table)

# Step 3: Baseline Linear Model
# One-hot encoding:
df_encoded = pd.get_dummies(ins_cleaned_df, drop_first=True)
# Include all the column names
X_full = df_encoded[["age", "sex_male", "bmi", "children", "smoker_yes",
                    "region_southwest", "region_northwest", "region_southeast"
                    ]].astype(float)
X_full = sm.add_constant(X_full)
y = ins_cleaned_df["charges"]
# Build full-variable model
model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())

# Build another model without sex variable
X_outsex = df_encoded[["age", "bmi", "children", "smoker_yes",
                    "region_southwest", "region_northwest", "region_southeast"
                    ]].astype(float)
X_outsex = sm.add_constant(X_outsex)

model_outsex = sm.OLS(y, X_outsex).fit()
print(model_outsex.summary())

# Print r-square and adj r-square of two models
print("*" * 15 + " R-square & Adjusted R-square " + "*" * 15)
print(f"Model with full-variable: R² = {model_full.rsquared:.4f}, Adj R² = {model_full.rsquared_adj:.4f}")
print(f"Model without sex: R² = {model_outsex.rsquared:.4f}, Adj R² = {model_outsex.rsquared_adj:.4f}")
print("*" * 60)

# Feature Importance (model_outsex):
import matplotlib.pyplot as plt

coef = model_outsex.params.drop("const")
plt.figure(figsize=(8, 5))
coef.sort_values(ascending=False).plot(kind="bar", color="firebrick")

plt.title("Feature Impact on Insurance Charges")
plt.xlabel("Coefficient Value")
plt.ylabel("Variables")

plt.show()
plt.close()

# Step 4: Improve the accuracy of prediction by models comparison
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Train/Test Split
X = df_encoded[["age", "bmi", "children", "smoker_yes",
                    "region_southwest", "region_northwest", "region_southeast"
                    ]].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build a function calculating r-square, mae and rmse
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:30s} R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    return r2, mae, rmse

# Model 1: 5-Variable baseline LM
X_train_c = sm.add_constant(X_train)
X_test_c = sm.add_constant(X_test)
model_base = sm.OLS(y_train, X_train_c).fit()
pred_base = model_base.predict(X_test_c)
evaluate("1. Baseline (5-variable)", y_test, pred_base)

# Model 2: Linear model with interaction terms
X_train_i = X_train.copy()
X_test_i = X_test.copy()
X_train_i["bmi_smoker"] = X_train_i["bmi"] * X_train_i["smoker_yes"]
X_test_i["bmi_smoker"] = X_test_i["bmi"] * X_test_i["smoker_yes"]

X_train_ic = sm.add_constant(X_train_i)
X_test_ic = sm.add_constant(X_test_i)
model_inter = sm.OLS(y_train, X_train_ic).fit()
pred_inter = model_inter.predict(X_test_ic)
evaluate("2. + smoker×bmi interaction", y_test, pred_inter)

# Model 3: Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
evaluate("3. Random Forest", y_test, pred_rf)

# Build Actual vs Predicted Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
preds = [pred_base, pred_inter, pred_rf]
titles = ["Baseline LM", "LM with Interaction Terms", "Random Forest Model"]

for ax, pred, title in zip(axes, preds, titles):
    ax.scatter(y_test, pred, alpha=0.4, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    ax.set_title(title)
    ax.legend()

plt.show()
plt.close()

# Comparison Dataframe and Bar charts
comparison_df = pd.DataFrame({
    "Model": ["Baseline LM", "LM with Interaction Terms", "Random Forest Model"],
    "R-Square": [0.8069, 0.8869, 0.8815],
    "MAE": [4177.65, 2794.21, 2570.25],
    "RMSE": [5957.53, 4559.75, 4666.49]
})

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, metric in zip(axes, ["R-Square", "MAE", "RMSE"]):
    ax.bar(comparison_df["Model"], comparison_df[metric], color=["gray", "steelblue", "firebrick"])
    ax.set_title(metric)
    ax.tick_params(axis='x', rotation=15)

plt.show()
plt.close()

# Step 5: Data Visualization
# bmi vs charges colored by smoking status
plt.figure(figsize=(8, 5))

colors = {0: "darkgreen", 1: "firebrick"}
labels = {0: "Non-smoker", 1: "Smoker"}
for smoker_status, subset in df_encoded.groupby("smoker_yes"):
    plt.scatter(subset["bmi"], subset["charges"],
                alpha=0.5, color=colors[smoker_status], label=labels[smoker_status])

plt.title("Charges vs. BMI, grouped by Smoking Status")
plt.xlabel("BMI")
plt.ylabel("Charges ($)")

plt.show()
plt.close()

# Step 6: Model Limitation
# Build a residual plot of LM with Interaction
residuals_inter = y_test - pred_inter

plt.figure(figsize=(8,5))
plt.scatter(pred_inter, residuals_inter, alpha=0.4)

plt.axhline(y=0, color="r", linestyle="--")

plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot -- LM with Interaction")

plt.show()
