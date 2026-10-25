# Heart Disease Prediction: Exploratory Data Analysis and Preprocessing

## Project Overview

This project focuses on exploring and preparing a heart disease dataset for machine learning. It includes data quality checks, handling invalid values, categorical encoding, numerical feature scaling, and statistical analysis of relationships between features and the target variable.

The goal of this stage is to create a clean, well-understood dataset that can be used in a later machine learning workflow.

## Dataset

- **Dataset size:** 918 rows and 12 original columns
- **Target column:** `HeartDisease`
- **Target values:** `0` and `1`

The original dataset contains demographic information, clinical measurements, symptoms, and the heart disease target.

### Original columns

| Column | Description |
|---|---|
| `Age` | Age of the patient |
| `Sex` | Sex category recorded in the dataset |
| `ChestPainType` | Chest pain category |
| `RestingBP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol measurement |
| `FastingBS` | Fasting blood sugar indicator |
| `RestingECG` | Resting electrocardiogram result |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina indicator |
| `Oldpeak` | Oldpeak measurement recorded in the dataset |
| `ST_Slope` | Slope of the peak exercise ST segment |
| `HeartDisease` | Target label: 0 or 1 |

> **Note:** Column meanings and units should be confirmed against the dataset's original source before clinical interpretation.

## Workflow

### 1. Create a working copy

A copy of the original DataFrame was used so the source data remained unchanged during preprocessing.

```python
heart_df = heart.copy()
```

### 2. Handle invalid zero values

Zero values in `Cholesterol` and `RestingBP` were treated as missing values for this analysis. These values were replaced with `NaN`.

```python
import numpy as np

heart_df[['Cholesterol', 'RestingBP']] = (
    heart_df[['Cholesterol', 'RestingBP']].replace(0, np.nan)
)
```

This decision should be documented because zero values may represent invalid or unrecorded measurements in this dataset, rather than true measurements.

### 3. Impute missing values

Missing values in `Cholesterol` and `RestingBP` were filled using each column's median.

```python
for col in ['Cholesterol', 'RestingBP']:
    heart_df[col] = heart_df[col].fillna(heart_df[col].median())
```

The median is less sensitive to extreme values than the mean. Missing-value counts should be checked after imputation.

### 4. Inspect outliers

The Interquartile Range (IQR) method was used to identify potential outliers in numerical columns.

The IQR is calculated as:

- `IQR = Q3 - Q1`
- Lower bound: `Q1 - 1.5 * IQR`
- Upper bound: `Q3 + 1.5 * IQR`

Potential outliers were inspected but **not automatically removed**. An IQR outlier is not necessarily an error, especially in health-related data where unusual values may be meaningful. Values should be investigated using domain knowledge and the dataset source before removal.

### 5. Encode categorical features

Categorical variables were converted into numeric indicator columns using one-hot encoding. With `drop_first=True`, one category is omitted as a reference category for each encoded feature.

```python
cat_cols = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

heart_df = pd.get_dummies(
    heart_df,
    columns=cat_cols,
    drop_first=True,
    dtype=int
)
```

This allows common machine learning algorithms to work with the categorical data without treating category labels as continuous numbers.

### 6. Scale numerical features

Standardization was applied to selected continuous numerical features using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scale_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

scaler = StandardScaler()
heart_df[scale_cols] = scaler.fit_transform(heart_df[scale_cols])
```

Standardization transforms values to a scale based on the feature's mean and standard deviation. Binary variables such as `FastingBS` and the target `HeartDisease` are not included in this scaling step.

**Important for model training:** In a final predictive workflow, split the data into training and test sets first. Fit the imputer and scaler on the training set only, then apply the fitted transformations to the test set. This helps prevent data leakage. A scikit-learn `Pipeline` or `ColumnTransformer` is recommended for a robust implementation.

### 7. Pearson correlation analysis

Pearson correlation was used to examine the direction and strength of linear relationships between selected numeric or binary features and `HeartDisease`.

```python
from scipy.stats import pearsonr

selected_features = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS',
    'MaxHR', 'Oldpeak'
]

correlation = {
    feature: pearsonr(heart_df[feature], heart_df['HeartDisease'])[0]
    for feature in selected_features
}
```

The correlation coefficient ranges from `-1` to `+1`:

- A positive value indicates a positive linear association.
- A negative value indicates a negative linear association.
- A value near zero indicates little or no linear association.

Categorical variables can also be included after encoding, provided the column names match the actual processed DataFrame. Correlation does not establish causation.

### 8. Chi-square test for categorical features

The Chi-square test of independence was used to assess whether categorical features and `HeartDisease` show a statistically significant association.

```python
import pandas as pd
from scipy.stats import chi2_contingency

alpha = 0.05
chi2_result = {}

for col in cat_features:
    contingency = pd.crosstab(heart_df[col], heart_df['HeartDisease'])
    chi_stat, p_value, _, _ = chi2_contingency(contingency)

    decision = (
        'Fail to Reject Null'
        if p_value > alpha
        else 'Reject Null'
    )

    chi2_result[col] = {
        'chi2_statistics': chi_stat,
        'p_value': p_value,
        'decision': decision
    }

chi2_df = pd.DataFrame(chi2_result).T
chi2_df.sort_values(by='p_value', ascending=True, inplace=True)

print(chi2_df)
```

Interpretation at a significance level of `0.05`:

- **p-value < 0.05:** Reject the null hypothesis; there is evidence of an association between the feature and target.
- **p-value >= 0.05:** Fail to reject the null hypothesis; the test does not provide sufficient evidence of an association.

A non-significant result does not automatically mean a feature must be dropped. Feature selection should also consider domain knowledge, model performance, sample size, and the possibility of relationships that the test does not capture.

For the Chi-square test, use categorical variables (or categorical indicator columns) and the unscaled target. Ensure `cat_features` contains valid column names in `heart_df`.

## Data Validation

After preprocessing, useful checks include:

```python
print(heart_df.shape)
print(heart_df.head())
print(heart_df.isna().sum())
print(heart_df.dtypes)
```

Confirm that:
- The target column `HeartDisease` is still present.
- No unexpected missing values remain.
- The expected encoded columns exist.
- The feature matrix does not include the target column.
- The original `heart` DataFrame remains unchanged.

## Key Decisions

- Kept the original dataset unchanged by processing a copy.
- Replaced zero values in `Cholesterol` and `RestingBP` with missing values before median imputation.
- Inspected potential outliers without automatically deleting them.
- Applied one-hot encoding to categorical variables.
- Standardized selected continuous numerical features.
- Used Pearson correlation and Chi-square testing as exploratory feature-analysis methods.

## Limitations

- Statistical association does not prove causation.
- Outlier and invalid-value decisions require verification against the dataset source and domain context.
- Feature selection based only on individual statistical tests may miss interactions and nonlinear relationships.
- Preprocessing must be fitted on training data only when building and evaluating a predictive model.
- This project is for data analysis and machine learning practice, not for clinical diagnosis or medical decision-making.

## Tools and Libraries

- Python
- Pandas
- NumPy
- SciPy
- scikit-learn

## Future Improvements

- Add visualizations for distributions, class balance, and feature relationships.
- Build a reproducible preprocessing pipeline.
- Split data into training and test sets before fitting preprocessing steps.
- Train and compare suitable classification models.
- Evaluate models using appropriate metrics such as precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Investigate model interpretability and the limitations of using the dataset for health-related predictions.

## Disclaimer

This repository is an educational machine learning project. It is not a medical device and must not be used to diagnose heart disease or guide treatment.
