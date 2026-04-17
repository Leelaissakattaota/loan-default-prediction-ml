# 🏦 Loan Default Prediction — Coursera Data Science Coding Challenge

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression%20%2B%20PCA-FF6B35?style=flat-square)
![Library](https://img.shields.io/badge/Library-Scikit--learn%20%7C%20Pandas%20%7C%20Seaborn-2E7D32?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-ROC%20AUC-0052CC?style=flat-square)
![Source](https://img.shields.io/badge/Source-Coursera%20Coding%20Challenge-6A0DAD?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📌 Project Overview

Coursera **Data Science Coding Challenge** — predicts the probability 
of loan default for **109,435 borrowers** using a full ML pipeline: 
EDA, feature engineering, dimensionality reduction with PCA, Logistic 
Regression, and competition-format submission evaluated on **ROC AUC**.

**Domain:** Financial ML — Credit Risk  
**Task:** Binary classification (loan default prediction)  
**Metric:** ROC AUC Score  
**Dataset:** 255,347 training loans + 109,435 test loans  

---

## 📂 Project Structure

```
Loan-Default-Prediction/
│
├── LoanDefaultPrediction.ipynb   # Full ML pipeline
├── train.csv                     # 255,347 loan records (labelled)
├── test.csv                      # 109,435 loan records (unlabelled)
├── data_descriptions.csv         # Feature descriptions
├── prediction_submission.csv     # Final predictions (LoanID + probability)
└── README.md
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `LoanID` | Identifier | Unique loan ID |
| `Age` | Numeric | Borrower age |
| `Income` | Numeric | Annual income |
| `LoanAmount` | Numeric | Loan amount |
| `Education` | Categorical | Education level |
| `EmploymentType` | Categorical | Employment type |
| `MaritalStatus` | Categorical | Marital status |
| `HasMortgage` | Categorical | Existing mortgage |
| `HasDependents` | Categorical | Has dependents |
| `LoanPurpose` | Categorical | Purpose of loan |
| `HasCoSigner` | Categorical | Has co-signer |
| `Default` | **Target** | 1 = Defaulted, 0 = No default |

---

## 🚀 ML Pipeline

### Step 1 — Exploratory Data Analysis
```python
print(train_df.info())
print(train_df.describe())
sns.countplot(x='Default', data=train_df)  # Class distribution
sns.heatmap(corr_matrix, cmap='coolwarm')  # Correlation heatmap
train_df.hist(bins=20, figsize=(20, 15))   # Feature distributions
```

### Step 2 — Data Cleaning & Preprocessing
```python
train_df.drop_duplicates()
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

# One-hot encode 7 categorical features
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus',
                    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
```

### Step 3 — Dimensionality Reduction with PCA
```python
# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal components (95% variance retained)
pca = PCA()
cumulative_variance = np.cumsum(pca.fit(X_scaled).explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.95) + 1

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
```

### Step 4 — Model Training & Evaluation
```python
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f'ROC AUC Score: {roc_auc}')
```

### Step 5 — Predict & Submit (109,435 rows)
```python
# Apply same scaler + PCA to test set
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Generate probability predictions
y_test_pred_proba = model.predict_proba(X_test_pca)[:, 1]

prediction_df = pd.DataFrame({
    'LoanID': test_df['LoanID'],
    'predicted_probability': y_test_pred_proba
})
prediction_df.to_csv("prediction_submission.csv", index=False)
```

---

## 🎓 Skills Demonstrated

- End-to-end ML pipeline — EDA → preprocessing → modeling → submission
- PCA dimensionality reduction — 95% variance threshold
- StandardScaler + fit/transform separation (train vs test)
- One-hot encoding of 7 categorical features
- ROC AUC evaluation for imbalanced binary classification
- Logistic Regression for credit risk scoring
- Competition-format submission (109,435 predictions)
- Correlation heatmap + feature distribution analysis
- Missing value imputation

---

## 📜 Certifications

| Certification | Issuer | Platform |
|---|---|---|
| IBM Data Science Professional Certificate | IBM | Coursera |
| IBM Generative AI Professional Certificate | IBM | Coursera |
| IBM Agentic AI with RAG Certificate | IBM | Coursera |
| IBM RAG and Agentic AI Professional Certificate | IBM | Coursera |

---

## 🤝 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Leela%20A-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leela-a)
[![Gmail](https://img.shields.io/badge/Gmail-attotaleelaissak@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:attotaleelaissak@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Leelaissakattaota-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Leelaissakattaota)
