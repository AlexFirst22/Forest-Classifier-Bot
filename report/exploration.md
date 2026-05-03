# Project Explanation — Random Forests (náhodné lesy)

## 1. Algorithm Overview

Random Forest is an ensemble machine learning method that builds many decision 
trees and combines their predictions by majority voting.
Each tree is trained differently, which makes the ensemble more robust 
and accurate than any single tree.

### Core Principles

**Bootstrap Aggregation (Bagging)**
Each of the 100 trees is trained on a random sample of the dataset 
drawn with replacement. On average ~63% of records appear in each 
bootstrap sample. The remaining ~37% form the out-of-bag (OOB) set 
used for free validation.

**Random Feature Subspace**
At every node split, only a random subset of features is evaluated.
For 11 features, typically sqrt(11) ≈ 3 features are considered per split.
This decorrelates the trees and reduces variance of the ensemble.

**Majority Voting**
Each tree independently predicts: Approved (0) or Default (1).
The final prediction is determined by majority vote across all 100 trees.

**OOB Score**
Each tree validates itself on its OOB samples — records it never saw 
during training. This gives a free, unbiased accuracy estimate 
without needing a separate validation set.

---

## 2. Dataset

**Credit Risk Dataset** — Kaggle
- 32,581 real loan records
- 11 features per record
- Binary target: 0 = Approved, 1 = Default (21.8% default rate)

### Features Used

| Feature | Type | Description |
|---|---|---|
| person_age | Numeric | Applicant age |
| person_income | Numeric | Annual income ($) |
| person_home_ownership | Categorical | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | Numeric | Years of employment |
| loan_intent | Categorical | Purpose of loan |
| loan_grade | Categorical | Credit grade A-G |
| loan_amnt | Numeric | Loan amount ($) |
| loan_int_rate | Numeric | Interest rate (%) |
| loan_percent_income | Numeric | loan_amnt / person_income |
| cb_person_default_on_file | Categorical | Previous default Y/N |
| cb_person_cred_hist_length | Numeric | Years of credit history |

### Data Preprocessing
- Removed outliers: age > 100, employment > 60 years
- Filled missing values with median (loan_int_rate, emp_length)
- Encoded categorical features with LabelEncoder
- loan_percent_income calculated automatically from income and loan amount

---

## 3. Effectiveness Metrics

| Metric | Value | Description |
|---|---|---|
| Accuracy | 93.50% | Overall correct predictions |
| Precision (Default) | 97% | When model says Default, it's right 97% of the time |
| Recall (Default) | 72% | Model catches 72% of all actual defaults |
| F1-score | 0.83 | Harmonic mean of precision and recall |
| ROC-AUC | 93.75% | Model's ability to distinguish classes |
| OOB Score | 93.01% | Free validation on unseen data |

---

## 4. Visualizations

| Plot | What it shows |
|---|---|
| Feature Importance | loan_percent_income and loan_grade are top predictors |
| Confusion Matrix | Model correctly classifies 99% of approved, 72% of defaults |
| ROC Curve | AUC = 0.9375, strong discrimination ability |
| Learning Curve | Accuracy stabilizes around 50-100 trees |
| Class Distribution | 78.2% approved vs 21.8% default |
| Correlation Matrix | Interest rate and loan grade are strongly correlated |
| Feature Distributions | Income and interest rate differ significantly by class |
| Categorical Features | Grade G and previous defaults = highest risk |

---

## 5. Functional Application — Telegram Bot

The bot implements a complete loan assessment pipeline:

### Input Collection (FSM)
The bot uses aiogram's Finite State Machine to collect 10 inputs:
- Numeric inputs: age, income, employment, loan amount, interest rate, credit history
- Button inputs: home ownership, loan intent, loan grade, previous default

### Automatic Feature Engineering
loan_percent_income is calculated automatically:
loan_percent_income = loan_amnt / person_income
The user never enters this directly.

### Categorical Encoding
LabelEncoder transforms text categories to numbers using the same 
encoders that were fitted during training — ensuring consistency.

### Prediction
The trained Random Forest model outputs:
- Class prediction: 0 (Approved) or 1 (Default)
- Probability scores: e.g. 87% Approved, 13% Default

### Result Explanation
If denied, the bot shows:
- Top 3 features with highest importance from the model
- Concrete tips: reduce loan amount, improve grade, build credit history

---

## 6. Technologies

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | latest | RandomForestClassifier, LabelEncoder, metrics |
| aiogram | 3.x | Telegram bot, FSM state management |
| pandas | latest | Data loading and preprocessing |
| numpy | latest | Numerical operations |
| matplotlib | latest | Plot generation |
| seaborn | latest | Statistical visualizations |
| joblib | latest | Model and encoder serialization |
| python-dotenv | latest | Token management |