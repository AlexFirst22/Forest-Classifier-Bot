# Project Explanation — Random Forests (náhodné lesy)

## 1. Algorithm Overview

Random Forest is an ensemble machine learning method that builds a large number
of decision trees during training and outputs the class that receives the most votes.

### Core principles:

**Bootstrap Aggregation (Bagging)**
Each tree is trained on a random sample of the dataset drawn with replacement.
On average, ~63% of samples appear in each bootstrap sample,
while the remaining ~37% form the out-of-bag (OOB) set.

**Random Feature Subspace**
At every node split, only a random subset of features is evaluated.
This decorrelates individual trees and reduces variance of the ensemble.

**Majority Voting**
Each tree independently predicts a class label.
The final prediction is determined by majority vote across all trees.

**OOB Score**
Since each tree never sees its OOB samples during training,
these samples serve as a built-in validation set — no separate cross-validation needed.

---

## 2. Datasets Used

### Wine Dataset (Primary)
- **Source:** `sklearn.datasets.load_wine` (UCI Machine Learning Repository)
- **Samples:** 178
- **Features:** 13 chemical characteristics (alcohol, flavanoids, proline, etc.)
- **Classes:** 3 Italian wines — Barolo, Grignolino, Barbera
- **Used for:** Bot predictions, all main visualizations

### Breast Cancer Dataset (Secondary)
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 (cell nucleus measurements)
- **Classes:** 2 (Malignant, Benign)
- **Used for:** Comparison, demonstrating RF scalability

---

## 3. Effectiveness Metrics

| Metric | Description | Wine | Breast Cancer |
|---|---|---|---|
| Accuracy | Ratio of correct predictions | 1.0000 | ~0.9649 |
| Precision | True positives / predicted positives | 1.0000 | ~0.96 |
| Recall | True positives / actual positives | 1.0000 | ~0.96 |
| F1-score | Harmonic mean of precision and recall | 1.0000 | ~0.96 |
| ROC-AUC | Area under the ROC curve | 1.0000 | ~0.9970 |
| OOB Score | Out-of-bag validation accuracy | 0.9789 | ~0.9614 |

---

## 4. Hyperparameter Analysis

### n_estimators (number of trees)
Tested values: 5, 10, 20, 50, 100, 150, 200
- More trees → more stable predictions
- Returns diminish after ~100 trees
- Best balance at n_estimators=100

### max_depth (tree depth)
Tested values: 1, 2, 3, 5, 10, None
- Shallow trees → underfitting
- Deep trees → potential overfitting
- None (full depth) works best for Wine dataset

### GridSearchCV
Full grid search over n_estimators, max_depth, min_samples_split
with 5-fold cross-validation to find optimal hyperparameters.

---

## 5. Visualizations

| Plot | What it shows |
|---|---|
| Feature Importance | Alcohol, flavanoids and proline are the most predictive |
| Confusion Matrix | All 36 test samples correctly classified |
| ROC Curve | AUC = 1.0 for all 3 classes |
| Single Tree | Structure of one tree from the forest (max_depth=3) |
| Learning Curve | Accuracy stabilizes around 50-100 trees |
| Depth Curve | Effect of max_depth on train/test accuracy |
| Comparison | RF performs well on both datasets |

---

## 6. Functional Application — Telegram Bot

The bot was built using **aiogram 3.x** and implements the following workflow:

1. User starts the bot with `/start`
2. User selects **🍷 Predict Wine**
3. Bot asks for 13 chemical values one by one (FSM — Finite State Machine)
4. Values are passed to the trained RF model via `ml/predict.py`
5. Bot returns the predicted wine type (Barolo / Grignolino / Barbera),
   confidence percentage, and individual tree votes
6. User can also view all model graphs directly in the chat
7. Algorithm explanation is available in plain language

### Key implementation details:
- **FSM (Finite State Machine)** — manages multi-step user input
- **joblib** — loads pre-trained model from disk
- **FSInputFile** — sends locally generated PNG plots to Telegram
- **InlineKeyboardMarkup** — interactive graph selection menu

---

## 7. Technologies

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | latest | RandomForestClassifier, GridSearchCV, metrics |
| aiogram | 3.x | Telegram bot framework |
| pandas | latest | Data manipulation |
| numpy | latest | Numerical operations |
| matplotlib | latest | Plot generation |
| seaborn | latest | Statistical visualizations |
| joblib | latest | Model serialization |
| python-dotenv | latest | Environment variable management |