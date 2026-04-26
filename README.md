# 🌲 Forest Classifier Bot

A Telegram bot for Italian wine classification using the **Random Forest** algorithm.

## 📋 Project Description

This project demonstrates the **Random Forest** ensemble learning algorithm applied to wine classification.
The bot allows users to input 13 chemical characteristics of wine and receive a real-time prediction
of the wine type using a trained RF model.

**Assignment topic:** Random Forests (náhodné lesy) – učenie súborom

## 🤖 Bot Features

| Button | Description |
|---|---|
| 🍷 Predict Wine | Step-by-step wine classification (13 inputs) |
| 📊 Model Metrics | Accuracy, ROC-AUC, OOB Score, parameters |
| 📈 Graphs | Feature importance, confusion matrix, ROC curve, etc. |
| ℹ️ How does RF work? | Plain-language explanation of the algorithm |

## 🍷 Wine Classes

| Class | Wine | Description |
|---|---|---|
| class_0 | **Barolo** | Bold, tannic, full-bodied |
| class_1 | **Grignolino** | Light, delicate, aromatic |
| class_2 | **Barbera** | Fruity, low tannin, high acidity |

## 📊 Datasets

### 1. Wine Dataset (Primary — used in bot)
- 178 samples, 13 features, 3 classes
- Accuracy: **100%** | OOB Score: **97.89%** | ROC-AUC: **100%**

### 2. Breast Cancer Dataset (Secondary — for comparison)
- 569 samples, 30 features, 2 classes
- Used to demonstrate RF performance on a larger, more complex dataset

## 🌲 How Random Forest Works

1. **Bootstrap** — each tree trains on a random sample with replacement (~63% of data)
2. **Random subspace** — random subset of features considered at each node split
3. **Voting** — each tree votes independently, majority class wins
4. **OOB Score** — ~37% of data not used per tree = free built-in validation

## 📁 Project Structure
```
Forest-Classifier-Bot/
│
├── bot/
│   ├── __init__.py
│   ├── bot.py
│   ├── handlers.py
│   └── keyboards.py
│
├── ml/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── models/
│   ├── rf_model.pkl
│   ├── feature_names.pkl
│   └── target_names.pkl
│
├── plots/
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── boxplot_features.png
│   ├── pairplot.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── learning_curve.png
│   ├── roc_curve.png
│   ├── single_tree.png
│   ├── depth_curve.png
│   ├── confusion_matrix_cancer.png
│   ├── feature_importance_cancer.png
│   └── comparison.png
│
├── notebooks/
│   └── exploration.ipynb
│
├── report/
│   └── explanation.md
│
├── .env
├── config.py
├── requirements.txt
└── README.md
```
## ⚙️ Installation & Setup

```bash
git clone https://github.com/yourusername/Forest-Classifier-Bot
cd Forest-Classifier-Bot
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

Train the model and generate plots:
```bash
python ml/train.py
```

Run the bot:
```bash
python -m bot.bot
```

## 📈 Model Performance

| Metric | Wine Dataset | Breast Cancer Dataset |
|---|---|---|
| Accuracy | 1.0000 | ~0.9649 |
| OOB Score | 0.9789 | ~0.9614 |
| ROC-AUC | 1.0000 | ~0.9970 |
| n_estimators | 100 | 100 |
| Features | 13 | 30 |
| Samples | 178 | 569 |

## 📊 Visualizations

| Plot | Description |
|---|---|
| `feature_importance.png` | Most predictive wine features |
| `confusion_matrix.png` | Prediction errors per class |
| `roc_curve.png` | ROC curve for all 3 classes |
| `single_tree.png` | One decision tree from the forest |
| `learning_curve.png` | Accuracy vs number of trees |
| `depth_curve.png` | Accuracy vs tree depth |
| `comparison.png` | Wine vs Breast Cancer comparison |
| `class_distribution.png` | Class balance in Wine dataset |
| `correlation_matrix.png` | Feature correlation heatmap |
| `boxplot_features.png` | Feature distributions per class |
| `pairplot.png` | Pairwise feature relationships |

## 🛠️ Technologies

| Library | Purpose |
|---|---|
| `scikit-learn` | Random Forest model, GridSearchCV |
| `aiogram 3.x` | Telegram bot framework |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `seaborn` | Visualizations |
| `joblib` | Model persistence |
| `python-dotenv` | Environment configuration |
