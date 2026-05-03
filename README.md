# 💳 Credit Risk Bot

A Telegram bot for credit risk assessment using the **Random Forest** algorithm.

## 📋 Project Description

This project demonstrates the **Random Forest** ensemble learning algorithm 
applied to a real-world credit risk classification problem.
The bot analyzes 10 personal and financial characteristics and predicts 
whether a loan application is likely to be **approved** or flagged as **high risk**.

**Assignment topic:** Random Forests (náhodné lesy) – učenie súborom

## 🤖 Bot Features

| Button | Description |
|---|---|
| 💳 Check Credit Risk | 10-step loan application analysis |
| 📊 Model Metrics | Accuracy, ROC-AUC, OOB Score |
| 📈 Graphs | Feature importance, confusion matrix, ROC curve, etc. |
| ℹ️ How does RF work? | Plain-language explanation of the algorithm |

## 📊 Dataset

**Credit Risk Dataset** (Kaggle)
- 32,000+ real loan records
- 11 features (age, income, loan amount, grade, intent, etc.)
- Binary classification: `0` = Approved, `1` = Default

## 📁 Project Structure
```
Forest-Classifier-Bot/
│
│
├── bot/
│   ├── init.py
│   ├── bot.py
│   ├── handlers.py
│   └── keyboards.py
│
├── ml/
│   ├── init.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── models/
│   ├── rf_model.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
│
├── plots/
│   └── *.png (8 visualizations)
│
├── data/
│   └── credit_risk_dataset.csv
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

| Metric | Value |
|---|---|
| Accuracy | 93.50% |
| ROC-AUC | 93.75% |
| OOB Score | 93.01% |
| n_estimators | 100 |
| Features | 11 |
| Training records | ~25,600 |

## 🛠️ Technologies

| Library | Purpose |
|---|---|
| `scikit-learn` | Random Forest, GridSearchCV, metrics |
| `aiogram 3.x` | Telegram bot framework |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `seaborn` | Visualizations |
| `joblib` | Model persistence |
| `python-dotenv` | Configuration |

