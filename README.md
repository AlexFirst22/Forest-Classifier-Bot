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
Forest-Classifier-Bot/
├── bot/
│   ├── bot.py              # Main bot entry point
│   ├── handlers.py         # FSM handlers + all commands
│   └── keyboards.py        # Reply and inline keyboards
├── ml/
│   ├── train.py            # Model training + saving
│   ├── predict.py          # Prediction function
│   └── visualize.py        # Plot generation
├── models/
│   ├── rf_model.pkl        # Trained Random Forest model
│   ├── feature_names.pkl   # Feature names
│   └── target_names.pkl    # Target class names
├── plots/                  # Generated visualizations (PNG)
├── notebooks/
│   └── exploration.ipynb   # EDA + training + analysis
├── report/
│   └── explanation.md      # Project explanation (SK)
├── config.py               # Token loader
├── requirements.txt
└── README.md