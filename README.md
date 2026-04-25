# 🌲 Forest Classifier Bot

Telegram bot for wine classification using **Random Forest** algorithm.

## 📋 Project Description

This project demonstrates the Random Forest algorithm applied to wine classification.
The bot allows users to input wine characteristics and get predictions in real-time.

**Assignment:** Random Forests (náhodné lesy) – učenie súborom

## 🤖 Bot Features

| Command | Description |
|---|---|
| 🍷 Предсказать вино | Step-by-step wine classification |
| 📊 Метрики модели | Model accuracy, ROC-AUC, OOB Score |
| 📈 Графики | Feature importance, confusion matrix, ROC curve, etc. |
| ℹ️ Как работает RF? | Explanation of Random Forest algorithm |

## 📊 Datasets

### 1. Wine Dataset (Primary)
- 178 samples, 13 features, 3 classes
- Accuracy: **100%** | OOB Score: **97.89%** | ROC-AUC: **100%**

### 2. Breast Cancer Dataset (Secondary)
- 569 samples, 30 features, 2 classes
- Used for comparison and demonstrating RF on larger dataset

## 🌲 How Random Forest Works

1. **Bootstrap** — each tree trains on a random sample with replacement
2. **Random subspace** — random subset of features at each split
3. **Voting** — each tree votes, majority wins
4. **OOB Score** — ~37% of data not used for training = free validation

## 📁 Project Structure
