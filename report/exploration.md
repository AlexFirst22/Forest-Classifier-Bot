# Vysvetlenie fungovania programu — Random Forests

## 1. Algoritmus Random Forest

Random Forest je metóda strojového učenia založená na princípe **ensemble learningu**.
Namiesto jedného rozhodovacieho stromu vytvorí veľký počet stromov (les)
a výsledok určí hlasovaním.

### Kľúčové vlastnosti:
- **Bootstrap agregácia (Bagging)** — každý strom sa učí na náhodnej podvzorke dát
- **Náhodný výber príznakov** — pri každom rozdelení uzla sa vyberá náhodná podmnožina príznakov
- **OOB Score** — ~37% dát sa nepoužije na tréning → bezplatná validácia

## 2. Datasety

### Wine Dataset
- 178 vzoriek, 13 príznakov, 3 triedy vín
- Accuracy: 100%, OOB: 97.89%, ROC-AUC: 100%

### Breast Cancer Dataset
- 569 vzoriek, 30 príznakov, 2 triedy
- Väčší dataset pre porovnanie výkonu RF

## 3. Metriky efektívnosti

| Metrika | Popis |
|---|---|
| Accuracy | Podiel správnych predpovedí |
| Precision | Presnosť pozitívnych predpovedí |
| Recall | Úplnosť pozitívnych predpovedí |
| F1-score | Harmonický priemer Precision a Recall |
| ROC-AUC | Plocha pod ROC krivkou |
| OOB Score | Validácia na out-of-bag vzorkách |

## 4. Vizualizácie

- **Feature Importance** — ktoré príznaky sú najdôležitejšie
- **Confusion Matrix** — kde model robí chyby
- **ROC Curve** — kompromis medzi TPR a FPR
- **Learning Curve** — vplyv počtu stromov na presnosť
- **Single Tree** — vizualizácia jedného stromu z lesa
- **Depth Curve** — vplyv hĺbky stromu na presnosť
- **Comparison** — porovnanie Wine vs Breast Cancer

## 5. Funkčná aplikácia — Telegram Bot

Bot umožňuje:
1. Zadať 13 chemických vlastností vína
2. Získať predpoveď triedy vína (class_0, class_1, class_2)
3. Zobraziť percent istoty a hlasovanie stromov
4. Zobraziť všetky vizualizácie priamo v chate
5. Prečítať vysvetlenie algoritmu

## 6. Technológie

- `scikit-learn` — RandomForestClassifier
- `aiogram 3.x` — Telegram bot
- `GridSearchCV` — optimalizácia hyperparametrov
- `matplotlib`, `seaborn` — vizualizácie