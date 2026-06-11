# ⚡ Energy Consumption Prediction

A machine learning regression project that predicts household appliance energy consumption using environmental and weather-related features.

## 🚀 Features

- Data cleaning and preprocessing
- Date-time feature engineering
- Log transformation of target variable
- XGBoost Regressor model
- Hyperparameter tuning
- Model evaluation using R² Score and RMSE

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## 📂 Project Structure

```text
.
├── main.py
├── preprocessing.py
├── train.py
├── evaluate.py
└── dataset/
```

## 📊 Workflow

1. Load dataset
2. Perform feature engineering
3. Transform target using `np.log1p()`
4. Train XGBoost Regressor
5. Evaluate model performance
6. Convert predictions back using `np.expm1()`

## ▶️ Run Project

```bash
python main.py
```