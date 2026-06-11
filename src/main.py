import os
import pandas as pd
from train import get_model
from evaluate import evaluate_score
import numpy as np

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" , 
    "KAG_energydata_complete.csv"
)
df = pd.read_csv(csv_path)

model , X_test , y_test = get_model(df)

r2 , rmse = evaluate_score(model , X_test , y_test)

y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)

print(f"Model Prediction : {y_pred}")
print()

print(f"R2 Score : {r2}")
print(f"Root Mean Squared Error : {rmse}")