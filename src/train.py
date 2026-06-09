import os
import pandas as pd
from preprocessing import get_X_y
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score , mean_absolute_error

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "KAG_energydata_complete.csv"
)

df = pd.read_csv(csv_path)

X , y = get_X_y(df)

X = X.drop(["Min" , "Month" , "Day"] , axis=1)

X_train , X_test , y_train , y_test = train_test_split(
    X ,
    y ,
    test_size = 0.2 ,
    random_state = 42
)

model = XGBRegressor(
    subsample = 0.8 ,
    colsample_bytree = 0.8 ,
    randome_state = 42
)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print(mean_absolute_error(y_test , y_pred))

print(r2_score(y_test , y_pred))