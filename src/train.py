from preprocessing import get_X_y
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def get_model(df) :

    X , y = get_X_y(df)

    X = X.drop(["Min" , "Month" , "Day"] , axis=1) # Check in EDA notebook

    X_train , X_test , y_train , y_test = train_test_split(
        X ,
        y ,
        test_size = 0.2 ,
        random_state = 42
    )

    model = XGBRegressor(
        subsample = 0.8 ,
        colsample_bytree = 0.8 ,
        random_state = 42 ,
        max_depth = 6 ,
        n_estimators = 321 ,
        min_child_weight = 4 ,
        gamma = 0.0372 ,
        learning_rate = 0.4228 ,
        reg_alpha = 5.0993 ,
        reg_lambda = 39.4425
    )

    # HyperParameter Tunning done in Notebook

    model.fit(X_train , y_train)

    return model , X_test , y_test