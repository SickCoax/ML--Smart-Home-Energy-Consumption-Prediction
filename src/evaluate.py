from sklearn.metrics import r2_score , root_mean_squared_error
import numpy as np

def evaluate_score(model , X_test , y_test) :

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test , y_pred)

    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)

    rmse = root_mean_squared_error(y_test , y_pred)

    return r2 , rmse