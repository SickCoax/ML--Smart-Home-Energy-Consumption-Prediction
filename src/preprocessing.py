import numpy as np

def get_X_y(df) :
    X = df.drop(["rv1" , "rv2"] , axis = 1)

    X["Time"] = (X["date"]).str[11:16]

    X["Hour"] = (X["Time"].str[0 : 2]).astype(int)
    X["Min"] = (X["Time"].str[3 :]).astype(int)
    X["Month"] = (X["date"].str[5 : 7]).astype(int)
    X["Day"] = (X["date"].str[8 : 10]).astype(int)

    X = X.drop(["date" , "Appliances" , "Time"] , axis = 1)

    y = df["Appliances"]

    y = np.log1p(y)

    return X , y