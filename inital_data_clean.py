""" preparing dataset for R Forecasting Problems """

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import date, timedelta
from tqdm import tqdm

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print("train shape:",train.shape)
print("test shape:", test.shape)
sub_columns = ["ID", "Number_Of_Sales", "Price"]

# Converting Datetime column to date format
train["Datetime"] = pd.to_datetime(train["Datetime"])
test["Datetime"] = pd.to_datetime(test["Datetime"])

IDsnotintest = set(list(train["Item_ID"].values)) - set(list(test["Item_ID"].values))
print("IDs not present in test:{}".format(len(IDsnotintest)))
train = train[~train["Item_ID"].isin(list(IDsnotintest))]
print("New train shape", train.shape)


rmodel = []
for i in tqdm(list(train["Item_ID"].unique())):
    x_train = train[train["Item_ID"] == i]
    x_test = test[test["Item_ID"] == i]

    d1 = min(x_train["Datetime"])
    d2 = max(train["Datetime"])
    delta = d2 - d1
    dates = []
    for m in range(delta.days+1):
        dates.append(d1+timedelta(days=m))

    X_original = pd.DataFrame([i for j in dates], dates).reset_index()
    X_original.columns = ["Datetime", "Item_ID"]
    X_original = X_original.merge(x_train[["Item_ID", "Datetime", "ID", "Category_1", "Category_2", "Category_3", "Price", "Number_Of_Sales"]], on = ["Item_ID", "Datetime"])
    rmodel.append(X_original)


submission = pd.concat(rmodel)
submission.to_csv("data/Rdataforecast.csv", index = False)
