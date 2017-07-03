""" model_4 lets use daily data instead of weekly data """

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from sklearn.metrics import mean_squared_error
from math import sqrt

# rms = sqrt(mean_squared_error(y_actual, y_predicted))

test = pd.read_csv("data/test.csv")
rforecast = pd.read_csv("data/movingaverage_data.csv")
print(rforecast.shape)


# Converting Datetime column to date format
rforecast["Datetime"] = pd.to_datetime(rforecast["Datetime"])
test["Datetime"] = pd.to_datetime(test["Datetime"])

## Creating test_features:
test["month"] = test["Datetime"].apply(lambda x: x.month)
test["week"] = test["Datetime"].apply(lambda x: x.week)
test["year"] = test["Datetime"].apply(lambda x: x.year)


# Collecting all the unique Item_IDs
item_id = list(rforecast["Item_ID"].value_counts().index)

fe_2_use = ["ID", "Item_ID", "Price", "Number_Of_Sales"]
valid_frame=[]
test_frame = []
for ids in tqdm(item_id): #For each stock
    X = rforecast[rforecast["Item_ID"] == ids]
    x_train = X[X["year"] != 2016] # train data
    x_valid = X[X["year"] == 2016] # Validation_data
    #print(x_train.shape, x_valid.shape)

    price_train = x_train["Price"].values # Price in train file
    price_valid = x_valid["Price"].values # Price in valid file


    # For Price: Applying Grid Search for obtaining best alpha and N(moving average) using validation RMSE
    rmse = []
    params = []
    for alpha in [0.1, 0.2]:
        for n in [7, 14, 30, 60,  184]:
            pred = mv(price_train, price_valid, alpha =alpha, n = n)
            m = x_valid[["year", "week"]]
            m["Price_Predicted"] = pred[-182:]
            m["Item_ID"] = ids
            m1 = rforecast[rforecast["Item_ID"] == ids]
            m1 = m1[m1["year"] == 2016]
            m = pd.merge(m1, m, how='left', on=["year", "week", "Item_ID"])
            m1 = m[["Item_ID", "Price", "Price_Predicted"]]
            rms = sqrt(mean_squared_error(m1["Price"].values, m1["Price_Predicted"]))
            rmse.append(rms)
            params.append((alpha, n))
    if ids == item_id[0]:
        print(params, rmse)

    index = rmse.index(min(rmse)) ## Searching for best rmse (low)
    param = params[index] # FInding the params of the best rmse(low)
    pred = mv(price_train, price_valid, alpha =param[0], n = param[1]) # Training on the entire data
    sales_train = x_train["Number_Of_Sales"].values
    sales_valid = x_valid["Number_Of_Sales"].values

    # For Number_Of_Sales Applying Grid Search for obtaining best alpha and N(moving average) using validation RMSE
    rmse_sales = []
    params_sales = []
    for alpha in [0.1, 0.2]:
        for n in [7, 14, 30, 60,  184]:
            pred_sales = mv(sales_train, sales_valid, alpha =alpha, n = n)
            m = x_valid[["year", "week"]]
            m["Number_Of_Sales_Predicted"] = pred_sales[-182:]
            m["Item_ID"] = ids
            m1 = rforecast[rforecast["Item_ID"] == ids]
            m1 = m1[m1["year"] == 2016]
            m = pd.merge(m1, m, how='left', on=["year", "week", "Item_ID"])
            m1 = m[["Item_ID", "Number_Of_Sales", "Number_Of_Sales_Predicted"]]
            rms = sqrt(mean_squared_error(m1["Number_Of_Sales"].values, m1["Number_Of_Sales_Predicted"]))
            rmse_sales.append(rms)
            params_sales.append((alpha, n))
    if ids == item_id[0]:
        print(params_sales, rmse_sales)

    index = rmse_sales.index(min(rmse_sales)) ## Searching for best rmse (low)
    param_sales = params_sales[index] # FInding the params of the best rmse(low)

    pred_sales = mv(sales_train, sales_valid, alpha =param_sales[0], n = param_sales[1])  # Training on the entire data

    ## Saving the validation file to check rmse on overall (All stocks combined) data
    xx_valid = x_valid[["year", "week"]]
    xx_valid["Price_Predicted"] = pred[-182:]
    xx_valid["Number_Of_Sales_Predicted"] = pred_sales[-182:]
    xx_valid["Item_ID"] = ids
    x_valid = rforecast[rforecast["Item_ID"] == ids]
    x_valid = x_valid[x_valid["year"] == 2016]
    xx_valid["Price"] = x_valid["Price"]
    xx_valid["Number_Of_Sales"] = x_valid["Number_Of_Sales"]
    x_valid = xx_valid[["Item_ID", "Price", "Number_Of_Sales", "Price_Predicted", "Number_Of_Sales_Predicted"]]
    valid_frame.append(x_valid) ## Appending the validation data of each stock to a list

    """ Validation Completed  Now Working on Test Data """


    price = X["Price"].values # collecting price values
    X_test = test[test["Item_ID"] == ids]
    XX_test = X_test[["year", "week"]]

    pred = mv(price, XX_test,  alpha =param[0], n = param[1]) # MV on complete train data Price


    sales = X["Number_Of_Sales"].values # collecting sales values

    pred_sales = mv(sales, XX_test, alpha =param_sales[0], n = param_sales[1]) # MV on complete train data Number_Of_Sales


    XX_test["Price"] = pred[-184:] # take the last 184 observations of price
    XX_test["Number_Of_Sales"] = pred_sales[-184:] # take the last 184 observations of Number_Of_Sales
    XX_test["Item_ID"] = ids
    XX_test["ID"] = X_test["ID"]
    test_frame.append(XX_test[fe_2_use]) # Appending the test data of each stock to a list

## convert the list of valid_frame and test_frame to dataframe for final submission

sub_columns = ["ID", "Number_Of_Sales", "Price"]
submission = pd.concat(test_frame)
submission = submission[sub_columns]

print(submission.head())
submission.to_csv("submission/model_4_daily_mv.csv", index = False)


submission_2 = pd.concat(valid_frame)
submission_2.to_csv("data/model_4_valid.csv", index = False)
