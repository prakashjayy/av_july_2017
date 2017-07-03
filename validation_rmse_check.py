""" Validation file to check the validation rmse. run this after running the Makefile """

from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np

test = pd.read_csv("data/model_4_valid.csv")
print(test.shape)
print(test.head())

price = test["Price"].values
price_predicted = test["Price_Predicted"].values


sales = test["Number_Of_Sales"].values
sales_predicted = test["Number_Of_Sales_Predicted"].values

rmse_price = sqrt(mean_squared_error(price, price_predicted))
rmse_sales = sqrt(mean_squared_error(sales, sales_predicted))

print("RMSE_PRICE:," rmse_price, "RMSE_SALES:", rmse_sales)
