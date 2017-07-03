""" moving average dataset """

import numpy as np
import pandas as pd
from tqdm import tqdm

test = pd.read_csv("data/test.csv")
rforecast = pd.read_csv("data/Rdataforecast.csv")
print(rforecast.shape)
print(rforecast.head())

# Converting Datetime column to date format
rforecast["Datetime"] = pd.to_datetime(rforecast["Datetime"])
test["Datetime"] = pd.to_datetime(test["Datetime"])

dates = rforecast["Datetime"].value_counts().index
dates = pd.DataFrame(dates)
dates.columns = ["Datetime"]
dates = dates.sort_values(["Datetime"])
dates.head()

dates["month"] = dates["Datetime"].apply(lambda x: x.month)
dates["week"] = dates["Datetime"].apply(lambda x: x.week)
dates["year"] = dates["Datetime"].apply(lambda x: x.year)

rforecast["month"] = rforecast["Datetime"].apply(lambda x: x.month)
rforecast["week"] = rforecast["Datetime"].apply(lambda x: x.week)
rforecast["year"] = rforecast["Datetime"].apply(lambda x: x.year)


item_id = list(rforecast["Item_ID"].value_counts().index)
print(len(item_id))

fe_2_use = ["Item_ID", "Datetime", "month", "year", "Price", "Number_Of_Sales"]
data = []
for i in tqdm(item_id):
    X = rforecast[rforecast["Item_ID"] == i]
    m = pd.merge(dates, X[fe_2_use], how='left', on=["Datetime", "month", "year"])
    m["Item_ID"] = m["Item_ID"].fillna(i)
    m = m.fillna(0)
    data.append(m)

submission = pd.concat(data)
submission.to_csv("data/movingaverage_data.csv", index = False)
