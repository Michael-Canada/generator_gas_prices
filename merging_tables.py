# environment: placebo_jupyter_env_new_python

from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import NamedTuple
import pytz
from datetime import date
import io
import os

plt.rcParams["figure.figsize"] = (10, 7)
import requests
import duckdb
import json
import jsonplus
import pickle
import numpy as np
import scipy.stats as stats

import input_reader


# upload the file 'combined_df_OLD.csv' into a DataFrame
df_old = pd.read_csv("combined_df_OLD.csv")
df = pd.read_csv("combined_df.csv")

df.drop(columns=["Unnamed: 0"], inplace=True)
df_old.drop(columns=["Unnamed: 0"], inplace=True)
df_old.drop(columns=["actual_price"], inplace=True)

# the values in the column 'month' of df_old should have a "0" beforehand in case that they are only one digit
df_old["month"] = df_old["month"].astype(str).str.zfill(2)
df_old["month"] = "2023-" + df_old["month"]
df_old.rename(columns={"month": "year_month"}, inplace=True)
df_old.rename(columns={"predicted_price": "predicted_price_OLD"}, inplace=True)
df.rename(columns={"predicted_price": "predicted_price_NEW"}, inplace=True)

# merge the tables on year_month and unit_id
df_both = pd.merge(df, df_old, on=["year_month", "unit_id"], how="right")

# add a column to df_both that will contain the difference between the predicted prices and the actual prices
df_both["diff_OLD"] = df_both["predicted_price_OLD"] - df_both["actual_price"]
df_both["diff"] = df_both["predicted_price_NEW"] - df_both["actual_price"]


df_both.to_csv("df_both.csv")

# Only look at cases where the quantity is above the 10th percentile
cutoff_quantity = df_both["Quantity"].quantile(0.1)
df_both = df_both[df_both["Quantity"] > cutoff_quantity]
df_both["diff_times_quantity_OLD"] = df_both["diff_OLD"] * df_both["Quantity"]
df_both["diff_times_quantity"] = df_both["diff"] * df_both["Quantity"]

print(
    f"Old error: {df_both['diff_times_quantity_OLD'].sum() / df_both['Quantity'].sum()}"
)
print(f"New error: {df_both['diff_times_quantity'].sum() / df_both['Quantity'].sum()}")


plt.plot(df_both["actual_price"])
plt.plot(df_both["predicted_price_OLD"])
plt.plot(df_both["predicted_price_NEW"])
plt.legend(["actual price", "Predicted Price OLD", "Predicted Price NEW"])
plt.xlabel("index")
plt.ylabel("price")
plt.title("actual price vs diff_OLD vs diff_NEW")


plt.plot(df_both["actual_price"], df_both["predicted_price_OLD"], "o")
plt.plot(df_both["actual_price"], df_both["predicted_price_NEW"], "x")
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.legend(["OLD", "NEW"])


print(18)
