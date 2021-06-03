import pandas as pd
import numpy as np

MAX_MONTH = 36


def fit_rev_per_month(df):
    month_difference = df.pop('month_difference')
    df['revenue'].div(min(month_difference, MAX_MONTH))
    df['revenue'] = df['revenue'].fillna(0)
    # fit the model


def predict_rev_per_month(test_set_path):
    df = pd.read_csv(test_set_path)
    month_difference = df.pop('month_difference', axis='columns')

    y = np.zero(df.shape[0])
    # call the predict function , y = column of revenues
    y = y * month_difference


fit_rev_per_month('train.csv')
predict_rev_per_month('validation.csv')
