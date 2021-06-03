import pandas as pd
from sklearn.linear_model import LinearRegression
import training_parser
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection

################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # parse data:
    train_df, test_df = training_parser.parse_train_test('all_train.csv', 'all_validation.csv')
    train_df = train_df.dropna()
    # is_NaN = train_df.isnull()
    # row_has_NaN = is_NaN.any(axis=1)
    # rows_with_NaN = train_df[row_has_NaN]

    # resque labels from data:
    # train_rev = train_df.pop('revenue')
    train_votes = train_df.pop('vote_average')
    test_rev = test_df.pop('revenue')
    test_votes = test_df.pop('vote_average')
    train_rev =  train_df.pop('revenue')
    # resque unreleased movies:
    train_released = train_df['month_difference'] != 0
    test_not_released = test_df['month_difference'] == 0

    # models:

    # Linear Reg:
    # linear_reg = LinearRegression().fit(train_df, train_rev)
    # pred = linear_reg.predict(test_df)
    # print(linear_reg.score(train_df, train_rev))
    # # pred = linear_reg.predict(test_df)
    # print(linear_reg.score(test_df, test_rev))
    # print(r2_score(test_rev, pred))

    # XGBRegressor:

    regres = XGBRegressor(colsample_bytree=0.6, gamma=0.7, max_depth=4,
                             min_child_weight=5,
                             subsample=0.8, objective='reg:squarederror')
    regres.fit(train_df, train_rev)
    regres_pred = regres.predict(test_df)
    regres_pred[test_not_released] = 0
    print(mean_squared_error(test_rev, regres_pred) ** 0.5)



    rating_regres = XGBRegressor(colsample_bytree=0.6, gamma=0.7, max_depth=4,
                             min_child_weight=5,
                             subsample=0.8, objective='reg:squarederror')
    rating_regres.fit(train_df, train_votes)
    reges_vote_pred = rating_regres.predict(test_df)
    reges_vote_pred[test_not_released] = 0
    print(mean_squared_error(test_votes, reges_vote_pred) ** 0.5)

    print()


    pass
predict('f')


