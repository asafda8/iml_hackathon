from training_parser import *
from xgboost import XGBRegressor
import pandas as pd
import ast
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


train , validation = parse_train_test('train.csv', 'validation.csv')
# train = main_parse('train.csv') # load train
# validation = main_parse('validation.csv') # load validation

##------------- mine!!! ----------##
alpha_list = [0.01, 0.1, 1, 10, 100]

def crossValid(ridge_models, lasso_models, train_sets, train_response):
    """
    calculate cross val score for each model.
    """
    MSE_ridge = []
    MSE_lasso = []
    for model in ridge_models:
        answer = cross_val_score(estimator=model, X=train_sets, y=train_response, cv=5)
        MSE_ridge.append(np.mean(answer))

    for model in lasso_models:
        answer = cross_val_score(estimator=model, X=train_sets, y=train_response, cv=5)
        MSE_lasso.append(np.mean(answer))
    print(MSE_ridge, MSE_lasso)
    return MSE_ridge, MSE_lasso

def mse_plot(MSE_ridge, MSE_lasso ):
    """
    plot the mse of each model.
    """
    plt.plot(alpha_list, MSE_ridge, label="ridge")
    plt.plot(alpha_list, MSE_lasso, label="lasso")
    plt.legend()
    plt.show()


def my_mse(ridge_models, lasso_models, train_sets, train_response, test_set, test_response):
    """
    fit the models and calculate the mse for each one.
    """
    MSE_ridge = []
    MSE_lasso = []
    for model in ridge_models:
        model.fit(train_sets, train_response)
        predicted = model.predict(test_set)
        MSE_ridge.append(r2_score(test_response, predicted))

    for model in lasso_models:
        model.fit(train_sets, train_response)
        predicted = model.predict(test_set)
        MSE_lasso.append(r2_score(test_response, predicted))

    regressor = XGBRegressor(colsample_bytree=0.6, gamma=0.7, max_depth=4,
                             min_child_weight=5,
                             subsample=0.8, objective='reg:squarederror')
    regressor.fit(train_sets, train_response)
    predicted_1 = regressor.predict(test_set)
    print("his = ", np.square(mse(test_response, predicted_1)))

    linear = LinearRegression().fit(train_sets, train_response)
    predicted_2 = linear.predict(test_set)
    array = np.arange(1,len(test_response)+1)
    # plt.scatter(array, predicted, s=1, label="predicted")
    plt.scatter(array, np.log(np.abs(test_response-predicted)), s=1, label="test_response")
    plt.legend()
    plt.show()
    print("linear regression = ", np.square(mse(test_response, predicted)))

    forest = RandomForestRegressor().fit(train_sets, train_response)
    predicted = forest.predict(test_set)
    print("forest = ", r2_score(test_response, predicted))

    print("ridge = " ,MSE_ridge)
    print("lasso = " , MSE_lasso)
    return MSE_ridge, MSE_lasso

def creat_models():
    ridge_models, lasso_models = [], []
    for a in alpha_list:
        ridge_models.append(Ridge(alpha=a))
        lasso_models.append(Lasso(alpha=a))
    return ridge_models, lasso_models

def final(data, validation):
    to_drop = ['vote_average']

    data = data.drop(to_drop, axis=1)
    train_sets = data.dropna(axis=0)
    train_response = train_sets.pop("revenue")

    validation = validation.drop(to_drop, axis=1)
    test_sets = validation.dropna(axis=0)
    test_response = test_sets.pop("revenue")

    ridge_models, lasso_models = creat_models()
    MSE_ridge, MSE_lasso = my_mse(ridge_models, lasso_models, train_sets, train_response, test_sets, test_response)
    mse_plot(MSE_ridge, MSE_lasso)



def linear_regression(data , validation):

    to_drop_jason = ['production_companies','production_countries', 'keywords', 'cast', 'crew']
    to_drop = ['original_title', 'overview', 'status', 'tagline', 'title']

    data = data.drop(to_drop_jason, axis=1)
    data = data.drop(to_drop, axis=1)
    train_sets = data.dropna(axis=0)
    validation = validation.drop(to_drop_jason, axis=1)
    validation = validation.drop(to_drop, axis=1)
    test_sets = validation.dropna(axis=0)

    train_sets.pop("vote_average")
    test_sets.pop("vote_average")
    # test_sets, train_sets = split_data(data)
    test_response = test_sets.pop("revenue")
    train_response = train_sets.pop("revenue")

    # rg = LinearRegression()
    # ridge = RidgeCV()
    # lasso = LassoCV()

    # rg.fit(train_sets, train_response)
    # ridge.fit(train_sets, train_response)
    # lasso.fit(train_sets, train_response)

    # all_accuracies_ridge = cross_val_score(estimator=ridge, X=data,
    #                                  y=response, cv=5)
    # all_accuracies_rg = cross_val_score(estimator=rg, X=data,
    #                                        y=response, cv=5)
    # all_accuracies_lasso = cross_val_score(estimator=lasso, X=data,
    #                                        y=response, cv=5)

    # print(all_accuracies_ridge.mean())
    # print(all_accuracies_rg.mean())
    # print(all_accuracies_lasso.mean())


    # predicted_linear = rg.predict(test_sets)
    # predicted_ridge = ridge.predict(test_sets)
    # predicted_lasso = lasso.predict(test_sets)


    regressor = XGBRegressor(colsample_bytree=0.6, gamma=0.7, max_depth=4,
                             min_child_weight=5,
                             subsample=0.8, objective='reg:squarederror')
    regressor.fit(train_sets, train_response)

    y_pred = regressor.predict(test_sets)
    print(r2_score(test_response, y_pred))
    # print("linear regression = ", np.sqrt(mse(test_response, predicted_linear)))
    # print("ridge = ",np.sqrt(mse(test_response, predicted_ridge)))
    # print("lasso = ",np.sqrt(mse(test_response, predicted_lasso)))
    # print("weird = ",np.sqrt(mse(test_response, y_pred)))




def split_data(X):
    '''
    this function splits the data into train and test (1/4) set randomly
    :param X: the dataframe to split
    :return: test_sets and train_sets
    '''

    train_sets, test_sets = train_test_split(X, test_size=0.25)

    return test_sets, train_sets


def mse(response, prediction):
    '''
    (1/m)sum(y_i_hat - y_i)^2
    :param response: vector
    :param prediction: vector
    :return: the MSE over the received samples
    '''

    prediction = prediction.flatten()

    # (prediction-response)^2
    inner_calculation = (prediction - response)
    inner_calculation = inner_calculation * inner_calculation
    return np.mean(inner_calculation)


final(train, validation)
