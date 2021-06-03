import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def process_data(data):
    to_drop_jason = ['belongs_to_collection', 'genres', 'production_companies',
                     'production_countries',
                     'keywords', 'cast', 'crew']
    to_drop = ['homepage', 'original_language', 'original_title',
               'overview', 'spoken_languages', 'status', 'tagline', 'title',
               'release_date', 'id']

    data = data.drop(to_drop_jason, axis=1)
    data = data.drop(to_drop, axis=1)
    print(len(list(data.columns)))
    return data

def compare_to_response(response, data):
    rows, columns = 2, 3
    i = 1
    fig = plt.figure()
    for (columnName, columnData) in data.iteritems():
        pearson_corr = np.cov(columnData, response)[0, 1] / (
                np.std(columnData) * np.std(response))
        fig.add_subplot(rows, columns, i)
        i += 1
        plot_feature_response_graph(columnName, columnData, response,
                                    pearson_corr)
    plt.show()


def compare_to_eachother(data):
    i = 1
    rows_2, columns_2 = 4, 4
    fig_2 = plt.figure()
    for (columnName_1, columnData_1) in data.iteritems():
        for (columnName_2, columnData_2) in data.iteritems():
            pearson_corr = np.cov(columnData_1, columnData_2)[0, 1] / (
                    np.std(columnData_1) * np.std(columnData_2))
            fig_2.add_subplot(rows_2, columns_2, i)
            i += 1
            plot_feature_response_graph(columnName_1, columnData_1,
                                        columnData_2,
                                        pearson_corr)
            plt.ylabel(str(columnName_2), fontsize=7)
    plt.show()


def get_to_know_data(path):
    data = pd.read_csv(path)
    response = data.pop("revenue")
    data = process_data(data)
    compare_to_response(response, data)
    compare_to_eachother(data)



def plot_feature_response_graph(feature_name, feature_data, response,
                                pearson_corr):
    plt.scatter(feature_data, response, marker='o', alpha=0.5, s=3)
    plt.title( str(feature_name) +" " +str(round(pearson_corr,2)), fontsize=7)
    # plt.title(str(feature_name) , fontsize=7)


# def try_regression(data, response):
#     response_1 = response.iloc[0]
#     data_1 = data.iloc[0]
#     print(data_1)
#     print(response_1)
#     response = response.drop(0)
#     data = data.drop(0)
#     print(data)
#     print(response)
#     reg = LinearRegression().fit(data, response)
#     answer = reg.predict(data_1)
#     print(answer , " ", response_1)
#
#
# data = pd.read_csv("train.csv")
# response = data.pop("revenue")
# data = process_data(data)
# try_regression(data, response)

def deal_with_missing_data(data):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(data)
    SimpleImputer()
    data_mean = imp_mean.transform(data)

get_to_know_data("train.csv")