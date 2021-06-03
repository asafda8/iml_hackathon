import pandas as pd
import ast
from datetime import datetime
import numpy as np
import math
from json_parser_test import encode_json_column_test
from json_parser import encode_json_column
genre_dict = {}
spoken_dict = {}
character_list = {}

def split_json(row):
    row = row[1:-1]
    ret_list = set()
    try:
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            genre_dict[val['id']] = val['name']
            ret_list.add(val['id'])
    except:
        genre_dict[dict['id']] = dict['name']
        ret_list.add(dict['id'])
    return ret_list


def dummy_variable_genre(df):
    df['genres'] = df['genres'].apply(split_json)

    def exist(dict_):
        if genre in dict_:
            return 1
        return 0

    for genre in genre_dict:
        df[str(genre) + '_genre'] = df['genres'].apply(exist)
    df = df.drop('genres', axis='columns')
    return df


def dummy_variable_orig_language(df):
    df = pd.get_dummies(df, drop_first=True, columns=['original_language'])
    return df


def split_json_spoken_language(row):
    row = row[1:-1]
    ret_dict = {}
    try:
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            spoken_dict[val['english_name']] = val['iso_639_1']
            ret_dict[val['english_name']] = val['iso_639_1']
    except:
        spoken_dict[dict['english_name']] = dict['iso_639_1']
        ret_dict[dict['english_name']] = dict['iso_639_1']
    return ret_dict


def dummy_variable_spoken_languages(df):
    df['spoken_languages'] = df['spoken_languages'].apply(split_json_spoken_language)

    def exist(dict_):
        if lan in dict_:
            return 1
        return 0

    for lan in spoken_dict:
        df[str(lan) + '_spoken_languages'] = df['spoken_languages'].apply(exist)
    df = df.drop('spoken_languages', axis='columns')
    return df

def remove_unuseful(df):
    df = df.drop('id', axis ='columns')
    df = df.drop('homepage', axis='columns')
    df = df.drop('original_language', axis='columns')
    df = df.drop('status', axis='columns')
    df = df.drop('original_title', axis='columns')
    df = df.drop('overview', axis='columns')
    df = df.drop('tagline', axis='columns')
    df = df.drop('production_countries', axis='columns')
    df = df.drop('title', axis='columns')
    return df
## function for spliting year and day

def split_json_production(row):
    row = row[1:-1]
    ret_dict = {}
    try:
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            spoken_dict[val['english_name']] = val['iso_639_1']
            ret_dict[val['english_name']] = val['iso_639_1']
    except:
        spoken_dict[dict['english_name']] = dict['iso_639_1']
        ret_dict[dict['english_name']] = dict['iso_639_1']
    return ret_dict


def dummy_variable_production(df):
    df['spoken_languages'] = df['spoken_languages'].apply(split_json_spoken_language)

    def exist(dict_):
        if lan in dict_:
            return 1
        return 0

    for lan in spoken_dict:
        df[str(lan) + '_spoken_languages'] = df['spoken_languages'].apply(exist)
    df = df.drop('spoken_languages', axis='columns')
    return df



def process_time(df):
    today = datetime.today()
    tyear = today.year
    tmonth = today.month
    def time_split(row):
        try:
            first_index = row['release_date'].find('/')
            last_index = row['release_date'].rfind('/')
            month = int(row['release_date'][first_index + 1:last_index])
            year = int(row['release_date'][last_index + 1:])
            diff = tyear - year
            total_month = diff * 12 + (tmonth - month)
            return max(total_month, 0), month
        except:
            try:
                first_index = row['release_date'].find('-')
                last_index = row['release_date'].rfind('-')
                month = int(row['release_date'][first_index + 1:last_index])
                year = int(row['release_date'][:first_index])
                diff = tyear - year
                total_month = diff * 12 + (tmonth - month)
                return max(total_month, 0), month
            except:
                if row['status'] == 'released':
                    return np.nan, np.nan
                return 0, 0
    p = df[['release_date', 'status']].apply(time_split, axis=1)
    df['month_difference'] = p.apply(lambda x: x[0])
    df['month_release'] = p.apply(lambda x: x[1])
    med = df[df['month_difference'] != 0]['month_difference'].median()
    df['month_difference'] = df['month_difference'].fillna(med)
    df = df.drop('release_date', axis='columns')
    return df


def vote_count_replace(df):
    df['vote_count'] = df['vote_count'].apply(lambda x: max(0, x))
    df['vote_count'] = df['vote_count'].apply(lambda x: min(500000, x))

def split_json_actors(row):
    try:
        row = row[1:-1]
        ret_list = set()
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            ret_list.add(val['name'])
    except:
        ret_list.add(dict['name'])
    return ret_list


def top_actors(df):
    actor_df = pd.read_csv('actors.csv')
    name_col = actor_df['Name']
    df['crew'] = df['crew'].apply(split_json_actors)

    def exist(dict_):
        if actor in dict_:
            return 1
        return 0

    for actor in name_col:
        df[str(actor) + '_actor'] = df['crew'].apply(exist)
    df = df.drop('crew', axis='columns')
    return df


def dummy_variable_for_collection(df):
    df['belongs_to_collection'] = df['belongs_to_collection'].fillna(0)
    df['belongs_to_collection'].replace(regex=r'[^0]+', value=1, inplace=True)

def replace_zero_with_Nan(df):
    for feature in df.columns:
        if feature not in {'revenue'}:
            df.loc[df[feature] == 0, feature] = np.nan #TODO less than 0
    return df

def replace_with_median(df, feature):
    med = df[feature].median()
    df[feature] = df[feature].fillna(med)
    return df, med

def main_parse(train_path):
    df = pd.read_csv(train_path)
    df = df[(df.status != 'Released') | (df.revenue != 0)]
    # df = dummy_variable_genre(df)
    # df = dummy_variable_spoken_languages(df)
    # vote_count_replace(df)
    df['cast'] = df['cast'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df['crew'] = df['crew'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df['keywords'] = df['keywords'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df['production_companies'] = df['production_companies'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df['genres'] = df['genres'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df, castcols = encode_json_column(df, 19, "name", 200, 1)

    df,crewcols = encode_json_column(df, 19, "name", 200, 1)
    df, keywords_cols = encode_json_column(df, 18, "name", 200, 1)
    df, spoken_languages_cols = encode_json_column(df, 14, "english_name", 50, 1)
    df, companies_cols = encode_json_column(df, 10, "name", 200, 1)
    df, genres_cols = encode_json_column(df, 3, "name", 1000, 1)

    dummy_variable_for_collection(df)
    df = process_time(df)
    df = remove_unuseful(df)
    df.loc[df['budget'] == 0, 'budget'] = np.nan
    df.loc[((df['runtime'] <= 0)|(df['runtime'] > 300)), 'runtime'] = np.nan
    df,budget_med = replace_with_median(df, 'budget')
    df,vote_med = replace_with_median(df, 'vote_count')
    df,runtime_med = replace_with_median(df, 'runtime')
    global_lst = [castcols, crewcols, keywords_cols, spoken_languages_cols, companies_cols, genres_cols]
    return df, [castcols, crewcols, keywords_cols, spoken_languages_cols, companies_cols, genres_cols] ,\
           [budget_med, vote_med, runtime_med]


def test_parse(path, lst, lst2):
    df2 = pd.read_csv(path)
    df2['cast'] = df2['cast'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2['crew'] = df2['crew'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2['keywords'] = df2['keywords'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2['spoken_languages'] = df2['spoken_languages'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2['production_companies'] = df2['production_companies'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2['genres'] = df2['genres'].apply(lambda x: x.replace("\'", "\"") if x == x else "")
    df2 = encode_json_column_test(df2, lst[0], 19, "name", 0)

    df2 = encode_json_column_test(df2, lst[1], 19, "name",  0)
    df2 = encode_json_column_test(df2, lst[2], 18, "name",  0)
    df2 = encode_json_column_test(df2, lst[3], 14, "english_name", 0)
    df2 = encode_json_column_test(df2, lst[4], 10, "name", 0)
    df2 = encode_json_column_test(df2, lst[5], 3, "name", 0)

    dummy_variable_for_collection(df2)
    df2 = process_time(df2)
    df2 = remove_unuseful(df2)
    df2.loc[df2['budget'] == 0, 'budget'] = np.nan
    df2.loc[((df2['runtime'] <= 0) | (df2['runtime'] > 300)), 'runtime'] = np.nan
    df2['budget'] = df2['budget'].fillna(lst2[0])
    df2['vote_count'] = df2['vote_count'].fillna(lst2[1])
    df2['runtime'] = df2['runtime'].fillna(lst2[2])
    return df2

def parse_train_test(train_path, test_path):
    train_df, lst, lst2 = main_parse(train_path)
    test_df = test_parse(test_path, lst, lst2)
    return train_df, test_df