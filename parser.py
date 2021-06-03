import pandas as pd
import ast
from datetime import datetime
import numpy as np
import math
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


def main_parse():
    df = pd.read_csv('train.csv')
    df = dummy_variable_genre(df)
    df = dummy_variable_spoken_languages(df)
    df = remove_unuseful(df)
    df = process_time(df)
    vote_count_replace(df)
    top_actors(df)
    dummy_variable_for_collection(df)
    df.to_csv('dummies.csv')
