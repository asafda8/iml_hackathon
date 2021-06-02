import pandas as pd
import ast

genre_dict = {}
spoken_dict = {}


def split_json(row):
    row = row[1:-1]
    ret_list = []
    try:
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            print(val)
            genre_dict[val['id']] = val['name']
            ret_list.append(val['id'])
    except:
        genre_dict[dict['id']] = dict['name']
        ret_list.append(dict['id'])
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
            print(val)
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


df = pd.read_csv('C:\\Users\\asafd\\PycharmProjects\\iml_hackathon\\train.csv')
df = dummy_variable_genre(df)
df = dummy_variable_orig_language(df)
df = dummy_variable_spoken_languages(df)
