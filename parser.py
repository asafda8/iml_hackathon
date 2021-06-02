import pandas as pd
import ast

genre_dict = {}
def split_json(row):
    row = row[1:-1]
    ret_dict = {}
    try:
        dict = ast.literal_eval(row)
    except:
        return {}
    try:
        for val in dict:
            print(val)
            genre_dict[val['id']] = val['name']
            ret_dict[val['id']] = val['name']
    except:
        genre_dict[dict['id']] = dict['name']
        ret_dict[dict['id']] = dict['name']
    return ret_dict




def dummy_variable_genre():
    df = pd.read_csv('C:\\Users\\asafd\\PycharmProjects\\iml_hackathon\\train.csv')
    df['genres'] = df['genres'].apply(split_json)
    print()

    def exist(dict_):
        if genre in dict_:
            return 1
        return 0

    for genre in genre_dict:
        df[str(genre) + '_genre'] = df['genres'].apply(exist)

dummy_variable_genre()