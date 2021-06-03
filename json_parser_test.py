import pandas as pd
import json
import operator


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except:
        return False
    return True


def encode_json_column_test(pandas_data_frame, name_lst, json_column_index=0, json_id_column="id",
                       remove_non_encoded=1):
    X = pandas_data_frame.iloc[:, :].values


    # keep track of whether a column has been encoded into the dataframe already, else we'd reset all the values to 0
    df_encodedcolumns = []

    count = 0
    for name in name_lst:
        pandas_data_frame[name] = 0

    # for each row in the data
    for row in X:

        # keep track of whether this row can be kept or not, based on if it has an encoded value
        has_an_encoded_value = 0

        if (is_json(row[json_column_index])):  # some data is just not json. ignore
            json_l = json.loads(row[json_column_index])
            # for each feature in the json
            for popular_name_from_training in name_lst:

                # pick out its id (the json identifier you specifc in json_id_column)
                # featureid = json_features[json_id_column]

                if popular_name_from_training in json_l:

                    # if this id hasn't been seen yet, add it to the dataframe with default 0

                    pandas_data_frame.loc[count, popular_name_from_training] = 1

        count += 1

    # drop the original json column
    pandas_data_frame = pandas_data_frame.drop(pandas_data_frame.columns[json_column_index], 1)

    return pandas_data_frame