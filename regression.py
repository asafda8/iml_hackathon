import pandas as pd
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

    #your code goes here...

    pass


def split_data():
    file_path = 'C:\\Users\\asafd\\PycharmProjects\\iml_hackathon\\movies_dataset.csv'
    df = pd.read_csv(file_path)
    train = df.sample(frac = 0.6)
    indices = train.index
    new_df = df.loc[~df.index.isin(indices)]
    validation = new_df.sample(frac=0.5)
    test = new_df.loc[~new_df.index.isin(validation.index)]
    train.to_csv('train.csv')
    validation.to_csv('validation.csv')
    test.to_csv('test.csv')

split_data()