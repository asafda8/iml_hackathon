import pandas as pd
def split_data():
    # CHANGE PATH
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