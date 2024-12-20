import pandas as pd

def preprocess_data(df):
    df = df.copy()

    if 'Cabin' in df.columns:
        df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)

    df[['Group', 'Number']] = df['PassengerId'].str.split('_', expand=True)

    numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Применение к данным
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

train_processed.to_csv('train_processed.csv', index=False)
test_processed.to_csv('test_processed.csv', index=False)
