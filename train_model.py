import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Загрузка данных
train_processed = pd.read_csv('train_processed.csv')

# Предобработка и создание признаков
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train = train_processed[numeric_features + categorical_features]
y_train = train_processed['Transported']

# Обучение модели
model.fit(X_train, y_train)

# Экспорт переменных и модели
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Экспорт переменных для других файлов
with open('features.pkl', 'wb') as f:
    pickle.dump((numeric_features, categorical_features), f)
