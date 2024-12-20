from sklearn.metrics import classification_report
import pandas as pd
import pickle

# Загрузка модели и признаков
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    numeric_features, categorical_features = pickle.load(f)

# Загрузка данных
train_processed = pd.read_csv('train_processed.csv')
X_train = train_processed[numeric_features + categorical_features]
y_train = train_processed['Transported']

train_predictions = model.predict(X_train)
print("\nОценка качества модели на тренировочном наборе:")
print(classification_report(y_train, train_predictions))
