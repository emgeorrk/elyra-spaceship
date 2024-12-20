import pandas as pd
import pickle

# Загрузка данных
test_processed = pd.read_csv('test_processed.csv')

# Загрузка модели и признаков
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    numeric_features, categorical_features = pickle.load(f)

# Предсказания модели
X_test = test_processed[numeric_features + categorical_features]
predictions = model.predict(X_test)

# Создание файла с результатами
submission = pd.DataFrame({
    'PassengerId': test_processed['PassengerId'],
    'Transported': predictions
})

# Сохранение результатов
submission.to_csv('submission.csv', index=False)
