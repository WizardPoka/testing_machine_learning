import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Функция для извлечения признаков из домена
def extract_features(domains):
    features = []
    for domain in domains:
        domain = str(domain)  # Убедимся, что домен - строка
        length = len(domain)
        num_digits = sum(c.isdigit() for c in domain)
        num_special_chars = sum(c in '-_' for c in domain)
        features.append([length, num_digits, num_special_chars])
    return features

# Загружаем данные и обучаем модель
def train():
    # Загружаем обучающие данные
    train_data = pd.read_csv('data/val.csv')

    # Извлекаем домены и метки
    domains = train_data['domain'].astype(str)  # Преобразуем все домены в строки
    labels = train_data['is_dga']

    # Извлечение признаков
    X = extract_features(domains)
    y = labels
    
    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Сохраняем модель
    joblib.dump(model, 'model.pkl')

if __name__ == "__main__":
    train()
