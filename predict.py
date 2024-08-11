import pandas as pd
import joblib

# Функция для извлечения признаков из домена
def extract_features(domains):
    features = []
    for domain in domains:
        length = len(domain)
        num_digits = sum(c.isdigit() for c in domain)
        num_special_chars = sum(c in '-_' for c in domain)
        features.append([length, num_digits, num_special_chars])
    return features

def predict():
    # Загрузите модель
    model = joblib.load('model.pkl')
    
    # Загружаем тестовые данные
    test_data = pd.read_csv('data/test.csv')
    domains = test_data['domain']
    
    X_test = extract_features(domains)
    predictions = model.predict(X_test)
    
    # Записываем результаты в prediction.csv
    output = pd.DataFrame({'domain': domains, 'is_dga': predictions})
    output.to_csv('prediction.csv', index=False)

if __name__ == "__main__":
    predict()
