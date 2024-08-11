import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

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

def validate():
    # Загрузите модель
    model = joblib.load('model.pkl')
    
    # Загружаем валидационные данные
    val_data = pd.read_csv('data/val.csv')
    domains = val_data['domain']
    true_labels = val_data['is_dga']
    
    # Убедимся, что true_labels является числовым типом данных
    true_labels = true_labels.astype(int)
    
    # Извлечение признаков
    X_val = extract_features(domains)
    
    # Ограничение использования памяти - делаем предсказания частями с прогресс-баром
    batch_size = 1000
    predictions = []
    
    for i in tqdm(range(0, len(X_val), batch_size), desc="Validation Progress"):
        batch = X_val[i:i+batch_size]
        # Предсказания модели также должны быть числовыми
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    
    # Убедимся, что predictions является числовым типом данных
    predictions = list(map(int, predictions))
    
    # Подсчёт метрик
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Записываем результаты в файл
    with open('validation.txt', 'w') as f:
        f.write(f"True positive: {tp}\n")
        f.write(f"False positive: {fp}\n")
        f.write(f"False negative: {fn}\n")
        f.write(f"True negative: {tn}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

if __name__ == "__main__":
    validate()
