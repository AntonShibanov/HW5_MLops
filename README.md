
Проект демонстрирует полный цикл воспроизводимого ML-эксперимента с использованием DVC и MLflow на датасете Iris.

## Структура проекта
├── data/

│ ├── raw/ # Сырые и обработанные данные

│ └── processed/ # Обработанные данные

├── src/

│ ├── prepare.py # Подготовка данных

│ └── train.py # Обучение модели

├── model/ # Сохранённая модель

├── dvc.yaml # Описание пайплайна

├── dvc.lock # Зафиксированные версии зависимостей

├── params.yaml # Гиперпараметры

├── requirements.txt # Зависимости

└── README.md

1. Клонировать репозиторий:
   ```bash
   git clone <repo-url> && cd <repo-folder>

2. Установить зависимости:
   ```bash
   pip install -r requirements.txt

3. Получить данные из DVC-хранилища:
   ```bash
   dvc pull

4. Воспроизвести пайплайн:
   ```bash
   dvc repro

5. Запустить MLflow Tracking UI:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --allowed-hosts

6. Открыть в браузере http://localhost:5000 или публичную ссылку, полученную через тунеллирование.
7. Описание пайплайна:
    
    prepare – загрузка сырых данных data/raw/iris.csv, масштабирование признаков, сохранение в data/processed/processed.csv.
    
    train – разделение на train/test, обучение логистической регрессии, логирование параметров, метрик и артефактов в MLflow, сохранение модели model/model.pkl.

8. После запуска сервера MLflow UI доступен по адресу, выведенному в консоли.
