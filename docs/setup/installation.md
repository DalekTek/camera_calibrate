 # Установка и настройка

## Системные требования

### Минимальные требования
- **ОС**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 или выше
- **RAM**: 4 GB
- **Место на диске**: 2 GB

### Рекомендуемые требования
- **ОС**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 или выше
- **RAM**: 8 GB
- **Место на диске**: 5 GB
- **GPU**: NVIDIA GPU с CUDA поддержкой (опционально)

## Установка Python

### Windows
```bash
# Скачайте Python с официального сайта
# https://www.python.org/downloads/

# Проверка установки
python --version
pip --version
```

### macOS
```bash
# Используя Homebrew
brew install python

# Или скачайте с официального сайта
# https://www.python.org/downloads/macos/
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## Создание виртуального окружения

### Windows
```bash
# Создание виртуального окружения
python -m venv .venv

# Активация
.venv\Scripts\activate

# Проверка
python --version
pip list
```

### macOS/Linux
```bash
# Создание виртуального окружения
python3 -m venv .venv

# Активация
source .venv/bin/activate

# Проверка
python --version
pip list
```

## Установка зависимостей

### Основные зависимости
```bash
# Обновление pip
pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt
```

### Альтернативная установка
```bash
# Установка по отдельности
pip install numpy>=1.21.0
pip install opencv-python>=4.5.0

# Для дополнительных возможностей
pip install opencv-contrib-python>=4.5.0
```

### Установка с CUDA поддержкой (опционально)
```bash
# Для NVIDIA GPU
pip install opencv-python-cuda>=4.5.0
```

## Проверка установки

### Тест OpenCV
```python
import cv2
import numpy as np

# Проверка версии
print(f"OpenCV version: {cv2.__version__}")

# Проверка доступных модулей
print(f"Fisheye support: {'fisheye' in dir(cv2)}")

# Создание тестового изображения
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
print("OpenCV работает корректно!")
```

### Тест NumPy
```python
import numpy as np

# Проверка версии
print(f"NumPy version: {np.__version__}")

# Тест базовых операций
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy работает: {arr.mean()}")
```

## Настройка проекта

### Создание структуры директорий
```bash
# Создание необходимых папок
mkdir calibration_images
mkdir fisheye_calibration_images
mkdir logs
mkdir my_files
```

### Настройка логирования
```python
# Создание файла logger_config.py
import logging
import os
from datetime import datetime

def setup_logger(name='camera_calibrator', log_dir='logs'):
    """Настройка логгера"""
    
    # Создание директории для логов
    os.makedirs(log_dir, exist_ok=True)
    
    # Создание логгера
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Файловый обработчик
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Добавление обработчиков
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

## Решение проблем установки

### Ошибка: "Microsoft Visual C++ 14.0 is required"
```bash
# Решение для Windows
# Скачайте и установите Microsoft Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Или используйте предкомпилированные пакеты
pip install --only-binary=all opencv-python
```

### Ошибка: "Permission denied"
```bash
# Решение для Linux/macOS
sudo pip install opencv-python

# Или используйте пользовательскую установку
pip install --user opencv-python
```

### Ошибка: "ImportError: No module named 'cv2'"
```bash
# Проверьте активацию виртуального окружения
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Переустановите OpenCV
pip uninstall opencv-python
pip install opencv-python
```

### Ошибка: "OpenCV was not built with fisheye support"
```bash
# Установите opencv-contrib-python
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Проверка работоспособности

### Тест калибровки
```python
from camera_calibrator import CameraCalibrator
from logger_config import setup_logger

# Настройка логгера
logger = setup_logger()

# Создание калибратора
calibrator = CameraCalibrator(logger)

# Проверка инициализации
print("Калибратор создан успешно!")
print(f"Размер паттерна: {calibrator.pattern_size}")
```

### Тест fisheye калибровки
```python
from fisheye_camera_calibrator import FisheyeCalibrator

# Создание fisheye калибратора
fisheye_calibrator = FisheyeCalibrator(logger)

# Проверка поддержки fisheye
print("Fisheye калибратор создан успешно!")
```