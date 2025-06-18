 # Camera Calibrate API

Библиотека для калибровки обычных и fisheye камер с использованием шахматной доски.

## Установка

```bash
# Установка из PyPI (когда будет опубликовано)
pip install camera-calibrate

# Установка из исходного кода
git clone https://github.com/DalekTek/camera-calibrate.git
cd camera-calibrate
pip install -e .
```

## Быстрый старт

### Калибровка обычной камеры

```python
from camera_calibrate import CameraCalibrator, setup_logger

# Настройка логирования
logger = setup_logger()

# Создание калибратора
calibrator = CameraCalibrator(logger, pattern_size=(9, 6))

# Калибровка
if calibrator.calibrate_from_images("calibration_images/"):
    # Сохранение результатов
    calibrator.save_calibration("camera_calibration.npz")
    
    # Исправление дисторсии
    undistorted = calibrator.undistort_image("test_image.jpg", "undistorted.jpg")
    print("Calibration completed successfully!")
```

### Калибровка fisheye камеры

```python
from camera_calibrate import FisheyeCalibrator, setup_logger

# Настройка логирования
logger = setup_logger()

# Создание fisheye калибратора
fisheye_calibrator = FisheyeCalibrator(logger, pattern_size=(9, 6))

# Калибровка
if fisheye_calibrator.calibrate_from_images("fisheye_images/"):
    # Сохранение результатов
    fisheye_calibrator.save_calibration("fisheye_calibration.npz")
    
    # Оценка поля зрения
    fov = fisheye_calibrator.estimate_fisheye_fov()
    print(f"Field of view: {fov:.1f}°")
    
    # Исправление дисторсии различными методами
    fisheye_calibrator.undistort_fisheye("test.jpg", "equirectangular.jpg", "equirectangular")
    fisheye_calibrator.undistort_fisheye("test.jpg", "perspective.jpg", "perspective")
```

## API Reference

### CameraCalibrator

Класс для калибровки обычных камер.

#### Конструктор

```python
CameraCalibrator(logger, pattern_size=(9, 6))
```

**Параметры:**
- `logger`: Объект логгера
- `pattern_size`: Размер шахматной доски (внутренние углы)

#### Методы

##### calibrate_from_images(images_path)

Калибровка по набору изображений.

```python
success = calibrator.calibrate_from_images("calibration_images/")
```

##### save_calibration(filename)

Сохранение параметров калибровки.

```python
calibrator.save_calibration("camera_calibration.npz")
```

##### load_calibration(filename)

Загрузка параметров калибровки.

```python
calibrator.load_calibration("camera_calibration.npz")
```

##### undistort_image(image_path, output_path=None)

Исправление дисторсии изображения.

```python
undistorted = calibrator.undistort_image("distorted.jpg", "undistorted.jpg")
```

##### get_calibration_info()

Получение информации о калибровке.

```python
info = calibrator.get_calibration_info()
print(f"Focal length: {info['focal_length']}")
```

### FisheyeCalibrator

Класс для калибровки fisheye камер.

#### Конструктор

```python
FisheyeCalibrator(logger, pattern_size=(9, 6))
```

#### Методы

##### calibrate_from_images(images_path)

Калибровка fisheye камеры.

```python
success = fisheye_calibrator.calibrate_from_images("fisheye_images/")
```

##### undistort_fisheye(image_path, output_path=None, method='equirectangular')

Исправление дисторсии fisheye изображения.

**Методы:**
- `equirectangular`: Равнопрямоугольная проекция
- `perspective`: Перспективная проекция
- `cylindrical`: Цилиндрическая проекция
- `stereographic`: Стереографическая проекция

```python
# Равнопрямоугольная проекция
fisheye_calibrator.undistort_fisheye("fisheye.jpg", "equirectangular.jpg", "equirectangular")

# Перспективная проекция
fisheye_calibrator.undistort_fisheye("fisheye.jpg", "perspective.jpg", "perspective")
```

##### estimate_fisheye_fov()

Оценка поля зрения fisheye камеры.

```python
fov = fisheye_calibrator.estimate_fisheye_fov()
print(f"Field of view: {fov:.1f}°")
```

### Утилиты

#### setup_logger(name, log_dir, level)

Настройка логгера.

```python
logger = setup_logger("my_calibration", "logs", "INFO")
```

#### create_chessboard(width, height, square_size, filename)

Создание шахматной доски.

```python
create_chessboard(9, 6, 30, "chessboard.png")
```

## Командная строка

### Калибровка обычной камеры

```bash
camera-calibrate regular --images calibration_images/ --output camera_calibration.npz
```

### Калибровка fisheye камеры

```bash
camera-calibrate fisheye --images fisheye_images/ --output fisheye_calibration.npz
```

### Создание шахматной доски

```bash
camera-calibrate create-pattern --width 9 --height 6 --output chessboard.png
```

### Исправление дисторсии

```bash
# Обычная камера
camera-calibrate undistort --calibration camera_calibration.npz --input image.jpg --output undistorted.jpg

# Fisheye камера
camera-calibrate undistort --calibration fisheye_calibration.npz --input fisheye.jpg --output undistorted.jpg --type fisheye --method equirectangular
```

### Информация о калибровке

```bash
camera-calibrate info --calibration camera_calibration.npz --type regular
```

## Примеры

### Полный пример калибровки

```python
from camera_calibrate import CameraCalibrator, setup_logger
import os

def full_calibration_example():
    # Настройка
    logger = setup_logger("full_calibration")
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    
    # Проверка наличия изображений
    images_path = "calibration_images"
    if not os.path.exists(images_path):
        logger.error(f"Directory {images_path} not found")
        return False
    
    # Калибровка
    logger.info("Starting calibration...")
    if calibrator.calibrate_from_images(images_path):
        # Сохранение
        calibrator.save_calibration("camera_calibration.npz")
        
        # Получение информации
        info = calibrator.get_calibration_info()
        logger.info(f"Calibration completed with {info['num_images']} images")
        
        # Исправление тестового изображения
        if os.path.exists("test_image.jpg"):
            calibrator.undistort_image("test_image.jpg", "undistorted.jpg")
            logger.info("Test image undistorted")
        
        return True
    else:
        logger.error("Calibration failed")
        return False

if __name__ == "__main__":
    full_calibration_example()
```

### Пакетная обработка изображений

```python
from camera_calibrate import FisheyeCalibrator, setup_logger
from pathlib import Path

def batch_undistort():
    # Загрузка калибровки
    logger = setup_logger("batch_processing")
    calibrator = FisheyeCalibrator(logger)
    calibrator.load_calibration("fisheye_calibration.npz")
    
    # Обработка всех изображений
    input_dir = Path("fisheye_images")
    output_dir = Path("undistorted_images")
    output_dir.mkdir(exist_ok=True)
    
    methods = ['equirectangular', 'perspective', 'cylindrical']
    
    for image_file in input_dir.glob("*.jpg"):
        for method in methods:
            output_file = output_dir / f"{image_file.stem}_{method}.jpg"
            calibrator.undistort_fisheye(str(image_file), str(output_file), method)
            logger.info(f"Processed {image_file.name} with {method} method")

if __name__ == "__main__":
    batch_undistort()
```

### Создание калибровочных мишеней

```python
from camera_calibrate import create_chessboard, create_calibration_target

def create_calibration_patterns():
    # Стандартная шахматная доска
    create_chessboard(9, 6, 30, "chessboard_9x6.png")
    
    # Шахматная доска с информацией
    create_calibration_target(9, 6, 30, filename="chessboard_9x6_info.png", add_info=True)
    
    # Различные размеры
    patterns = [(7, 5), (8, 6), (10, 7), (11, 8)]
    for width, height in patterns:
        create_chessboard(width, height, 25, f"chessboard_{width}x{height}.png")

if __name__ == "__main__":
    create_calibration_patterns()
```

## Структура проекта

```
camera_calibrate/
├── camera_calibrate/
│   ├── __init__.py              # Основной API
│   ├── constants.py             # Константы
│   ├── calibrators/
│   │   ├── __init__.py
│   │   ├── camera_calibrator.py # Калибратор обычной камеры
│   │   └── fisheye_calibrator.py # Калибратор fisheye камеры
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger_config.py     # Конфигурация логирования
│   │   ├── chessboard_generator.py # Генератор шахматных досок
│   │   └── image_utils.py       # Утилиты для работы с изображениями
│   └── cli.py                   # Командная строка интерфейс
├── setup.py                     # Конфигурация пакета
├── requirements.txt             # Зависимости
├── README.md                    # Документация
└── examples/                    # Примеры использования
```

## Зависимости

- Python 3.8+
- numpy>=1.21.0
- opencv-python>=4.5.0

## Лицензия

MIT License