 # Быстрый старт

Это руководство поможет вам быстро начать работу с проектом калибровки камеры.

## Шаг 1: Подготовка шахматной доски

### Требования к шахматной доске
- **Размер**: Рекомендуется 9x6 внутренних углов
- **Материал**: Жесткий, не гнущийся
- **Печать**: Высокое качество, четкие границы
- **Размер квадрата**: 20-30 мм (для удобства)

### Создание шахматной доски
```python
import cv2
import numpy as np

def create_chessboard(width=9, height=6, square_size=30, filename="chessboard.png"):
    """Создание шахматной доски для печати"""
    
    # Размер изображения
    img_width = (width + 1) * square_size
    img_height = (height + 1) * square_size
    
    # Создание изображения
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # Рисование шахматной доски
    for i in range(height + 1):
        for j in range(width + 1):
            if (i + j) % 2 == 0:
                y1, y2 = i * square_size, (i + 1) * square_size
                x1, x2 = j * square_size, (j + 1) * square_size
                img[y1:y2, x1:x2] = 0
    
    # Сохранение
    cv2.imwrite(filename, img)
    print(f"Chessboard saved: {filename}")

# Создание доски 9x6
create_chessboard(9, 6, 30, "chessboard_9x6.png")
```

## Шаг 2: Съемка изображений

### Рекомендации по съемке
```python
# Примеры углов съемки
angles = [
    "Прямо (0°)",
    "Под углом 15°",
    "Под углом 30°",
    "Под углом 45°",
    "Сбоку (90°)",
    "Сверху",
    "Снизу"
]

# Примеры расстояний
distances = [
    "Близко (20-30 см)",
    "Среднее (50-70 см)",
    "Далеко (1-1.5 м)"
]
```

### Минимальное количество изображений
- **Обычная камера**: 10-15 изображений
- **Fisheye камера**: 15-20 изображений

### Пример скрипта для съемки
```python
import cv2
import os
from datetime import datetime

def capture_calibration_images(camera_id=0, output_dir="calibration_images", num_images=15):
    """Съемка изображений для калибровки"""
    
    # Создание директории
    os.makedirs(output_dir, exist_ok=True)
    
    # Открытие камеры
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Failed to open camera")
        return
    
    image_count = 0
    
    while image_count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Отображение инструкций
        cv2.putText(frame, f"Images: {image_count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, ESC to exit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibration Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            # Сохранение изображения
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"calib_{timestamp}_{image_count:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            image_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {image_count} images")

# Запуск съемки
capture_calibration_images()
```

## Шаг 3: Калибровка обычной камеры

### Базовый пример
```python
from camera_calibrator import CameraCalibrator
from logger_config import setup_logger
import os

def calibrate_regular_camera():
    """Калибровка обычной камеры"""
    
    # Настройка логгера
    logger = setup_logger("regular_calibration")
    
    # Создание калибратора
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    
    # Путь к изображениям
    images_path = "calibration_images"
    
    # Проверка наличия изображений
    if not os.path.exists(images_path):
        logger.error(f"Directory {images_path} not found!")
        return False
    
    # Выполнение калибровки
    logger.info("Starting regular camera calibration...")
    
    if calibrator.calibrate_from_images(images_path):
        # Сохранение результатов
        calibrator.save_calibration("regular_camera_calibration.npz")
        logger.info("Regular camera calibration completed successfully!")
        return True
    else:
        logger.error("Regular camera calibration failed!")
        return False

# Запуск калибровки
if __name__ == "__main__":
    calibrate_regular_camera()
```

## Шаг 4: Калибровка fisheye камеры

### Базовый пример
```python
from fisheye_camera_calibrator import FisheyeCalibrator
from logger_config import setup_logger

def calibrate_fisheye_camera():
    """Калибровка fisheye камеры"""
    
    # Настройка логгера
    logger = setup_logger("fisheye_calibration")
    
    # Создание fisheye калибратора
    calibrator = FisheyeCalibrator(logger, pattern_size=(9, 6))
    
    # Путь к изображениям
    images_path = "fisheye_calibration_images"
    
    # Выполнение калибровки
    logger.info("Starting fisheye camera calibration...")
    
    if calibrator.calibrate_from_images(images_path):
        # Сохранение результатов
        calibrator.save_calibration("fisheye_camera_calibration.npz")
        
        # Оценка поля зрения
        fov = calibrator.estimate_fisheye_fov()
        logger.info(f"Оценка поля зрения: {fov:.1f}°")
        
        logger.info("Fisheye calibration completed successfully!")
        return True
    else:
        logger.error("Fisheye calibration failed!")
        return False

# Запуск калибровки
if __name__ == "__main__":
    calibrate_fisheye_camera()
```

## Шаг 5: Исправление дисторсии

### Обычная камера
```python
def undistort_regular_image():
    """Исправление дисторсии обычной камеры"""
    
    # Загрузка параметров калибровки
    calibrator = CameraCalibrator(logger)
    if not calibrator.load_calibration("regular_camera_calibration.npz"):
        print("Error loading calibration parameters")
        return
    
    # Исправление изображения
    input_image = "test_image.jpg"
	
    # Подготовка директории для сохранения исправленных изображений
    output_dir = "undistorted_perspective"
    os.makedirs(output_dir, exist_ok=True)
	
    undistorted = calibrator.undistort_image(input_image, output_dir)
    
    if undistorted is not None:
        print(f"Image corrected: {output_image}")
        
        # Показать сравнение
        calibrator.show_comparison(input_image, undistorted)
```

### Fisheye камера
```python
def undistort_fisheye_image():
    """Исправление дисторсии fisheye камеры"""
    
    # Загрузка параметров калибровки
    calibrator = FisheyeCalibrator(logger)
    if not calibrator.load_calibration("fisheye_camera_calibration.npz"):
        print("Error loading calibration parameters")
        return
    
    # Исправление разными методами
    input_image = "test_fisheye.jpg"
    output_dir = "undistorted_perspective"
	os.makedirs(output_dir, exist_ok=True)
	
    methods = ['equirectangular', 'perspective', 'cylindrical']
    
    undistorted = calibrator.undistort_fisheye(input_image, output_dir, "perspective")
    if undistorted is not None:
        print(f"✓ Created: {output_path} ({description})")
    
	# Показать 
	calibrator.show_undistortion_comparison(input_image, undistorted)
```

## Шаг 6: Проверка результатов

### Анализ качества калибровки
```python
def analyze_calibration_quality():
    """Анализ качества калибровки"""
    
    # Загрузка параметров
    data = np.load("regular_camera_calibration.npz")
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print("=== Calibration Quality Analysis ===")
    print(f"Focal length fx: {camera_matrix[0, 0]:.2f}")
    print(f"Focal length fy: {camera_matrix[1, 1]:.2f}")
    print(f"Principal point cx: {camera_matrix[0, 2]:.2f}")
    print(f"Principal point cy: {camera_matrix[1, 2]:.2f}")
    
    print("\nDistortion coefficients:")
    for i, coeff in enumerate(dist_coeffs[0]):
        print(f"k{i+1}: {coeff:.6f}")
    
    # Оценка качества
    fx_ratio = camera_matrix[0, 0] / camera_matrix[1, 1]
    if 0.9 < fx_ratio < 1.1:
        print("✓ Focal length is stable")
    else:
        print("⚠ Focal length is unstable")
```

## Полный пример использования

```python
#!/usr/bin/env python3
"""
Полный пример калибровки камеры
"""

import os
from camera_calibrator import CameraCalibrator
from fisheye_camera_calibrator import FisheyeCalibrator
from logger_config import setup_logger

def main():
    """Основная функция"""
    
    # Настройка логгера
    logger = setup_logger("full_calibration")
    
    logger.info("=== Starting the calibration process ===")
    
    # 1. Калибровка обычной камеры
    logger.info("1. Regular camera calibration")
    regular_calibrator = CameraCalibrator(logger)
    
    if regular_calibrator.calibrate_from_images("calibration_images"):
        regular_calibrator.save_calibration("regular_calibration.npz")
        logger.info("✓ Regular camera calibrated")
    else:
        logger.error("✗ Error calibrating regular camera")
    
    # 2. Калибровка fisheye камеры
    logger.info("2. Fisheye camera calibration")
    fisheye_calibrator = FisheyeCalibrator(logger)
    
    if fisheye_calibrator.calibrate_from_images("fisheye_calibration_images"):
        fisheye_calibrator.save_calibration("fisheye_calibration.npz")
        fisheye_calibrator.estimate_fisheye_fov()
        logger.info("✓ Fisheye camera calibrated")
    else:
        logger.error("✗ Error calibrating fisheye camera")
    
    logger.info("=== Calibration process completed ===")

if __name__ == "__main__":
    main()
```