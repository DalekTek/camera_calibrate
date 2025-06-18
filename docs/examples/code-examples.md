 # Примеры кода

## Базовые примеры

### 1. Простая калибровка обычной камеры

```python
#!/usr/bin/env python3
"""
Простая калибровка обычной камеры
"""

import os
import numpy as np
from camera_calibrator import CameraCalibrator
from logger_config import setup_logger

def simple_calibration():
    """Простая калибровка обычной камеры"""
    
    # Настройка логгера
    logger = setup_logger("simple_calibration")
    
    # Создание калибратора
    # pattern_size=(9, 6) означает 9x6 внутренних углов шахматной доски
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    
    # Путь к изображениям
    images_path = "calibration_images"
    
    # Проверка наличия изображений
    if not os.path.exists(images_path):
        logger.error(f"Директория {images_path} не найдена!")
        return False
    
    # Выполнение калибровки
    logger.info("Начинаем калибровку...")
    
    if calibrator.calibrate_from_images(images_path):
        # Сохранение результатов
        calibrator.save_calibration("simple_calibration.npz")
        logger.info("Калибровка завершена успешно!")
        return True
    else:
        logger.error("Калибровка не удалась!")
        return False

if __name__ == "__main__":
    simple_calibration()
```

### 2. Калибровка с анализом качества

```python
#!/usr/bin/env python3
"""
Калибровка с подробным анализом качества
"""

import cv2
import numpy as np
from camera_calibrator import CameraCalibrator
from logger_config import setup_logger

def quality_calibration():
    """Калибровка с анализом качества"""
    
    logger = setup_logger("quality_calibration")
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    
    # Выполнение калибровки
    if calibrator.calibrate_from_images("calibration_images"):
        
        # Анализ результатов
        camera_matrix = calibrator.camera_matrix
        dist_coeffs = calibrator.dist_coeffs
        
        logger.info("=== Анализ результатов калибровки ===")
        
        # Проверка фокусного расстояния
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        fx_ratio = fx / fy
        
        logger.info(f"Фокусное расстояние fx: {fx:.2f}")
        logger.info(f"Фокусное расстояние fy: {fy:.2f}")
        logger.info(f"Отношение fx/fy: {fx_ratio:.3f}")
        
        if 0.9 < fx_ratio < 1.1:
            logger.info("✓ Фокусное расстояние стабильно")
        else:
            logger.warning("⚠ Фокусное расстояние нестабильно")
        
        # Проверка главной точки
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        image_width = calibrator.image_size[0]
        image_height = calibrator.image_size[1]
        
        logger.info(f"Главная точка: ({cx:.1f}, {cy:.1f})")
        logger.info(f"Размер изображения: {image_width}x{image_height}")
        
        # Проверка расположения главной точки
        cx_ratio = cx / image_width
        cy_ratio = cy / image_height
        
        if 0.4 < cx_ratio < 0.6 and 0.4 < cy_ratio < 0.6:
            logger.info("✓ Главная точка в центре изображения")
        else:
            logger.warning("⚠ Главная точка смещена от центра")
        
        # Анализ коэффициентов дисторсии
        k1, k2, p1, p2, k3 = dist_coeffs[0]
        
        logger.info("Коэффициенты дисторсии:")
        logger.info(f"k1 (радиальная дисторсия): {k1:.6f}")
        logger.info(f"k2 (радиальная дисторсия): {k2:.6f}")
        logger.info(f"p1 (тангенциальная дисторсия): {p1:.6f}")
        logger.info(f"p2 (тангенциальная дисторсия): {p2:.6f}")
        logger.info(f"k3 (радиальная дисторсия): {k3:.6f}")
        
        # Оценка силы дисторсии
        distortion_strength = abs(k1) + abs(k2) + abs(k3)
        
        if distortion_strength < 0.1:
            logger.info("✓ Слабая дисторсия")
        elif distortion_strength < 0.5:
            logger.info("✓ Умеренная дисторсия")
        else:
            logger.warning("⚠ Сильная дисторсия")
        
        # Сохранение результатов
        calibrator.save_calibration("quality_calibration.npz")
        
        return True
    
    return False

if __name__ == "__main__":
    quality_calibration()
```

### 3. Калибровка fisheye камеры

```python
#!/usr/bin/env python3
"""
Калибровка fisheye камеры с анализом поля зрения
"""

import cv2
import numpy as np
from fisheye_camera_calibrator import FisheyeCalibrator
from logger_config import setup_logger

def fisheye_calibration():
    """Калибровка fisheye камеры"""
    
    logger = setup_logger("fisheye_calibration")
    calibrator = FisheyeCalibrator(logger, pattern_size=(9, 6))
    
    # Выполнение калибровки
    if calibrator.calibrate_from_images("fisheye_calibration_images"):
        
        # Анализ fisheye параметров
        camera_matrix = calibrator.camera_matrix
        dist_coeffs = calibrator.dist_coeffs
        
        logger.info("=== Анализ fisheye камеры ===")
        
        # Коэффициенты fisheye дисторсии
        k1, k2, k3, k4 = dist_coeffs.flatten()
        
        logger.info("Fisheye коэффициенты дисторсии:")
        logger.info(f"k1: {k1:.6f}")
        logger.info(f"k2: {k2:.6f}")
        logger.info(f"k3: {k3:.6f}")
        logger.info(f"k4: {k4:.6f}")
        
        # Определение типа fisheye
        if k1 < -0.1:
            fisheye_type = "Сильный fisheye эффект"
        elif k1 < 0:
            fisheye_type = "Умеренный fisheye эффект"
        else:
            fisheye_type = "Слабый fisheye эффект"
        
        logger.info(f"Тип камеры: {fisheye_type}")
        
        # Расчет поля зрения
        fov_x = 2 * np.arctan(calibrator.image_size[0] / (2 * camera_matrix[0, 0])) * 180 / np.pi
        fov_y = 2 * np.arctan(calibrator.image_size[1] / (2 * camera_matrix[1, 1])) * 180 / np.pi
        
        logger.info(f"Поле зрения по горизонтали: {fov_x:.1f}°")
        logger.info(f"Поле зрения по вертикали: {fov_y:.1f}°")
        
        # Оценка максимального поля зрения
        max_fov = calibrator.estimate_fisheye_fov()
        if max_fov:
            logger.info(f"Максимальное поле зрения: {max_fov:.1f}°")
        
        # Сохранение результатов
        calibrator.save_calibration("fisheye_calibration.npz")
        
        return True
    
    return False

if __name__ == "__main__":
    fisheye_calibration()
```

## Продвинутые примеры

### 4. Сравнение методов исправления дисторсии

```python
#!/usr/bin/env python3
"""
Сравнение различных методов исправления дисторсии fisheye камеры
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fisheye_camera_calibrator import FisheyeCalibrator
from logger_config import setup_logger

def compare_undistortion_methods():
    """Сравнение методов исправления дисторсии"""
    
    logger = setup_logger("undistortion_comparison")
    
    # Загрузка параметров калибровки
    calibrator = FisheyeCalibrator(logger)
    if not calibrator.load_calibration("fisheye_calibration.npz"):
        logger.error("Не удалось загрузить параметры калибровки")
        return
    
    # Тестовое изображение
    input_image = "test_fisheye.jpg"
    
    if not os.path.exists(input_image):
        logger.error(f"Тестовое изображение не найдено: {input_image}")
        return
    
    # Методы исправления дисторсии
    methods = {
        'equirectangular': 'Равнопрямоугольная проекция',
        'perspective': 'Перспективная проекция',
        'cylindrical': 'Цилиндрическая проекция',
        'stereographic': 'Стереографическая проекция'
    }
    
    # Создание исправленных версий
    results = {}
    
    for method, description in methods.items():
        logger.info(f"Применяем метод: {description}")
        
        output_path = f"undistorted_{method}.jpg"
        undistorted = calibrator.undistort_fisheye(input_image, output_path, method)
        
        if undistorted is not None:
            results[method] = {
                'image': undistorted,
                'description': description,
                'path': output_path
            }
            logger.info(f"✓ Сохранено: {output_path}")
        else:
            logger.warning(f"✗ Ошибка при применении метода {method}")
    
    # Создание сравнительного изображения
    if len(results) > 1:
        create_comparison_image(input_image, results)
    
    return results

def create_comparison_image(original_path, results):
    """Создание сравнительного изображения"""
    
    # Загрузка оригинального изображения
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Подготовка изображений для сравнения
    images = [original_rgb]
    titles = ['Оригинал']
    
    for method, data in results.items():
        img_rgb = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        titles.append(data['description'])
    
    # Создание сетки изображений
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Скрытие пустых подграфиков
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('undistortion_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Сравнительное изображение сохранено: undistortion_comparison.png")

if __name__ == "__main__":
    compare_undistortion_methods()
```

### 5. Интерактивная калибровка

```python
#!/usr/bin/env python3
"""
Интерактивная калибровка с визуализацией процесса
"""

import cv2
import numpy as np
import os
from camera_calibrator import CameraCalibrator
from logger_config import setup_logger

def interactive_calibration():
    """Интерактивная калибровка с визуализацией"""
    
    logger = setup_logger("interactive_calibration")
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    
    # Создание окна для отображения
    cv2.namedWindow("Calibration Progress", cv2.WINDOW_NORMAL)
    
    # Обработка изображений с визуализацией
    images_path = "calibration_images"
    image_files = []
    
    if os.path.isdir(images_path):
        image_files = [f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    successful_images = []
    failed_images = []
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(images_path, filename)
        
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            failed_images.append(filename)
            continue
        
        # Поиск углов
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, calibrator.pattern_size, None)
        
        # Визуализация
        display_img = img.copy()
        
        if ret:
            # Уточнение углов
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Отрисовка углов
            cv2.drawChessboardCorners(display_img, calibrator.pattern_size, corners2, ret)
            
            # Добавление информации
            cv2.putText(display_img, f"Image {i+1}/{len(image_files)}: SUCCESS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            successful_images.append(filename)
            
            # Сохранение точек
            calibrator.object_points.append(calibrator.objp)
            calibrator.image_points.append(corners2)
            
        else:
            # Отрисовка неудачного изображения
            cv2.putText(display_img, f"Image {i+1}/{len(image_files)}: FAILED", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            failed_images.append(filename)
        
        # Отображение прогресса
        cv2.imshow("Calibration Progress", display_img)
        
        # Ожидание нажатия клавиши
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    
    # Вывод результатов
    logger.info(f"Обработано изображений: {len(image_files)}")
    logger.info(f"Успешно: {len(successful_images)}")
    logger.info(f"Неудачно: {len(failed_images)}")
    
    if failed_images:
        logger.warning("Неудачные изображения:")
        for filename in failed_images:
            logger.warning(f"  - {filename}")
    
    # Выполнение калибровки
    if len(successful_images) >= 10:
        logger.info("Выполняем калибровку...")
        
        if calibrator.calibrate_from_images(images_path):
            calibrator.save_calibration("interactive_calibration.npz")
            logger.info("✓ Калибровка завершена успешно!")
            return True
        else:
            logger.error("✗ Калибровка не удалась!")
            return False
    else:
        logger.error("Недостаточно успешных изображений для калибровки!")
        return False

if __name__ == "__main__":
    interactive_calibration()
```

### 6. Пакетная обработка изображений

```python
#!/usr/bin/env python3
"""
Пакетная обработка изображений с калибровкой
"""

import os
import cv2
import numpy as np
from pathlib import Path
from camera_calibrator import CameraCalibrator
from fisheye_camera_calibrator import FisheyeCalibrator
from logger_config import setup_logger

def batch_process_images(input_dir, output_dir, camera_type="regular"):
    """Пакетная обработка изображений"""
    
    logger = setup_logger("batch_processing")
    
    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)
    
    # Выбор калибратора
    if camera_type.lower() == "fisheye":
        calibrator = FisheyeCalibrator(logger)
        calibration_file = "fisheye_calibration.npz"
    else:
        calibrator = CameraCalibrator(logger)
        calibration_file = "regular_calibration.npz"
    
    # Загрузка параметров калибровки
    if not calibrator.load_calibration(calibration_file):
        logger.error(f"Не удалось загрузить параметры калибровки: {calibration_file}")
        return False
    
    # Поиск изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    logger.info(f"Найдено изображений: {len(image_files)}")
    
    # Обработка изображений
    processed_count = 0
    failed_count = 0
    
    for image_path in image_files:
        try:
            # Определение выходного пути
            output_path = Path(output_dir) / f"undistorted_{image_path.name}"
            
            # Исправление дисторсии
            if camera_type.lower() == "fisheye":
                undistorted = calibrator.undistort_fisheye(
                    str(image_path), str(output_path), method='equirectangular'
                )
            else:
                undistorted = calibrator.undistort_image(
                    str(image_path), str(output_path)
                )
            
            if undistorted is not None:
                processed_count += 1
                logger.info(f"✓ Обработано: {image_path.name}")
            else:
                failed_count += 1
                logger.warning(f"✗ Ошибка обработки: {image_path.name}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"✗ Исключение при обработке {image_path.name}: {e}")
    
    # Вывод результатов
    logger.info("=== Результаты пакетной обработки ===")
    logger.info(f"Всего изображений: {len(image_files)}")
    logger.info(f"Обработано успешно: {processed_count}")
    logger.info(f"Ошибок обработки: {failed_count}")
    
    return processed_count > 0

def create_processing_report(input_dir, output_dir, camera_type):
    """Создание отчета о обработке"""
    
    report_path = Path(output_dir) / "processing_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Отчет о пакетной обработке изображений\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Тип камеры: {camera_type}\n")
        f.write(f"Входная директория: {input_dir}\n")
        f.write(f"Выходная директория: {output_dir}\n")
        f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Статистика файлов
        input_files = list(Path(input_dir).glob("*"))
        output_files = list(Path(output_dir).glob("*"))
        
        f.write(f"Файлов во входной директории: {len(input_files)}\n")
        f.write(f"Файлов в выходной директории: {len(output_files)}\n")
    
    print(f"Отчет сохранен: {report_path}")

if __name__ == "__main__":
    # Пример использования
    input_directory = "raw_images"
    output_directory = "processed_images"
    camera_type = "fisheye"  # или "regular"
    
    success = batch_process_images(input_directory, output_directory, camera_type)
    
    if success:
        create_processing_report(input_directory, output_directory, camera_type)
        print("Пакетная обработка завершена успешно!")
    else:
        print("Ошибка при пакетной обработке!")
```