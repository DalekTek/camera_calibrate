 # Частые ошибки и их решения

## Ошибки при установке

### 1. "Microsoft Visual C++ 14.0 is required"

**Описание**: Ошибка при установке OpenCV на Windows.

**Причина**: Отсутствуют инструменты сборки Microsoft Visual C++.

**Решение**:
```bash
# Вариант 1: Установка Build Tools
# Скачайте и установите Microsoft Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Вариант 2: Использование предкомпилированных пакетов
pip install --only-binary=all opencv-python

# Вариант 3: Использование conda
conda install opencv
```

### 2. "Permission denied" при установке

**Описание**: Ошибка доступа при установке пакетов.

**Причина**: Недостаточно прав для установки в системные директории.

**Решение**:
```bash
# Вариант 1: Использование виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Вариант 2: Пользовательская установка
pip install --user opencv-python

# Вариант 3: Использование sudo (Linux/macOS)
sudo pip install opencv-python
```

### 3. "ImportError: No module named 'cv2'"

**Описание**: Python не может найти модуль OpenCV.

**Причина**: OpenCV не установлен или установлен в неправильном окружении.

**Решение**:
```python
# Проверка установки
import sys
print(sys.path)  # Проверьте пути Python

# Переустановка OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python

# Проверка версии
import cv2
print(cv2.__version__)
```

## Ошибки при калибровке

### 1. "No images found!"

**Описание**: Калибратор не может найти изображения для калибровки.

**Причина**: Неправильный путь к изображениям или отсутствие файлов.

**Решение**:
Проверьте:
1. Существует ли директория
2. Правильный ли формат файлов (.jpg, .png, .jpeg)
3. Права доступа к файлам

### 2. "No corners found in image"

**Описание**: Не удается найти углы шахматной доски на изображении.

**Причины и решения**:

#### Плохое освещение
```python
# Улучшение освещения
import cv2
import numpy as np

def improve_lighting(image):
    """Улучшение освещения изображения"""
    
    # Конвертация в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Объединение каналов
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

# Применение улучшения
enhanced_image = improve_lighting(original_image)
```

#### Неправильный размер паттерна
```python
# Проверка размера паттерна
def check_pattern_size(image_path, pattern_size=(9, 6)):
    """Проверка правильности размера паттерна"""
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Попробуем разные размеры
    sizes_to_try = [(9, 6), (8, 6), (7, 5), (6, 4)]
    
    for size in sizes_to_try:
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            print(f"✓ Найдены углы для размера {size}")
            return size
    
    print("✗ Не удалось найти углы ни для одного размера")
    return None
```

#### Размытие изображения
```python
# Проверка резкости
def check_image_sharpness(image):
    """Проверка резкости изображения"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f"Резкость изображения: {laplacian_var:.2f}")
    
    if laplacian_var < 100:
        print("⚠ Изображение размыто!")
        return False
    else:
        print("✓ Изображение достаточно резкое")
        return True
```

### 3. "Calibration failed"

**Описание**: Процесс калибровки завершился неудачно.

**Причины и решения**:

#### Недостаточно изображений
```python
# Проверка количества изображений
def check_calibration_requirements(image_points, min_images=10):
    """Проверка требований для калибровки"""
    
    if len(image_points) < min_images:
        print(f"⚠ Недостаточно изображений: {len(image_points)} < {min_images}")
        print("Рекомендации:")
        print("1. Сделайте больше снимков (15-20)")
        print("2. Разнообразьте углы съемки")
        print("3. Убедитесь в качестве изображений")
        return False
    
    print(f"✓ Достаточно изображений: {len(image_points)}")
    return True
```

#### Плохое качество изображений
```python
# Анализ качества изображений
def analyze_image_quality(image_paths):
    """Анализ качества изображений"""
    
    quality_scores = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Проверка резкости
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Проверка контраста
            contrast = gray.std()
            
            # Общий балл качества
            quality = (sharpness / 1000) + (contrast / 50)
            quality_scores.append((path, quality))
    
    # Сортировка по качеству
    quality_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Топ-5 лучших изображений:")
    for i, (path, score) in enumerate(quality_scores[:5]):
        print(f"{i+1}. {os.path.basename(path)}: {score:.2f}")
    
    return quality_scores
```

### 4. "OpenCV(4.11.0) error: (-215:Assertion failed) _src1.sameSize(_src2)"

**Описание**: Ошибка при расчете ошибки репроекции в fisheye калибровке.

**Причина**: Несоответствие размеров массивов при сравнении точек.

**Решение**:
```python
# Исправленная версия расчета ошибки репроекции
def calculate_reprojection_error_safe(object_points, image_points, rvecs, tvecs, 
                                    camera_matrix, dist_coeffs):
    """Безопасный расчет ошибки репроекции"""
    
    total_error = 0
    valid_images = 0
    
    for i in range(len(object_points)):
        try:
            # Проекция точек
            imgpoints2, _ = cv2.fisheye.projectPoints(
                object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            
            # Проверка размеров
            if image_points[i].shape != imgpoints2.shape:
                print(f"⚠ Несоответствие размеров в изображении {i}")
                continue
            
            # Приведение к одинаковой форме
            imgpoints2 = imgpoints2.reshape(-1, 2)
            image_points_flat = image_points[i].reshape(-1, 2)
            
            # Расчет ошибки
            error = cv2.norm(image_points_flat, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            valid_images += 1
            
        except Exception as e:
            print(f"⚠ Ошибка при расчете для изображения {i}: {e}")
            continue
    
    if valid_images > 0:
        mean_error = total_error / valid_images
        return mean_error, valid_images
    else:
        return None, 0
```

## Ошибки при исправлении дисторсии

### 1. "First perform calibration or load parameters!"

**Описание**: Попытка исправить дисторсию без загруженных параметров калибровки.

**Решение**:
```python
def safe_undistort(calibrator, image_path, output_path):
    """Безопасное исправление дисторсии"""
    
    # Проверка наличия параметров калибровки
    if calibrator.camera_matrix is None:
        print("Ошибка: Параметры калибровки не загружены!")
        print("Сначала выполните калибровку или загрузите параметры:")
        print("calibrator.load_calibration('calibration.npz')")
        return None
    
    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл {image_path} не найден!")
        return None
    
    # Исправление дисторсии
    try:
        undistorted = calibrator.undistort_image(image_path, output_path)
        return undistorted
    except Exception as e:
        print(f"Ошибка при исправлении дисторсии: {e}")
        return None
```

### 2. "Failed to load image"

**Описание**: Не удается загрузить изображение для исправления дисторсии.

**Решение**:
```python
def load_image_safe(image_path):
    """Безопасная загрузка изображения"""
    
    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return None
    
    # Проверка размера файла
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        print(f"Файл пустой: {image_path}")
        return None
    
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        print("Возможные причины:")
        print("1. Неподдерживаемый формат файла")
        print("2. Поврежденный файл")
        print("3. Недостаточно памяти")
        return None
    
    print(f"✓ Изображение загружено: {img.shape}")
    return img
```

## Ошибки логирования

### 1. "Permission denied" при записи логов

**Описание**: Нет прав на запись в директорию логов.

**Решение**:
```python
def setup_logger_safe(name='camera_calibrator', log_dir='logs'):
    """Безопасная настройка логгера"""
    
    try:
        # Создание директории с проверкой прав
        os.makedirs(log_dir, exist_ok=True)
        
        # Проверка прав на запись
        test_file = os.path.join(log_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
    except PermissionError:
        print(f"⚠ Нет прав на запись в {log_dir}")
        print("Используем временную директорию")
        log_dir = '/tmp' if os.name != 'nt' else os.environ.get('TEMP', '.')
    
    # Настройка логгера
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Только консольный вывод
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

## Общие рекомендации по отладке

### 1. Включение подробного логирования
```python
import logging

# Настройка подробного логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Логирование в файл
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
```

### 2. Проверка окружения
```python
def check_environment():
    """Проверка окружения"""
    
    import sys
    import cv2
    import numpy as np
    
    print("=== Проверка окружения ===")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # Проверка поддержки fisheye
    if hasattr(cv2, 'fisheye'):
        print("✓ Поддержка fisheye: есть")
    else:
        print("✗ Поддержка fisheye: нет")
    
    # Проверка доступной памяти
    import psutil
    memory = psutil.virtual_memory()
    print(f"Доступная память: {memory.available / 1024**3:.1f} GB")
```

### 3. Создание минимального примера
```python
def create_minimal_example():
    """Создание минимального примера для тестирования"""
    
    # Создание тестового изображения
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (540, 380), (255, 255, 255), 2)
    
    # Сохранение
    cv2.imwrite("test_image.jpg", img)
    
    # Простая калибровка
    from camera_calibrator import CameraCalibrator
    from logger_config import setup_logger
    
    logger = setup_logger()
    calibrator = CameraCalibrator(logger)
    
    print("Минимальный пример создан")
    print("Проверьте файл test_image.jpg")
```

## Получение помощи

Если проблема не решена:

1. **Проверьте логи**: Изучите файлы в директории `logs/`
2. **Создайте минимальный пример**: Упростите код до минимума
3. **Проверьте версии**: Убедитесь в совместимости версий библиотек