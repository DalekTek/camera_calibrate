 """
Генератор шахматных досок для калибровки камеры.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def create_chessboard(width: int = 9, 
                     height: int = 6, 
                     square_size: int = 30, 
                     margin: int = 50,
                     filename: Optional[str] = None) -> np.ndarray:
    """
    Создание шахматной доски для печати.
    
    Args:
        width: Количество внутренних углов по ширине
        height: Количество внутренних углов по высоте
        square_size: Размер квадрата в пикселях
        margin: Отступ от краев в пикселях
        filename: Путь для сохранения файла (опционально)
        
    Returns:
        np.ndarray: Изображение шахматной доски
    """
    
    # Размер изображения (включая внешние углы)
    img_width = (width + 1) * square_size + 2 * margin
    img_height = (height + 1) * square_size + 2 * margin
    
    # Создание изображения
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # Рисование шахматной доски
    for i in range(height + 1):
        for j in range(width + 1):
            if (i + j) % 2 == 0:
                y1 = margin + i * square_size
                y2 = margin + (i + 1) * square_size
                x1 = margin + j * square_size
                x2 = margin + (j + 1) * square_size
                img[y1:y2, x1:x2] = 0
    
    # Сохранение файла
    if filename:
        cv2.imwrite(filename, img)
        print(f"Chessboard saved: {filename}")
    
    return img


def create_calibration_target(width: int = 9,
                            height: int = 6,
                            square_size: int = 30,
                            margin: int = 50,
                            add_info: bool = True,
                            filename: Optional[str] = None) -> np.ndarray:
    """
    Создание калибровочной мишени с дополнительной информацией.
    
    Args:
        width: Количество внутренних углов по ширине
        height: Количество внутренних углов по высоте
        square_size: Размер квадрата в пикселях
        margin: Отступ от краев в пикселях
        add_info: Добавить информацию о размерах
        filename: Путь для сохранения файла (опционально)
        
    Returns:
        np.ndarray: Изображение калибровочной мишени
    """
    
    # Создание базовой шахматной доски
    img = create_chessboard(width, height, square_size, margin)
    
    # Конвертация в цветное изображение
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if add_info:
        # Добавление информации
        info_text = [
            f"Chessboard Pattern: {width}x{height}",
            f"Square Size: {square_size}px",
            f"Total Size: {img.shape[1]}x{img.shape[0]}px",
            "Use for camera calibration"
        ]
        
        # Позиция для текста
        text_x = margin
        text_y = img.shape[0] - margin - 20 * len(info_text)
        
        for i, text in enumerate(info_text):
            y_pos = text_y + i * 20
            cv2.putText(img_color, text, (text_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Сохранение файла
    if filename:
        cv2.imwrite(filename, img_color)
        print(f"Saved: {filename}")
    
    return img_color


def create_multiple_patterns(pattern_sizes: list,
                           output_dir: str = "chessboards",
                           square_size: int = 30) -> None:
    """
    Создание нескольких шахматных досок разных размеров.
    
    Args:
        pattern_sizes: Список размеров паттернов [(width, height), ...]
        output_dir: Директория для сохранения
        square_size: Размер квадрата в пикселях
    """
    
    # Создание директории
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for width, height in pattern_sizes:
        filename = Path(output_dir) / f"chessboard_{width}x{height}.png"
        create_chessboard(width, height, square_size, filename=str(filename))
        
        # Создание версии с информацией
        info_filename = Path(output_dir) / f"chessboard_{width}x{height}_info.png"
        create_calibration_target(width, height, square_size, filename=str(info_filename))


def validate_chessboard_image(image_path: str, 
                            pattern_size: Tuple[int, int]) -> bool:
    """
    Проверка корректности шахматной доски на изображении.
    
    Args:
        image_path: Путь к изображению
        pattern_size: Ожидаемый размер паттерна
        
    Returns:
        bool: True если шахматная доска корректна
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Поиск углов
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        print(f"✓ Chessboard {pattern_size} found in {image_path}")
        return True
    else:
        print(f"✓ Chessboard {pattern_size} not found in {image_path}")
        return False


def print_chessboard_info(width: int, height: int, square_size_mm: float = 25.0):
    """
    Вывод информации о шахматной доске.
    
    Args:
        width: Количество внутренних углов по ширине
        height: Количество внутренних углов по высоте
        square_size_mm: Размер квадрата в миллиметрах
    """
    
    # Расчет размеров
    total_width = (width + 1) * square_size_mm
    total_height = (height + 1) * square_size_mm
    num_corners = width * height
    
    print("=== Information about the chessboard ===")
    print(f"Pattern size: {width}x{height} inner corners")
    print(f"Number of corners: {num_corners}")
    print(f"Square size: {square_size_mm} mm")
    print(f"Total size: {total_width:.1f} x {total_height:.1f} mm")
    print(f" Area: {total_width * total_height:.1f} mm2")
        
    # Рекомендации
    print("\n=== Recommendations===")
    print(f"Minimum shooting distance: {total_width*2:.0f} mm")
    print(f"Recommended distance: {total_width*3:.0f} mm")
    print(f"Minimum number of images: 10")
    print(f"Recommended number of images: 15-20")