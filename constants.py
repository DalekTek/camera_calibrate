"""
Константы библиотеки калибровки камеры.
"""
import os
from typing import Tuple, List, Dict, Any

# Размеры по умолчанию
DEFAULT_PATTERN_SIZE: Tuple[int, int] = (9, 6)
DEFAULT_SQUARE_SIZE: float = 1.0

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

# Пороги качества калибровки (в пикселях)
CALIBRATION_QUALITY_THRESHOLDS: Dict[str, float] = {
    'excellent': 0.5,
    'good': 1.0,
    'acceptable': 2.0,
    'poor': float('inf')
}

# Методы исправления дисторсии для fisheye
FISHEYE_UNDISTORTION_METHODS: List[str] = [
    'equirectangular',
    'perspective', 
    'cylindrical',
    'stereographic'
]

# Флаги калибровки fisheye
FISHEYE_CALIBRATION_FLAGS: Dict[str, int] = {
    'recompute_extrinsic': 1,
    'check_condition': 2,
    'fix_skew': 4
}

# Параметры поиска углов шахматной доски
CHESSBOARD_DETECTION_FLAGS: Dict[str, int] = {
    'adaptive_thresh': 1,
    'fast_check': 2,
    'normalize_image': 4
}

# Критерии уточнения углов
CORNER_REFINEMENT_CRITERIA: Tuple[int, int, float] = (
    30,  # max_iter
    0.001,  # epsilon
    1  # type (EPS + MAX_ITER)
)

# Размеры окна для уточнения углов
CORNER_REFINEMENT_WINDOW_SIZE: Tuple[int, int] = (11, 11)

# Параметры логирования
LOGGING_CONFIG: Dict[str, Any] = {
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'encoding': 'utf-8'
}

# Сообщения об ошибках
ERROR_MESSAGES: Dict[str, str] = {
    'no_images_found': 'No images found in the specified directory',
    'calibration_failed': 'Camera calibration failed',
    'invalid_pattern_size': 'Invalid pattern size. Must be positive integers',
    'no_corners_found': 'No chessboard corners found in image',
    'insufficient_images': 'Insufficient images for calibration. Need at least 10',
    'load_calibration_failed': 'Failed to load calibration parameters',
    'save_calibration_failed': 'Failed to save calibration parameters',
    'invalid_image_path': 'Invalid image path or file does not exist',
    'unsupported_format': 'Unsupported image format',
    'fisheye_not_supported': 'Fisheye calibration not supported in this OpenCV build'
}

# Предупреждения
WARNING_MESSAGES: Dict[str, str] = {
    'few_images': 'Warning: Using fewer than 15 images may result in poor calibration',
    'high_error': 'Warning: High reprojection error indicates poor calibration quality',
    'unstable_focal': 'Warning: Unstable focal length may indicate calibration issues',
    'off_center_principal': 'Warning: Principal point is far from image center',
    'strong_distortion': 'Warning: Strong distortion detected'
}

# Информационные сообщения
INFO_MESSAGES: Dict[str, str] = {
    'calibration_success': 'Camera calibration completed successfully',
    'calibration_saved': 'Calibration parameters saved successfully',
    'calibration_loaded': 'Calibration parameters loaded successfully',
    'corners_found': 'Chessboard corners found successfully',
    'undistortion_complete': 'Image undistortion completed successfully'
}

# Путь к папке с калибруемыми изображениями (в корне проекта)
PATH_TO_CALIBRATE_IMG = "calibration_images/frames"
# Путь к папке с изображениями для калибровки fisheye (в корне проекта)
PATH_TO_CALIBRATE_FISHEYE_IMG = "fisheye_calibration_images/frames"
