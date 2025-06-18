"""
Camera Calibrate - A comprehensive camera calibration library.

This library provides tools for calibrating regular and fisheye cameras
using chessboard patterns, with support for distortion correction and
various projection methods.

Example:
    >>> from camera_calibrate import CameraCalibrator, FisheyeCalibrator
    >>> from camera_calibrate.logger_config import setup_logger
    >>>
    >>> # Setup logger
    >>> logger = setup_logger()
    >>>
    >>> # Calibrate regular camera
    >>> calibrator = CameraCalibrator(logger, pattern_size=(9, 6))
    >>> if calibrator.calibrate_from_images("calibration_images/"):
    >>>     calibrator.save_calibration("camera_calibration.npz")
    >>>
    >>> # Calibrate fisheye camera
    >>> fisheye_calibrator = FisheyeCalibrator(logger, pattern_size=(9, 6))
    >>> if fisheye_calibrator.calibrate_from_images("fisheye_images/"):
    >>>     fisheye_calibrator.save_calibration("fisheye_calibration.npz")
"""

__version__ = "1.0.0"
__author__ = "Nadezhda Shiryaeva"
__email__ = "sns0998@mail.ru"


# Основные классы
from .calibrators.camera_calibrator import CameraCalibrator
from .calibrators.fisheye_calibrator import FisheyeCalibrator

# Утилиты
from .utils.logger_config import setup_logger
from .utils.chessboard_generator import create_chessboard
from .utils.image_utils import load_image_safe, save_image_safe

# Константы
from .constants import (
    DEFAULT_PATTERN_SIZE,
    DEFAULT_SQUARE_SIZE,
    SUPPORTED_IMAGE_FORMATS,
    CALIBRATION_QUALITY_THRESHOLDS,
)

# Все публичные API
__all__ = [
    # Основные классы
    "CameraCalibrator",
    "FisheyeCalibrator",
    
    # Утилиты
    "setup_logger",
    "create_chessboard",
    "load_image_safe",
    "save_image_safe",
    
    # Константы
    "DEFAULT_PATTERN_SIZE",
    "DEFAULT_SQUARE_SIZE",
    "SUPPORTED_IMAGE_FORMATS",
    "CALIBRATION_QUALITY_THRESHOLDS",
]