import logging
from typing import Tuple, List, Optional, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseCalibrator(ABC):
    """
    Абстрактный базовый класс для калибраторов камер.
    Определяет общий интерфейс и хранит общие поля.
    """
    def __init__(self, logger: logging.Logger, pattern_size: Tuple[int, int], square_size: float = 1.0):
        self.logger: logging.Logger = logger
        self.pattern_size: Tuple[int, int] = pattern_size
        self.square_size: float = square_size
        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.image_size: Optional[Tuple[int, int]] = None
        self.image_paths: List[str] = []

    @abstractmethod
    def find_chessboard_corners(self, image_path: str) -> bool:
        """Поиск углов шахматной доски на изображении."""
        pass

    @abstractmethod
    def calibrate_from_images(self, images_path: str) -> bool:
        """Калибровка камеры по набору изображений."""
        pass

    @abstractmethod
    def print_calibration_results(self) -> None:
        """Вывод результатов калибровки."""
        pass

    @abstractmethod
    def save_calibration(self, filename: str) -> bool:
        """Сохранение параметров калибровки."""
        pass

    @abstractmethod
    def load_calibration(self, filename: str) -> bool:
        """Загрузка параметров калибровки."""
        pass
