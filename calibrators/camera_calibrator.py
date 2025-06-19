"""
Калибратор для обычных камер.

Предоставляет функциональность для калибровки обычных камер
с исправлением радиальной и тангенциальной дисторсии.
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, List, Optional, Union
from pathlib import Path

from calibrators.base_calibrator import BaseCalibrator
from utils.image_handler import get_image_files
from constants import (
    DEFAULT_PATTERN_SIZE,
    DEFAULT_SQUARE_SIZE,
    CALIBRATION_QUALITY_THRESHOLDS,
    CHESSBOARD_DETECTION_FLAGS,
    CORNER_REFINEMENT_CRITERIA,
    CORNER_REFINEMENT_WINDOW_SIZE,
    ERROR_MESSAGES,
    WARNING_MESSAGES,
    INFO_MESSAGES,
    PATH_TO_CALIBRATE_IMG
)



class CameraCalibrator(BaseCalibrator):
    """
    Калибратор для обычных камер.
    
    Предоставляет методы для калибровки обычных камер с использованием
    шахматной доски и исправления дисторсии изображений.
    
    Attributes:
        pattern_size: Размер шахматной доски (внутренние углы)
        square_size: Размер квадрата в произвольных единицах
        camera_matrix: Матрица камеры (3x3)
        dist_coeffs: Коэффициенты дисторсии
        rvecs: Векторы поворота
        tvecs: Векторы перемещения
        image_size: Размер изображения
        object_points: 3D точки объекта
        image_points: 2D точки изображения
        logger: Объект логгера
    """
    def __init__(self, logger: logging.Logger, pattern_size: Tuple[int, int] = DEFAULT_PATTERN_SIZE):
        """
        Инициализация калибратора камеры.
        
        Args:
            logger: Объект логгера для записи сообщений
            pattern_size: Размер шахматной доски (внутренние углы) - (ширина, высота)
        """
        super().__init__(logger, pattern_size, DEFAULT_SQUARE_SIZE)
        self.objp: np.ndarray = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

    def find_chessboard_corners(self, image_path: Union[str, Path]) -> bool:
        """
        Поиск углов шахматной доски на изображении.

        Args:
            image_path: Путь к изображению

        Returns:
            bool: True если углы найдены, False иначе
        """
        img: Optional[np.ndarray] = cv2.imread(str(image_path))

        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return False

        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]

        # Поиск углов шахматной доски
        flags = (CHESSBOARD_DETECTION_FLAGS['adaptive_thresh'] + 
                CHESSBOARD_DETECTION_FLAGS['fast_check'] + 
                CHESSBOARD_DETECTION_FLAGS['normalize_image'])
        
        ret: bool
        corners: Optional[np.ndarray]
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if ret:
            # Уточнение координат углов
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                       CORNER_REFINEMENT_CRITERIA[0], 
                       CORNER_REFINEMENT_CRITERIA[1])
            
            corners2: np.ndarray = cv2.cornerSubPix(gray, corners, CORNER_REFINEMENT_WINDOW_SIZE, (-1, -1), criteria)

            # Сохранение найденных точек
            self.object_points.append(self.objp)
            self.image_points.append(corners2)

            self.logger.info(f"✓ Corners found in image: {os.path.basename(str(image_path))}")
            return True
        else:
            self.logger.warning(f"✗ No corners found in image: {os.path.basename(str(image_path))}")
            return False

    def calibrate_from_images(self, images_path: Union[str, Path]) -> bool:
        """
        Калибровка камеры по набору изображений.

        Args:
            images_path: Путь к папке с изображениями или паттерн для glob

        Returns:
            bool: True если калибровка успешна, False иначе
        """
        # Получение списка изображений
        image_files: List[str] = get_image_files(images_path)

        if not image_files:
            self.logger.error(ERROR_MESSAGES['no_images_found'])
            return False

        self.logger.info(f"Found {len(image_files)} images")

        # Поиск углов на каждом изображении
        successful_detections = 0
        for img_path in image_files:
            if self.find_chessboard_corners(img_path):
                successful_detections += 1

        self.logger.info(f"\nSuccessfully processed {successful_detections} out of {len(image_files)} images")

        if successful_detections < 10:
            self.logger.warning(WARNING_MESSAGES['few_images'])

        if successful_detections == 0:
            self.logger.error(ERROR_MESSAGES['calibration_failed'])
            return False

        # Выполнение калибровки
        self.logger.info("\nPerforming calibration...")
        ret: bool
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, None, None
        )

        if ret:
            self.logger.info(INFO_MESSAGES['calibration_success'])
            self.print_calibration_results()
            return True
        else:
            self.logger.error(ERROR_MESSAGES['calibration_failed'])
            return False

    def print_calibration_results(self) -> None:
        """Вывод результатов калибровки."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("CALIBRATION RESULTS")
        self.logger.info("=" * 50)

        self.logger.info("\nCamera Matrix:")
        matrix_str = np.array2string(self.camera_matrix, precision=8, separator=' ', prefix='    ')
        for line in matrix_str.split('\n'):
            self.logger.info(line)

        self.logger.info(f"\nFocal Length:")
        self.logger.info(f"fx = {self.camera_matrix[0, 0]:.2f}")
        self.logger.info(f"fy = {self.camera_matrix[1, 1]:.2f}")

        self.logger.info(f"\nPrincipal Point:")
        self.logger.info(f"cx = {self.camera_matrix[0, 2]:.2f}")
        self.logger.info(f"cy = {self.camera_matrix[1, 2]:.2f}")

        self.logger.info(f"\nDistortion Coefficients:")
        self.logger.info(f"k1 = {self.dist_coeffs[0][0]:.6f}")
        self.logger.info(f"k2 = {self.dist_coeffs[0][1]:.6f}")
        self.logger.info(f"p1 = {self.dist_coeffs[0][2]:.6f}")
        self.logger.info(f"p2 = {self.dist_coeffs[0][3]:.6f}")
        self.logger.info(f"k3 = {self.dist_coeffs[0][4]:.6f}")

        # Оценка качества калибровки
        mean_error = self._calculate_reprojection_error()
        self.logger.info(f"\nMean reprojection error: {mean_error:.3f} pixels")

        # Оценка качества
        quality = self._assess_calibration_quality(mean_error)
        self.logger.info(f"Calibration quality: {quality}")

    def _calculate_reprojection_error(self) -> float:
        """
        Расчет ошибки репроекции.
        
        Returns:
            float: Средняя ошибка репроекции в пикселях
        """
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        return total_error / len(self.object_points)

    def _assess_calibration_quality(self, mean_error: float) -> str:
        """
        Оценка качества калибровки.
        
        Args:
            mean_error: Средняя ошибка репроекции
            
        Returns:
            str: Оценка качества
        """
        if mean_error < CALIBRATION_QUALITY_THRESHOLDS['excellent']:
            return "Excellent"
        elif mean_error < CALIBRATION_QUALITY_THRESHOLDS['good']:
            return "Good"
        elif mean_error < CALIBRATION_QUALITY_THRESHOLDS['acceptable']:
            return "Acceptable"
        else:
            return "Poor"

    def save_calibration(self, filename: Union[str, Path]) -> bool:
        """
        Сохранение параметров калибровки.
        
        Args:
            filename: Путь к файлу для сохранения
            
        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if self.camera_matrix is None:
            self.logger.error("First perform calibration!")
            return False

        try:
            np.savez(filename,
                     camera_matrix=self.camera_matrix,
                     dist_coeffs=self.dist_coeffs,
                     image_size=self.image_size)
            self.logger.info(f"Calibration parameters saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filename: Union[str, Path]) -> bool:
        """
        Загрузка параметров калибровки.
        
        Args:
            filename: Путь к файлу с параметрами калибровки
            
        Returns:
            bool: True если загрузка успешна, False иначе
        """
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size'])
            self.logger.info(f"Calibration parameters loaded from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return False

    def undistort_image(self, image_path: Union[str, Path], 
                       output_dir: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Исправление дисторсии изображения.

        Args:
            image_path: Путь к исходному изображению
            output_dir: путь к выходной папке c исправленными изображениями

        Returns:
            Optional[np.ndarray]: Исправленное изображение или None в случае ошибки
        """
        if self.camera_matrix is None:
            self.logger.error("First perform calibration or load parameters!")
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return None

        # Исправление дисторсии
        undistorted: np.ndarray = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, None)

        # Генерация имени выходного файла
        name, ext = os.path.splitext(image_path)
        output_filename = f"{name}_undistorted{ext}"
        output_path = os.path.join(output_dir, output_filename)

        if output_path:
            cv2.imwrite(str(output_path), undistorted)
            self.logger.info(f"Corrected image saved: {output_path}")

        return undistorted

    def show_comparison(self, image_path: Union[str, Path], undistorted: np.ndarray) -> None:
        """
        Показать сравнение исходного и исправленного изображения.
        
        Args:
            image_path: Путь к изображению для сравнения
        """
        if self.camera_matrix is None:
            self.logger.error("First perform calibration!")
            return

        img: Optional[np.ndarray] = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return

        # Создание сравнительного изображения
        comparison: np.ndarray = np.hstack((img, undistorted))

        # Добавление текста
        cv2.putText(comparison, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(comparison, "Undistorted", (img.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Показать результат
        cv2.imshow("Calibration Result", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_calibration_info(self) -> dict:
        """
        Получение информации о калибровке.
        
        Returns:
            dict: Словарь с информацией о калибровке
        """
        if self.camera_matrix is None:
            return {}
            
        return {
            'focal_length': {
                'fx': float(self.camera_matrix[0, 0]),
                'fy': float(self.camera_matrix[1, 1])
            },
            'principal_point': {
                'cx': float(self.camera_matrix[0, 2]),
                'cy': float(self.camera_matrix[1, 2])
            },
            'distortion_coefficients': {
                'k1': float(self.dist_coeffs[0][0]),
                'k2': float(self.dist_coeffs[0][1]),
                'p1': float(self.dist_coeffs[0][2]),
                'p2': float(self.dist_coeffs[0][3]),
                'k3': float(self.dist_coeffs[0][4])
            },
            'image_size': self.image_size,
            'pattern_size': self.pattern_size,
            'num_images': len(self.object_points)
        }


def run_calibrate(logger: logging.Logger, images_folder: str = PATH_TO_CALIBRATE_IMG):
    """Пример использования калибратора"""

    # Создание калибратора
    # Для стандартной шахматной доски 10x7 внутренних углов используйте (9,6)
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))

    logger.info("Starting camera calibration...")
    logger.info("-" * 60)

    if images_folder is None:
        images_folder = PATH_TO_CALIBRATE_IMG
    
    # Подготовка директории для сохранения исправленных изображений
    output_dir = os.path.join(images_folder, "undistorted_perspective")
    os.makedirs(output_dir, exist_ok=True)
     
    # Выполнение калибровки
    if calibrator.calibrate_from_images(images_folder):
        # Сохранение результатов
        calibrator.save_calibration("my_camera_calibration.npz")

        # Пример исправления дисторсии
        if calibrator.image_paths and calibrator.camera_matrix is not None:
            for image_path in calibrator.image_paths:
                logger.info(f"Processing image: {image_path}")
                undistorted = calibrator.undistort_image(image_path, output_dir)
        
                # calibrator.show_comparison(image_path, undistorted)

    logger.info("\nDone!")
