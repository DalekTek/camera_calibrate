import cv2
import numpy as np
import os
import glob
import logging
from typing import Tuple, List, Optional


class CameraCalibrator:
    def __init__(self, logger: logging.Logger, pattern_size: Tuple[int, int] = (9, 6)):
        """
        Инициализация калибратора камеры
        Args:
            pattern_size: размер шахматной доски (внутренние углы) - (ширина, высота)
        """
        self.pattern_size: Tuple[int, int] = pattern_size
        self.square_size = 1.0  # размер квадрата в произвольных единицах

        # Массивы для хранения точек объекта и изображения
        self.object_points: List[np.ndarray] = []  # 3D точки в реальном мире
        self.image_points: List[np.ndarray] = []  # 2D точки на изображении

        # Подготовка объектных точек (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Параметры калибровки (будут заполнены после калибровки)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.image_size: Optional[Tuple[int, int]] = None

        self.logger = logger
        self.image_paths = []

    def find_chessboard_corners(self, image_path: str) -> bool:
        """
        Поиск углов шахматной доски на изображении

        Args:
            image_path: путь к изображению

        Returns:
            bool: True если углы найдены, False иначе
        """
        img: Optional[np.ndarray] = cv2.imread(image_path)

        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return False

        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size: Tuple[int, int] = gray.shape[::-1]

        # Поиск углов шахматной доски
        ret: bool
        corners: Optional[np.ndarray]
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if ret:
            # Уточнение координат углов
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Сохранение найденных точек
            self.object_points.append(self.objp)
            self.image_points.append(corners2)

            self.logger.info(f"✓ Corners found in image: {os.path.basename(image_path)}")
            return True
        else:
            self.logger.warning(f"✗ No corners found in image: {os.path.basename(image_path)}")
            return False

    def calibrate_from_images(self, images_path: str) -> bool:
        """
        Калибровка камеры по набору изображений

        Args:
            images_path: путь к папке с изображениями или паттерн для glob
        """
        # Получение списка изображений
        image_files: List[str]
        if os.path.isdir(images_path):
            image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                          glob.glob(os.path.join(images_path, "*.png")) + \
                          glob.glob(os.path.join(images_path, "*.jpeg"))
        else:
            image_files = glob.glob(images_path)

        if not image_files:
            self.logger.error("No images found!")
            return False

        self.logger.info(f"Found {len(image_files)} images")

        # Поиск углов на каждом изображении
        successful_detections = 0
        for img_path in image_files:
            if self.find_chessboard_corners(img_path):
                successful_detections += 1
                self.image_paths.append(img_path)

        self.logger.info(f"\nSuccessfully processed {successful_detections} out of {len(image_files)} images")

        if successful_detections < 10:
            self.logger.warning("Warning: at least 10 images are recommended for good calibration")

        if successful_detections == 0:
            self.logger.error("Error: no chessboard patterns found in any image")
            return False

        # Выполнение калибровки
        self.logger.info("\nPerforming calibration...")
        ret: bool
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, None, None
        )

        if ret:
            self.logger.info("✓ Calibration completed successfully!")
            self.print_calibration_results()
            return True
        else:
            self.logger.error("✗ Calibration failed")
            return False

    def print_calibration_results(self):
        """Вывод результатов калибровки"""
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
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(self.object_points[i], self.rvecs[i],
                                              self.tvecs[i], self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.object_points)
        self.logger.info(f"\nMean reprojection error: {mean_error:.3f} pixels")

        if mean_error < 0.5:
            self.logger.info("Excellent calibration quality!")
        elif mean_error < 1.0:
            self.logger.info("Good calibration quality")
        else:
            self.logger.warning("Average calibration quality - recommend more images")

    def save_calibration(self, filename: str = "camera_calibration.npz") -> bool:
        """
        Сохранение параметров калибровки
        Args:
            filename: путь к файлу для сохранения
        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if self.camera_matrix is None:
            self.logger.error("First perform calibration!")
            return False

        np.savez(filename,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 image_size=self.image_size)
        self.logger.info(f"Calibration parameters saved to {filename}")
        return True

    def load_calibration(self, filename: str = "camera_calibration.npz") -> bool:
        """
        Загрузка параметров калибровки
        Args:
            filename: путь к файлу с параметрами калибровки
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
        except:
            self.logger.error(f"Error loading {filename}")
            return False

    def undistort_image(self, image_path: str, output_dir: str) -> Optional[np.ndarray]:
        """
        Исправление дисторсии изображения

        Args:
            image_path: путь к исходному изображению
            output_dir: путь к выходной папке
        """
        if self.camera_matrix is None:
            self.logger.error("First perform calibration or load parameters!")
            return None

        img = cv2.imread(image_path)
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
            cv2.imwrite(output_path, undistorted)
            self.logger.info(f"Corrected image saved: {output_path}")

        return undistorted

    def show_comparison(self, image_path: str, undistorted: np.ndarray):
        """Показать сравнение исходного и исправленного изображения"""
        if self.camera_matrix is None:
            self.logger.error("First perform calibration!")
            return

        img: Optional[np.ndarray] = cv2.imread(image_path)
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


def run_calibrate(logger: logging.Logger, images_folder: str = "calibration_images"):
    """Пример использования калибратора"""

    # Создание калибратора
    # Для стандартной шахматной доски 10x7 внутренних углов используйте (9,6)
    calibrator = CameraCalibrator(logger, pattern_size=(9, 6))

    logger.info("Starting camera calibration...")
    logger.info("-" * 60)

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
