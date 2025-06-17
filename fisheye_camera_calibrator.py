import cv2
import numpy as np
import os
import glob
import logging
from typing import Tuple, List, Optional


class FisheyeCalibrator:
    def __init__(self, logger: logging.Logger, pattern_size=(9, 6)):
        """
        Инициализация калибратора fisheye камеры

        Args:
            pattern_size: размер шахматной доски (внутренние углы) - (ширина, высота)
        """
        self.logger = logger
        self.pattern_size = pattern_size
        self.square_size = 1.0  # размер квадрата в произвольных единицах

        # Массивы для хранения точек объекта и изображения
        self.object_points = []  # 3D точки в реальном мире
        self.image_points = []  # 2D точки на изображении

        # Подготовка объектных точек (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((1, pattern_size[1] * pattern_size[0], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp[0] *= self.square_size

        # Параметры калибровки fisheye (будут заполнены после калибровки)
        self.camera_matrix = None  # K - внутренние параметры
        self.dist_coeffs = None  # D - коэффициенты дисторсии fisheye [k1, k2, k3, k4]
        self.rvecs = None  # векторы поворота
        self.tvecs = None  # векторы перемещения
        self.image_size = None

        # Флаги калибровки fisheye
        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                                 cv2.fisheye.CALIB_CHECK_COND + \
                                 cv2.fisheye.CALIB_FIX_SKEW

    def find_chessboard_corners(self, image_path):
        """
        Поиск углов шахматной доски на изображении fisheye камеры

        Args:
            image_path: путь к изображению

        Returns:
            bool: True если углы найдены, False иначе
        """
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't upload image: {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]

        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Уточнение координат углов для fisheye
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

            # Сохранение найденных точек
            self.object_points.append(self.objp)
            self.image_points.append(corners2)

            self.logger.info(f"✓ Corners found in image: {os.path.basename(image_path)}")
            return True
        else:
            self.logger.error(f"✗ Corners not found in image: {os.path.basename(image_path)}")
            return False

    def calibrate_from_images(self, images_path):
        """
        Калибровка fisheye камеры по набору изображений

        Args:
            images_path: путь к папке с изображениями или паттерн для glob
        """
        # Получение списка изображений
        if os.path.isdir(images_path):
            image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                          glob.glob(os.path.join(images_path, "*.png")) + \
                          glob.glob(os.path.join(images_path, "*.jpeg"))
        else:
            image_files = glob.glob(images_path)

        if not image_files:
            self.logger.error("Images not found!")
            return False

        self.logger.info(f"Found {len(image_files)} images")

        # Поиск углов на каждом изображении
        successful_detections = 0
        for img_path in image_files:
            if self.find_chessboard_corners(img_path):
                successful_detections += 1

        self.logger.info(f"\nSuccessfully processed {successful_detections} of {len(image_files)} images")

        if successful_detections < 15:
            self.logger.warning("Warning: for high-quality fisheye calibration, it is recommended to use at least 15 images")

        if successful_detections == 0:
            self.logger.error("Error: no images with a chessboard found")
            return False

        # Подготовка данных для калибровки fisheye
        N_OK = len(self.object_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        # Выполнение калибровки fisheye
        self.logger.info("\nPerforming fisheye calibration...")
        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                self.object_points,
                self.image_points,
                self.image_size,
                K,
                D,
                rvecs,
                tvecs,
                self.calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

            self.camera_matrix = K
            self.dist_coeffs = D
            self.rvecs = rvecs
            self.tvecs = tvecs

            self.logger.info("✓ Fisheye calibration completed successfully!")
            self.logger.info(f"RMS error: {rms:.4f}")
            self.print_calibration_results()
            return True

        except Exception as e:
            self.logger.error(f"Error during fisheye calibration: {e}")
            return False

    def print_calibration_results(self):
        """Вывод результатов калибровки fisheye"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FISHEYE CAMERA CALIBRATION RESULTS")
        self.logger.info("=" * 60)

        self.logger.info("\nCamera Matrix K):")
        self.logger.info(self.camera_matrix)

        self.logger.info(f"\nInternal parameters:")
        self.logger.info(f"fx = {self.camera_matrix[0, 0]:.2f}")
        self.logger.info(f"fy = {self.camera_matrix[1, 1]:.2f}")
        self.logger.info(f"cx = {self.camera_matrix[0, 2]:.2f}")
        self.logger.info(f"cy = {self.camera_matrix[1, 2]:.2f}")

        self.logger.info(f"\nFisheye distortion coefficients (D):")
        self.logger.info(f"k1 = {self.dist_coeffs[0][0]:.6f}")
        self.logger.info(f"k2 = {self.dist_coeffs[1][0]:.6f}")
        self.logger.info(f"k3 = {self.dist_coeffs[2][0]:.6f}")
        self.logger.info(f"k4 = {self.dist_coeffs[3][0]:.6f}")

        # Анализ типа fisheye
        k1 = self.dist_coeffs[0][0]
        if k1 < -0.1:
            fisheye_type = "Strong fisheye effect"
        elif k1 < 0:
            fisheye_type = "Moderate fisheye effect"
        else:
            fisheye_type = "Weak fisheye effect or normal camera"

        self.logger.info(f"\nCamera type: {fisheye_type}")

        # Поле зрения (FOV)
        fov_x = 2 * np.arctan(self.image_size[0] / (2 * self.camera_matrix[0, 0])) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_size[1] / (2 * self.camera_matrix[1, 1])) * 180 / np.pi
        self.logger.info(f"\nField of view (FOV):")
        self.logger.info(f"Horizontal: {fov_x:.1f}°")
        self.logger.info(f"Vertical: {fov_y:.1f}°")

        # Оценка качества калибровки
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.fisheye.projectPoints(
                self.object_points[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.object_points)
        self.logger.info(f"\nAverage reprojection error: {mean_error:.3f} pixels")

        if mean_error < 0.5:
            self.logger.info("Excellent calibration quality!")
        elif mean_error < 1.0:
            self.logger.info("Good calibration quality")
        elif mean_error < 2.0:
            self.logger.info("Acceptable calibration quality")
        else:
            self.logger.info("Low quality - need more high-quality images")

    def save_calibration(self, filename="fisheye_calibration.npz"):
        """Сохранение параметров калибровки fisheye"""
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return False

        np.savez(filename,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 image_size=self.image_size)
        self.logger.info(f"Fisheye calibration parameters saved to {filename}")
        return True

    def load_calibration(self, filename="fisheye_calibration.npz"):
        """Загрузка параметров калибровки fisheye"""
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size'])
            self.logger.info(f"Fisheye calibration parameters loaded from {filename}")
            return True
        except:
            self.logger.error(f"Error loading {filename}")
            return False

    def undistort_fisheye(self, image_path, output_path=None, method='equirectangular'):
        """
        Исправление дисторсии fisheye изображения

        Args:
            image_path: путь к исходному изображению
            output_path: путь для сохранения исправленного изображения
            method: метод исправления ('equirectangular', 'perspective', 'cylindrical', 'stereographic')
        """
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration or load parameters!")
            return None

        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't load image: {image_path}")
            return None

        # Различные методы исправления fisheye
        if method == 'equirectangular':
            # Equirectangular projection - хорошо для панорам
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3),
                self.camera_matrix, self.image_size, cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        elif method == 'perspective':
            # Perspective projection - как обычная камера
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix, self.dist_coeffs, self.image_size, np.eye(3), balance=0.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3),
                new_K, self.image_size, cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        elif method == 'cylindrical':
            # Cylindrical projection - хорошо для широких сцен
            undistorted = self._cylindrical_projection(img)

        elif method == 'stereographic':
            # Stereographic projection - сохраняет углы
            undistorted = self._stereographic_projection(img)

        else:
            self.logger.error(f"Unknown method: {method}")
            return None

        if output_path:
            cv2.imwrite(output_path, undistorted)
            self.logger.info(f"Undistorted fisheye image ({method}) saved: {output_path}")

        return undistorted

    def _cylindrical_projection(self, img):
        """Цилиндрическая проекция fisheye изображения"""
        h, w = img.shape[:2]

        # Создание карты координат для цилиндрической проекции
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]

        for y in range(h):
            for x in range(w):
                # Нормализованные координаты
                theta = (x - cx) / fx
                phi = (y - cy) / fy

                # Цилиндрическая проекция
                x_cyl = fx * np.tan(theta) + cx
                y_cyl = fy * phi / np.cos(theta) + cy

                if 0 <= x_cyl < w and 0 <= y_cyl < h:
                    map_x[y, x] = x_cyl
                    map_y[y, x] = y_cyl

        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def _stereographic_projection(self, img):
        """Стереографическая проекция fisheye изображения"""
        # Упрощенная реализация
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, np.eye(3),
            self.camera_matrix * 0.7, self.image_size, cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def show_undistortion_comparison(self, image_path):
        """Показать сравнение различных методов исправления fisheye"""
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return

        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't load image: {image_path}")
            return

        # Создание исправленных версий
        methods = ['equirectangular', 'perspective']
        results = [img]  # Оригинал

        for method in methods:
            undistorted = self.undistort_fisheye(image_path, method=method)
            if undistorted is not None:
                results.append(undistorted)

        # Изменение размера для отображения
        display_height = 300
        display_results = []
        for result in results:
            aspect_ratio = result.shape[1] / result.shape[0]
            display_width = int(display_height * aspect_ratio)
            resized = cv2.resize(result, (display_width, display_height))
            display_results.append(resized)

        # Создание сравнительного изображения
        comparison = np.hstack(display_results)

        # Добавление подписей
        labels = ['Original', 'Equirectangular', 'Perspective']
        x_offset = 0
        for i, (result, label) in enumerate(zip(display_results, labels)):
            cv2.putText(comparison, label, (x_offset + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            x_offset += result.shape[1]

        # Показать результат
        cv2.imshow("Fisheye Undistortion Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def estimate_fisheye_fov(self):
        """Оценка максимального поля зрения fisheye камеры"""
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return None

        # Расчет максимального радиуса в изображении
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        max_radius = min(
            np.sqrt(cx ** 2 + cy ** 2),
            np.sqrt((self.image_size[0] - cx) ** 2 + cy ** 2),
            np.sqrt(cx ** 2 + (self.image_size[1] - cy) ** 2),
            np.sqrt((self.image_size[0] - cx) ** 2 + (self.image_size[1] - cy) ** 2)
        )

        # Угол для максимального радиуса
        k1 = self.dist_coeffs[0][0]
        if k1 != 0:
            # Приближенная оценка для fisheye модели
            max_angle = 2 * np.arctan(max_radius / (2 * self.camera_matrix[0, 0]))
            max_fov = max_angle * 180 / np.pi * 2
        else:
            # Стандартная модель камеры
            max_fov = 2 * np.arctan(max_radius / self.camera_matrix[0, 0]) * 180 / np.pi

        self.logger.info(f"Estimated maximum field of view: {max_fov:.1f}°")
        return max_fov


def run_fisheye_calibration(logger: logging.Logger):
    """Пример использования fisheye калибратора"""

    # Создание fisheye калибратора
    calibrator = FisheyeCalibrator(logger,pattern_size=(9, 6))

    # Папка с изображениями fisheye камеры
    images_folder = "fisheye_calibration_images"

    # Выполнение калибровки
    if calibrator.calibrate_from_images(images_folder):
        # Сохранение результатов
        calibrator.save_calibration("my_fisheye_calibration.npz")

        # Оценка поля зрения
        calibrator.estimate_fisheye_fov()

        # Пример исправления дисторсии разными методами
        test_image = "test_fisheye.jpg"
        if os.path.exists(test_image):
            calibrator.undistort_fisheye(test_image, "undistorted_equirectangular.jpg", "equirectangular")
            calibrator.undistort_fisheye(test_image, "undistorted_perspective.jpg", "perspective")

            # Показать сравнение методов
            calibrator.show_undistortion_comparison(test_image)
