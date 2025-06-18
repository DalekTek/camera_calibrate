import cv2
import numpy as np
import os
import glob
import logging
from typing import Tuple, List, Optional, Dict


class FisheyeCalibrator:
    def __init__(self, logger: logging.Logger, pattern_size: Tuple[int, int] = (9, 6)):
        """
        Инициализация калибратора fisheye камеры

        Args:
            logger: объект логгера для записи сообщений
            pattern_size: размер шахматной доски (внутренние углы) - (ширина, высота)
        """
        self.logger = logger
        self.image_paths = []
        self.pattern_size = pattern_size
        self.square_size = 1.0  # размер квадрата в произвольных единицах

        # Массивы для хранения точек объекта и изображения
        self.object_points: List[np.ndarray] = []  # 3D точки в реальном мире
        self.image_points: List[np.ndarray] = []  # 2D точки на изображении

        # Подготовка объектных точек (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp: np.ndarray = np.zeros((1, pattern_size[1] * pattern_size[0], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp[0] *= self.square_size

        # Параметры калибровки fisheye (будут заполнены после калибровки)
        self.camera_matrix: Optional[np.ndarray] = None  # K - внутренние параметры
        self.dist_coeffs: Optional[np.ndarray] = None  # D - коэффициенты дисторсии fisheye [k1, k2, k3, k4]
        self.rvecs: Optional[List[np.ndarray]] = None  # векторы поворота
        self.tvecs: Optional[List[np.ndarray]] = None  # векторы перемещения
        self.image_size: Optional[Tuple[int, int]] = None

        # Флаги калибровки fisheye
        self.calibration_flags: int = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                                     cv2.fisheye.CALIB_CHECK_COND + \
                                     cv2.fisheye.CALIB_FIX_SKEW

    def find_chessboard_corners(self, image_path: str) -> bool:
        """
        Поиск углов шахматной доски на изображении fisheye камеры

        Args:
            image_path: путь к изображению

        Returns:
            bool: True если углы найдены, False иначе
        """
        img: Optional[np.ndarray] = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't upload image: {image_path}")
            return False

        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size: Tuple[int, int] = gray.shape[::-1]

        # Поиск углов шахматной доски
        ret: bool
        corners: Optional[np.ndarray]
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Уточнение координат углов для fisheye
            criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners2: np.ndarray = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

            # Сохранение найденных точек
            self.object_points.append(self.objp)
            self.image_points.append(corners2)

            self.logger.info(f"✓ Corners found in image: {os.path.basename(image_path)}")
            return True
        else:
            self.logger.error(f"✗ Corners not found in image: {os.path.basename(image_path)}")
            return False

    def calibrate_from_images(self, images_path: str) -> bool:
        """
        Калибровка fisheye камеры по набору изображений

        Args:
            images_path: путь к папке с изображениями или паттерн для glob

        Returns:
            bool: True если калибровка успешна, False иначе
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
            self.logger.error("Images not found!")
            return False

        self.logger.info(f"Found {len(image_files)} images")

        # Поиск углов на каждом изображении
        successful_detections: int = 0
        for img_path in image_files:
            if self.find_chessboard_corners(img_path):
                successful_detections += 1
                self.image_paths.append(img_path)

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
            rms: float
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
        matrix_str = np.array2string(self.camera_matrix, precision=8, separator=' ', prefix='    ')
        for line in matrix_str.split('\n'):
            self.logger.info(line)

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
        fisheye_type: str
        if k1 < -0.1:
            fisheye_type = "Strong fisheye effect"
        elif k1 < 0:
            fisheye_type = "Moderate fisheye effect"
        else:
            fisheye_type = "Weak fisheye effect or normal camera"

        self.logger.info(f"\nCamera type: {fisheye_type}")

        # Поле зрения (FOV)
        fov_x: float = 2 * np.arctan(self.image_size[0] / (2 * self.camera_matrix[0, 0])) * 180 / np.pi
        fov_y: float = 2 * np.arctan(self.image_size[1] / (2 * self.camera_matrix[1, 1])) * 180 / np.pi
        self.logger.info(f"\nField of view (FOV):")
        self.logger.info(f"Horizontal: {fov_x:.1f}°")
        self.logger.info(f"Vertical: {fov_y:.1f}°")

        # Оценка качества калибровки
        total_error: float = 0
        for i in range(len(self.object_points)):
            try:
                imgpoints2: np.ndarray
                imgpoints2, _ = cv2.fisheye.projectPoints(
                    self.object_points[i],
                    self.rvecs[i],
                    self.tvecs[i],
                    self.camera_matrix,
                    self.dist_coeffs
                )

                # show_reprojection_cv2(
                #     image_path=self.image_paths[i],
                #     image_points=self.image_points[i],
                #     projected_points=imgpoints2,
                #     window_name=f"Frame {i} — Reprojection"
                # )

                # Проверка размеров массивов
                if self.image_points[i].shape != imgpoints2.shape:
                    self.logger.warning(f"Mismatch in array sizes in image {i}")
                    continue

                error: float = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error

            except Exception as e:
                self.logger.error(f"Error in calculating the reproduction error for the image {i}: {str(e)}")
                continue

        if len(self.object_points) > 0:
            mean_error: float = total_error / len(self.object_points)
            self.logger.info(f"\nAverage reprojection error: {mean_error:.3f} pixels")

            if mean_error < 0.5:
                self.logger.info("Excellent calibration quality!")
            elif mean_error < 1.0:
                self.logger.info("Good calibration quality")
            elif mean_error < 2.0:
                self.logger.info("Acceptable calibration quality")
            else:
                self.logger.info("Low quality - need more high-quality images")
        else:
            self.logger.warning("Reproduction error could not be calculated")

    def save_calibration(self, filename: str = "fisheye_calibration.npz") -> bool:
        """
        Сохранение параметров калибровки fisheye
        
        Args:
            filename: путь к файлу для сохранения
            
        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return False

        np.savez(filename,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 image_size=self.image_size)
        self.logger.info(f"Fisheye calibration parameters saved to {filename}")
        return True

    def load_calibration(self, filename: str = "fisheye_calibration.npz") -> bool:
        """
        Загрузка параметров калибровки fisheye
        
        Args:
            filename: путь к файлу с параметрами калибровки
            
        Returns:
            bool: True если загрузка успешна, False иначе
        """
        try:
            data: Dict[str, np.ndarray] = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size'])
            self.logger.info(f"Fisheye calibration parameters loaded from {filename}")
            return True
        except:
            self.logger.error(f"Error loading {filename}")
            return False

    def undistort_fisheye(self, image_path: str, output_dir: str, method: str = 'equirectangular') -> Optional[np.ndarray]:
        """
        Исправление дисторсии fisheye изображения

        Args:
            image_path: путь к исходному изображению
            output_dir: путь к выходной папке
            method: метод исправления ('equirectangular', 'perspective', 'cylindrical', 'stereographic')

        Returns:
            Optional[np.ndarray]: исправленное изображение или None в случае ошибки
        """
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration or load parameters!")
            return None

        img: Optional[np.ndarray] = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't load image: {image_path}")
            return None

        # Различные методы исправления fisheye
        undistorted: Optional[np.ndarray] = None
        if method == 'equirectangular':
            # Equirectangular projection - хорошо для панорам
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3),
                self.camera_matrix, self.image_size, cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        elif method == 'perspective':
            # Perspective projection - как обычная камера
            new_K: np.ndarray = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
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

        # Генерация имени выходного файла
        name, ext = os.path.splitext(image_path)
        output_filename = f"{name}_undistorted{ext}"
        output_path = os.path.join(output_dir, output_filename)

        if output_path and undistorted is not None:
            cv2.imwrite(output_path, undistorted)
            self.logger.info(f"Undistorted fisheye image ({method}) saved: {output_path}")

        return undistorted

    def _cylindrical_projection(self, img: np.ndarray) -> np.ndarray:
        """
        Цилиндрическая проекция fisheye изображения
        
        Args:
            img: исходное изображение
            
        Returns:
            np.ndarray: изображение после цилиндрической проекции
        """
        h, w = img.shape[:2]

        # Создание карты координат для цилиндрической проекции
        map_x: np.ndarray = np.zeros((h, w), dtype=np.float32)
        map_y: np.ndarray = np.zeros((h, w), dtype=np.float32)

        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]

        for y in range(h):
            for x in range(w):
                # Нормализованные координаты
                theta = (x - cx) / fx
                phi= (y - cy) / fy

                # Цилиндрическая проекция
                x_cyl = fx * np.tan(theta) + cx
                y_cyl = fy * phi / np.cos(theta) + cy

                if 0 <= x_cyl < w and 0 <= y_cyl < h:
                    map_x[y, x] = x_cyl
                    map_y[y, x] = y_cyl

        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def _stereographic_projection(self, img: np.ndarray) -> np.ndarray:
        """
        Стереографическая проекция fisheye изображения
        
        Args:
            img: исходное изображение
            
        Returns:
            np.ndarray: изображение после стереографической проекции
        """
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, np.eye(3),
            self.camera_matrix * 0.7, self.image_size, cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def show_undistortion_comparison(self, image_path: str, undistorted: np.ndarray):
        """
        Показать сравнение различных методов исправления fisheye
        
        Args:
            image_path: путь к изображению для сравнения
        """
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return

        img: Optional[np.ndarray] = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Couldn't load image: {image_path}")
            return

        # Создание сравнительного изображения
        comparison: np.ndarray = np.hstack((img, undistorted))

        # Добавление текста
        cv2.putText(comparison, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(comparison, "Undistorted", (img.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Показать результат
        cv2.imshow("Fisheye Undistortion Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def estimate_fisheye_fov(self) -> Optional[float]:
        """
        Оценка максимального поля зрения fisheye камеры
        
        Returns:
            Optional[float]: оценка максимального поля зрения в градусах или None в случае ошибки
        """
        if self.camera_matrix is None:
            self.logger.error("First, perform calibration!")
            return None

        # Расчет максимального радиуса в изображении
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        max_radius: float = min(
            np.sqrt(cx ** 2 + cy ** 2),
            np.sqrt((self.image_size[0] - cx) ** 2 + cy ** 2),
            np.sqrt(cx ** 2 + (self.image_size[1] - cy) ** 2),
            np.sqrt((self.image_size[0] - cx) ** 2 + (self.image_size[1] - cy) ** 2)
        )

        # Угол для максимального радиуса
        k1  = self.dist_coeffs[0][0]
        max_fov: float
        if k1 != 0:
            # Приближенная оценка для fisheye модели
            max_angle: float = 2 * np.arctan(max_radius / (2 * self.camera_matrix[0, 0]))
            max_fov = max_angle * 180 / np.pi * 2
        else:
            # Стандартная модель камеры
            max_fov = 2 * np.arctan(max_radius / self.camera_matrix[0, 0]) * 180 / np.pi

        self.logger.info(f"Estimated maximum field of view: {max_fov:.1f}°")
        return max_fov


def show_reprojection_cv2(image_path: str, image_points: np.ndarray, 
                         projected_points: np.ndarray, window_name: str = "Reprojection"):
    """
    Показывает изображение с исходными и перепроецированными точками через cv2.imshow.
    
    Args:
        image_path: путь к изображению
        image_points: массив найденных точек (Nx1x2)
        projected_points: массив перепроецированных точек (1xNx2)
        window_name: имя окна для отображения
    """
    image_points = image_points.reshape(-1, 2)
    projected_points = projected_points.reshape(-1, 2)

    img: Optional[np.ndarray] = cv2.imread(image_path)
    if img is None:
        print(f"[!] Не удалось загрузить изображение: {image_path}")
        return

    if image_points.shape != projected_points.shape:
        print(f"[!] Shape mismatch: {image_points.shape} vs {projected_points.shape}")
        return

    img_draw: np.ndarray = img.copy()

    # Красные — найденные углы
    for pt in image_points:
        x: float
        y: float
        x, y = pt.ravel()
        cv2.circle(img_draw, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Зелёные — перепроецированные точки
    for pt in projected_points:
        x, y = pt.ravel()
        cv2.circle(img_draw, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Показываем изображение
    cv2.imshow(window_name, img_draw)
    print("[i] Нажмите любую клавишу в окне изображения...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def run_fisheye_calibration(logger: logging.Logger, images_folder: str = "fisheye_calibration_images"):
    """
    Пример использования fisheye калибратора
    
    Args:
        logger: объект логгера для записи сообщений
        images_folder: путь к папке с изображениями для калибровки
    """
    # Создание fisheye калибратора
    calibrator = FisheyeCalibrator(logger, pattern_size=(9, 6))

    logger.info("Starting fisheye camera calibration...")
    logger.info("-" * 60)

    # Подготовка директории для сохранения исправленных изображений
    output_dir = os.path.join(images_folder, "undistorted_perspective")
    os.makedirs(output_dir, exist_ok=True)

    # Выполнение калибровки
    if calibrator.calibrate_from_images(images_folder):
        # Сохранение результатов
        calibrator.save_calibration("my_fisheye_calibration.npz")

        # Оценка поля зрения
        calibrator.estimate_fisheye_fov()

        # Пример исправления дисторсии разными методами
        if calibrator.image_paths and calibrator.camera_matrix is not None:
            for image_path in calibrator.image_paths:
                logger.info(f"Processing image: {image_path}")
                # calibrator.undistort_fisheye(image_path, output_dir, "equirectangular")
                undistorted = calibrator.undistort_fisheye(image_path, output_dir, "perspective")

                # Показать исходное и исправленное изображения
                # calibrator.show_undistortion_comparison(image_path, undistorted)