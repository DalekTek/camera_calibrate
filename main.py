from colorama import init
from logger_config import setup_logger
from camera_calibrator import run_calibrate
from fisheye_camera_calibrator import run_fisheye_calibration

def main():
    """Проверка работы калибровки камеры и fisheye калибровки"""
    init()    # Инициализация colorama
    logger = setup_logger('CameraCalibrator')    # Инициализация логгера

    # run_calibrate(logger, images_folder=r"D:\PyCharPRJ\camera_calibrate\fisheye_calibration_images\frames")
    run_fisheye_calibration(logger, images_folder=r"D:\PyCharPRJ\camera_calibrate\fisheye_calibration_images\frames")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()