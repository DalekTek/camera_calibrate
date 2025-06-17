from colorama import init
from logger_config import setup_logger
from camera_calibrator import CameraCalibrator, run_calibrate
from fisheye_camera_calibrator import run_fisheye_calibration

def main():
    """Пример использования калибратора"""
    # Инициализация colorama
    init()

    # Инициализация логгера
    logger = setup_logger('CameraCalibrator')
    logger.info("Starting camera calibration...")
    logger.info("-" * 60)

    run_calibrate(logger)

    logger.info("Starting fisheye camera calibration...")
    logger.info("-" * 60)
    
    run_fisheye_calibration(logger)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()