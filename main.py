import argparse
from colorama import init
from utils.logger_config import setup_logger
from calibrators.camera_calibrator import run_calibrate
from calibrators.fisheye_calibrator import run_fisheye_calibration

def main():
    """CLI для запуска калибровки обычной или fisheye камеры"""
    init()    # Инициализация colorama
    parser = argparse.ArgumentParser(description="Calibration of a regular or fisheye camera")
    parser.add_argument('--mode', choices=['normal', 'fisheye'], default='normal', help='Calibration type: normal or fisheye')
    parser.add_argument('--images', type=str, required=False, help='The path to the folder with images for calibration')
    args = parser.parse_args()

    logger = setup_logger('CameraCalibrator')    # Инициализация логгера

    if args.mode == 'normal':
        run_calibrate(logger, images_folder=args.images)
    else:
        run_fisheye_calibration(logger, images_folder=args.images)

    logger.info("\nDone!")

if __name__ == "__main__":
    main()