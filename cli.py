import argparse
from utils.logger_config import setup_logger
from calibrators.camera_calibrator import run_calibrate
from calibrators.fisheye_calibrator import run_fisheye_calibration


def main():
    parser = argparse.ArgumentParser(
        description="CLI for calibrating a regular or fisheye camera"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Type of calibration")

    # Обычная камера
    parser_regular = subparsers.add_parser("regular", help="Calibration of a regular camera")
    parser_regular.add_argument("--images", type=str, required=True, help="Path to the folder with images for calibration")
    parser_regular.add_argument("--output", type=str, default="camera_calibration.npz", help="File to save calibration parameters")

    # Fisheye камера
    parser_fisheye = subparsers.add_parser("fisheye", help="Calibration of a fisheye camera")
    parser_fisheye.add_argument("--images", type=str, required=True, help="Path to the folder with images for calibration")
    parser_fisheye.add_argument("--output", type=str, default="fisheye_calibration.npz", help="Файл для сохранения параметров калибровки")

    args = parser.parse_args()
    logger = setup_logger('CameraCalibrateCLI')

    if args.mode == "regular":
        logger.info("=== Calibrating a regular camera ===")
        calibrator = run_calibrate(logger, images_folder=args.images)
        if calibrator:
            calibrator.save_calibration(args.output)
            logger.info(f"The calibration parameters are saved in {args.output}")
    elif args.mode == "fisheye":
        logger.info("=== Calibrating a fisheye camera ===")
        calibrator = run_fisheye_calibration(logger, images_folder=args.images)
        if calibrator:
            calibrator.save_calibration(args.output)
            logger.info(f"The calibration parameters are saved in {args.output}")
    else:
        logger.error("Unknown calibration mode")

if __name__ == "__main__":
    main() 