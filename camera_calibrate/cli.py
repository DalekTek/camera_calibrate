 """
Командная строка интерфейс для библиотеки калибровки камеры.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .calibrators import CameraCalibrator, FisheyeCalibrator
from .utils import setup_logger, create_chessboard
from .constants import DEFAULT_PATTERN_SIZE, SUPPORTED_IMAGE_FORMATS


def main():
    """Основная функция командной строки интерфейса."""
    
    parser = argparse.ArgumentParser(
        description="Camera Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate regular camera
  camera-calibrate regular --images calibration_images/ --output camera_calibration.npz
  
  # Calibrate fisheye camera
  camera-calibrate fisheye --images fisheye_images/ --output fisheye_calibration.npz
  
  # Create chessboard pattern
  camera-calibrate create-pattern --width 9 --height 6 --output chessboard.png
  
  # Undistort image
  camera-calibrate undistort --calibration camera_calibration.npz --input image.jpg --output undistorted.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Команда калибровки обычной камеры
    regular_parser = subparsers.add_parser('regular', help='Calibrate regular camera')
    regular_parser.add_argument('--images', required=True, help='Path to calibration images')
    regular_parser.add_argument('--output', required=True, help='Output calibration file')
    regular_parser.add_argument('--pattern-size', nargs=2, type=int, 
                               default=DEFAULT_PATTERN_SIZE, help='Pattern size (width height)')
    regular_parser.add_argument('--log-level', default='INFO', 
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               help='Logging level')
    
    # Команда калибровки fisheye камеры
    fisheye_parser = subparsers.add_parser('fisheye', help='Calibrate fisheye camera')
    fisheye_parser.add_argument('--images', required=True, help='Path to calibration images')
    fisheye_parser.add_argument('--output', required=True, help='Output calibration file')
    fisheye_parser.add_argument('--pattern-size', nargs=2, type=int,
                               default=DEFAULT_PATTERN_SIZE, help='Pattern size (width height)')
    fisheye_parser.add_argument('--log-level', default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               help='Logging level')
    
    # Команда создания шахматной доски
    pattern_parser = subparsers.add_parser('create-pattern', help='Create chessboard pattern')
    pattern_parser.add_argument('--width', type=int, default=9, help='Pattern width')
    pattern_parser.add_argument('--height', type=int, default=6, help='Pattern height')
    pattern_parser.add_argument('--square-size', type=int, default=30, help='Square size in pixels')
    pattern_parser.add_argument('--output', required=True, help='Output image file')
    pattern_parser.add_argument('--add-info', action='store_true', help='Add information text')
    
    # Команда исправления дисторсии
    undistort_parser = subparsers.add_parser('undistort', help='Undistort image')
    undistort_parser.add_argument('--calibration', required=True, help='Calibration file')
    undistort_parser.add_argument('--input', required=True, help='Input image')
    undistort_parser.add_argument('--output', required=True, help='Output image')
    undistort_parser.add_argument('--type', choices=['regular', 'fisheye'], 
                                 default='regular', help='Camera type')
    undistort_parser.add_argument('--method', choices=['equirectangular', 'perspective', 'cylindrical'],
                                 default='equirectangular', help='Undistortion method (fisheye only)')
    
    # Команда информации о калибровке
    info_parser = subparsers.add_parser('info', help='Show calibration information')
    info_parser.add_argument('--calibration', required=True, help='Calibration file')
    info_parser.add_argument('--type', choices=['regular', 'fisheye'], 
                            default='regular', help='Camera type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'regular':
            return calibrate_regular(args)
        elif args.command == 'fisheye':
            return calibrate_fisheye(args)
        elif args.command == 'create-pattern':
            return create_pattern(args)
        elif args.command == 'undistort':
            return undistort_image(args)
        elif args.command == 'info':
            return show_calibration_info(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def calibrate_regular(args) -> int:
    """Калибровка обычной камеры."""
    
    # Настройка логгера
    logger = setup_logger('regular_calibration', level=getattr(args, 'log_level', 'INFO'))
    
    # Создание калибратора
    pattern_size = tuple(getattr(args, 'pattern_size', DEFAULT_PATTERN_SIZE))
    calibrator = CameraCalibrator(logger, pattern_size=pattern_size)
    
    logger.info(f"Starting regular camera calibration...")
    logger.info(f"Pattern size: {pattern_size}")
    logger.info(f"Images path: {args.images}")
    
    # Выполнение калибровки
    if calibrator.calibrate_from_images(args.images):
        # Сохранение результатов
        if calibrator.save_calibration(args.output):
            logger.info(f"Calibration saved to: {args.output}")
            return 0
        else:
            logger.error("Failed to save calibration")
            return 1
    else:
        logger.error("Calibration failed")
        return 1


def calibrate_fisheye(args) -> int:
    """Калибровка fisheye камеры."""
    
    # Настройка логгера
    logger = setup_logger('fisheye_calibration', level=getattr(args, 'log_level', 'INFO'))
    
    # Создание калибратора
    pattern_size = tuple(getattr(args, 'pattern_size', DEFAULT_PATTERN_SIZE))
    calibrator = FisheyeCalibrator(logger, pattern_size=pattern_size)
    
    logger.info(f"Starting fisheye camera calibration...")
    logger.info(f"Pattern size: {pattern_size}")
    logger.info(f"Images path: {args.images}")
    
    # Выполнение калибровки
    if calibrator.calibrate_from_images(args.images):
        # Сохранение результатов
        if calibrator.save_calibration(args.output):
            logger.info(f"Calibration saved to: {args.output}")
            
            # Оценка поля зрения
            fov = calibrator.estimate_fisheye_fov()
            if fov:
                logger.info(f"Estimated FOV: {fov:.1f}°")
            
            return 0
        else:
            logger.error("Failed to save calibration")
            return 1
    else:
        logger.error("Calibration failed")
        return 1


def create_pattern(args) -> int:
    """Создание шахматной доски."""
    
    print(f"Creating chessboard pattern {args.width}x{args.height}...")
    
    if getattr(args, 'add_info', False):
        create_calibration_target(
            width=args.width,
            height=args.height,
            square_size=args.square_size,
            filename=args.output
        )
    else:
        create_chessboard(
            width=args.width,
            height=args.height,
            square_size=args.square_size,
            filename=args.output
        )
    
    print(f"Pattern saved to: {args.output}")
    return 0


def undistort_image(args) -> int:
    """Исправление дисторсии изображения."""
    
    # Загрузка калибровки
    if args.type == 'regular':
        calibrator = CameraCalibrator(setup_logger('undistort'))
    else:
        calibrator = FisheyeCalibrator(setup_logger('undistort'))
    
    if not calibrator.load_calibration(args.calibration):
        print(f"Failed to load calibration: {args.calibration}")
        return 1
    
    # Исправление дисторсии
    if args.type == 'regular':
        undistorted = calibrator.undistort_image(args.input, args.output)
    else:
        undistorted = calibrator.undistort_fisheye(args.input, args.output, args.method)
    
    if undistorted is not None:
        print(f"Image undistorted and saved to: {args.output}")
        return 0
    else:
        print("Failed to undistort image")
        return 1


def show_calibration_info(args) -> int:
    """Показать информацию о калибровке."""
    
    # Загрузка калибровки
    if args.type == 'regular':
        calibrator = CameraCalibrator(setup_logger('info'))
    else:
        calibrator = FisheyeCalibrator(setup_logger('info'))
    
    if not calibrator.load_calibration(args.calibration):
        print(f"Failed to load calibration: {args.calibration}")
        return 1
    
    # Получение информации
    info = calibrator.get_calibration_info()
    
    if not info:
        print("No calibration information available")
        return 1
    
    print("=== Calibration Information ===")
    print(f"Camera type: {args.type}")
    print(f"Image size: {info.get('image_size', 'Unknown')}")
    print(f"Pattern size: {info.get('pattern_size', 'Unknown')}")
    print(f"Number of images: {info.get('num_images', 'Unknown')}")
    
    if 'focal_length' in info:
        print(f"Focal length: fx={info['focal_length']['fx']:.2f}, fy={info['focal_length']['fy']:.2f}")
    
    if 'principal_point' in info:
        print(f"Principal point: cx={info['principal_point']['cx']:.2f}, cy={info['principal_point']['cy']:.2f}")
    
    if 'distortion_coefficients' in info:
        print("Distortion coefficients:")
        for key, value in info['distortion_coefficients'].items():
            print(f"  {key}: {value:.6f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())