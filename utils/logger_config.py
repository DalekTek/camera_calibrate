"""
Конфигурация логирования для библиотеки калибровки камеры.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from constants import LOGGING_CONFIG


def setup_logger(name: str = 'camera_calibrator', 
                log_dir: str = 'logs',
                level: int = logging.INFO,
                console_output: bool = True,
                file_output: bool = True) -> logging.Logger:
    """
    Настройка логгера для записи сообщений.
    
    Args:
        name: Имя логгера
        log_dir: Директория для сохранения логов
        level: Уровень логирования
        console_output: Включить вывод в консоль
        file_output: Включить запись в файл
        
    Returns:
        logging.Logger: Настроенный объект логгера
    """
    
    # Создание логгера
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Очистка существующих обработчиков
    logger.handlers.clear()
    
    # Форматтер
    formatter = logging.Formatter(
        LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['date_format']
    )
    
    # Файловый обработчик
    if file_output:
        try:
            # Создание директории для логов
            os.makedirs(log_dir, exist_ok=True)
            
            # Создание имени файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f'{name}_{timestamp}.log'
            
            file_handler = logging.FileHandler(
                log_file, 
                encoding=LOGGING_CONFIG['encoding']
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Консольный обработчик
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'camera_calibrator') -> logging.Logger:
    """
    Получение существующего логгера.
    
    Args:
        name: Имя логгера
        
    Returns:
        logging.Logger: Объект логгера
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Установка уровня логирования.
    
    Args:
        logger: Объект логгера
        level: Новый уровень логирования
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(logger: logging.Logger, 
                    log_file: str, 
                    level: int = logging.INFO) -> None:
    """
    Добавление файлового обработчика к существующему логгеру.
    
    Args:
        logger: Объект логгера
        log_file: Путь к файлу лога
        level: Уровень логирования
    """
    try:
        # Создание директории если нужно
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создание обработчика
        file_handler = logging.FileHandler(
            log_file, 
            encoding=LOGGING_CONFIG['encoding']
        )
        file_handler.setLevel(level)
        
        # Форматтер
        formatter = logging.Formatter(
            LOGGING_CONFIG['format'],
            datefmt=LOGGING_CONFIG['date_format']
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"Warning: Could not add file handler: {e}")


def create_calibration_logger(calibration_name: str, 
                            log_dir: str = 'logs') -> logging.Logger:
    """
    Создание специализированного логгера для калибровки.
    
    Args:
        calibration_name: Имя калибровки
        log_dir: Директория для логов
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"calibration_{calibration_name}_{timestamp}"
    
    return setup_logger(
        name=logger_name,
        log_dir=log_dir,
        level=logging.INFO,
        console_output=True,
        file_output=True
    )