import logging
import os
from datetime import datetime
from colorama import init, Fore, Style
import re

# Инициализация colorama
init()

class ColoredFormatter(logging.Formatter):
    """Форматтер для цветного логирования"""
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record: logging.LogRecord) -> str:
        # Применяем цвет к сообщению
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}{Style.RESET_ALL}"
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

class FileFormatter(logging.Formatter):
    """Форматтер для файла логов без цветовых кодов"""
    def format(self, record: logging.LogRecord) -> str:
        # Сохраняем оригинальные значения
        original_msg = record.msg
        original_levelname = record.levelname
        
        # Удаляем все ANSI-коды из сообщения
        if isinstance(record.msg, str):
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', record.msg)
            record.levelname = ansi_escape.sub('', record.levelname)
            
            # Добавляем отступы для строк матрицы
            if '[' in record.msg and ']' in record.msg:
                lines = record.msg.split('\n')
                formatted_lines = []
                for line in lines:
                    if '[' in line and ']' in line:
                        # Добавляем отступ для строк матрицы
                        formatted_lines.append('    ' + line.strip())
                    else:
                        formatted_lines.append(line)
                record.msg = '\n'.join(formatted_lines)
        
        # Форматируем запись
        formatted = super().format(record)
        
        # Восстанавливаем оригинальные значения
        record.msg = original_msg
        record.levelname = original_levelname
    
        return formatted

def setup_logger(name: str) -> logging.Logger:
    # Создание директории для логов, если она не существует
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Генерация имени файла лога с текущей датой и временем
    log_filename = os.path.join(log_dir, f"camera_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Настройка логгера
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Создание обработчика для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Создание обработчика для записи в файл
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Создание форматтеров
    console_formatter = ColoredFormatter('%(levelname)s: %(message)s')
    file_formatter = FileFormatter('%(asctime)s - %(levelname)s - %(message)s')

    # Применение форматтеров к обработчикам
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Добавление обработчиков к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger