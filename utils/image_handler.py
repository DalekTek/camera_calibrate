"""
# utils/image_handler.py
Обработчик изображений для получения списка файлов изображений
"""
import glob
from typing import List, Union
from pathlib import Path
from constants import SUPPORTED_IMAGE_FORMATS

def get_image_files(images_path: Union[str, Path]) -> List[str]:
    """
    Получение списка файлов изображений.

    Args:
        images_path: Путь к папке с изображениями
    Returns:
        List[str]: Список путей к файлам изображений
    """
    images_path = Path(images_path)

    if images_path.is_dir():
        image_files = []
        for ext in SUPPORTED_IMAGE_FORMATS:
            image_files.extend(glob.glob(str(images_path / f"*{ext}")))
            image_files.extend(glob.glob(str(images_path / f"*{ext.upper()}")))
        return image_files
    else:
        return glob.glob(str(images_path))