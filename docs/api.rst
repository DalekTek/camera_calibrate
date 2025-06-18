 API Reference
=============

Основные классы
--------------

CameraCalibrator
~~~~~~~~~~~~~~~

.. autoclass:: camera_calibrate.CameraCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

FisheyeCalibrator
~~~~~~~~~~~~~~~~~

.. autoclass:: camera_calibrate.FisheyeCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Утилиты
-------

Логирование
~~~~~~~~~~~

.. autofunction:: camera_calibrate.setup_logger

Генерация шахматных досок
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: camera_calibrate.create_chessboard

.. autofunction:: camera_calibrate.create_calibration_target

Работа с изображениями
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: camera_calibrate.load_image_safe

.. autofunction:: camera_calibrate.save_image_safe

Константы
---------

.. automodule:: camera_calibrate.constants
   :members:
   :undoc-members:

Командная строка
---------------

.. automodule:: camera_calibrate.cli
   :members:
   :undoc-members: