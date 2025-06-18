 from setuptools import setup, find_packages
import os

# Чтение README файла
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Чтение requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="camera-calibrate",
    version="1.0.0",
    author="Nadezhda Shiryaeva",
    author_email="sns0998@mail.ru",
    description="A comprehensive camera calibration library for regular and fisheye cameras",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/camera-calibrate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "examples": [
            "matplotlib>=3.3",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "camera-calibrate=camera_calibrate.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "camera_calibrate": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="camera calibration, computer vision, fisheye, opencv, numpy",
    project_urls={
        "Bug Reports": "https://github.com/DalekTek/camera-calibrate/issues",
        "Source": "https://github.com/DalekTek/camera-calibrate",
        "Documentation": "https://camera-calibrate.readthedocs.io/",
    },
)