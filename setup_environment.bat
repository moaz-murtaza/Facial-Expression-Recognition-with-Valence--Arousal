@echo off
REM Environment Setup Script for Facial Expression Recognition Assignment
REM Run this script before executing the main Python file

echo ========================================
echo Setting up Deep Learning Environment
echo ========================================

REM Create virtual environment
echo Creating virtual environment 'dl_assignment_venv'...
python -m venv dl_assignment_venv

REM Activate virtual environment
echo Activating virtual environment...
call dl_assignment_venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Pytorch (GPU version if available)
echo Installing TensorFlow...
pip install torch torchvision torchaudio

REM Install core packages
echo Installing core packages...
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python scipy

REM Install additional ML packages
echo Installing additional packages...
pip install timm krippendorff

REM Install optional packages for better performance
echo Installing optional performance packages...
pip install pillow-simd

REM Verify installation
echo Verifying installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, cv2, timm, krippendorff; print('All packages imported successfully!')"

echo ========================================
echo Environment setup complete!
echo ========================================
echo.
echo To activate the environment in future sessions, run:
echo dl_assignment_venv\Scripts\activate.bat
echo.
echo To run the facial expression recognition script:
echo python facial_expression_recognition.py
echo.
echo Make sure to download the dataset from the Google Drive link
echo and extract it to a 'Dataset' folder in this directory.
echo.
pause

