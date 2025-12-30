@echo off
echo =================================
echo CS4185 Image Retrieval System
echo =================================

echo Installing required packages...
pip install opencv-python==4.6.0.66 numpy==1.24.3 pillow

echo.
echo Checking for required files...
if exist "..\source\CS4185 Image retrieval system.py" (
    echo Found main Python file
) else (
    echo ERROR: "CS4185 Image retrieval system.py" not found in source folder!
    pause
    exit
)

if exist "..\source\image.orig" (
    echo Found image database
) else (
    echo ERROR: image.orig folder not found!
    pause
    exit
)

if exist "..\source\image.query" (
    echo Found query images
) else (
    echo ERROR: image.query folder not found!
    pause
    exit
)

echo.
echo Running Image Retrieval System...
cd ..\source
python "CS4185 Image retrieval system.py"

echo.
echo Program finished.
pause