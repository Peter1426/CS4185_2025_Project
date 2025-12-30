CS4185 Image Retrieval System
=============================

QUICK START:
1. Go to /binary folder
2. Double-click "run_project.bat" (Windows) or "run_project.sh" (MacOS)
3. Program will auto-install packages and run

MANUAL SETUP:
1. Install Python 3.9.12
2. Run: pip install -r requirements.txt
3. Go to /source folder
4. Run: python "CS4185 Image retrieval system.py"

REQUIRED PACKAGES:
- OpenCV-python==4.6.0.66
- NumPy
- pillow

FOLDER STRUCTURE:
/source - Python code and images
/binary - Executables

SYSTEM FEATURES:
- 1. Single image retrieval
- 2. Test all 7 queries
- 3. Retrieve similar images with precision/recall *
- 4. Exit

** These features may take a few minutes to compute except feature 4.

* For feature 3, when run this feature once, a folder that contain the retrieval similar images result will be created in source folder. If required running feature 3 again on the same categories of image, please delete the created folder of that categories first. Otherwise error will occur.
