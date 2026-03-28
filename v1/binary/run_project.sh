#!/bin/bash
echo "================================="
echo "CS4185 Image Retrieval System"
echo "================================="

echo "Installing compatible packages..."
pip3 uninstall opencv-python numpy pillow -y
pip3 install opencv-python==4.6.0.66 numpy==1.24.3 pillow

echo ""
echo "Checking for required files..."
if [ -f "../source/CS4185 Image retrieval system.py" ]; then
    echo "Found main Python file"
else
    echo "ERROR: Python file not found in source folder!"
    read -p "Press enter to exit..."
    exit 1
fi

if [ -d "../source/image.orig" ]; then
    echo "Found image database"
else
    echo "ERROR: image.orig folder not found!"
    read -p "Press enter to exit..."
    exit 1
fi

if [ -d "../source/image.query" ]; then
    echo "Found query images"
else
    echo "ERROR: image.query folder not found!"
    read -p "Press enter to exit..."
    exit 1
fi

echo ""
echo "Running Image Retrieval System..."
cd ../source
python3 "CS4185 Image retrieval system.py"

echo ""
echo "Program finished."
read -p "Press enter to continue..."