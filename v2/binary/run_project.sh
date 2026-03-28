#!/bin/bash
echo "================================="
echo "CS4185 Image Retrieval System"
echo "Deep Learning + FAISS"
echo "================================="

echo "Installing compatible packages..."
pip3 install opencv-python==4.6.0.66 numpy==1.24.3 pillow tensorflow faiss-cpu

echo ""
echo "Checking for required files..."
if [ -f "../source/main.py" ]; then
    echo "Found main Python file"
else
    echo "ERROR: main.py not found in source folder!"
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
python3 main.py

echo ""
echo "Program finished."
read -p "Press enter to continue..."