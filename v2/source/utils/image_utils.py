import cv2 as cv
import os

# Resize image if it exceeds max dimensions
def resize_if_large(img, max_size=1000):
    if img is None:
        return None
    height, width = img.shape[:2]
    if width > max_size or height > max_size:
        return cv.resize(img, (int(width*0.5), int(height*0.5)))
    return img

# Extract category from filename based on image number
def get_category_from_filename(filepath):
    filename = os.path.basename(filepath)
    try:
        img_number = int(filename.replace('.jpg', ''))
        
        if 0 <= img_number <= 99:
            return 'african'
        elif 100 <= img_number <= 199:
            return 'beach'
        elif 200 <= img_number <= 299:
            return 'building'
        elif 300 <= img_number <= 399:
            return 'bus'
        elif 400 <= img_number <= 499:
            return 'dinosaur'
        elif 500 <= img_number <= 599:
            return 'elephant'
        elif 600 <= img_number <= 699:
            return 'flower'
        elif 700 <= img_number <= 799:
            return 'horse'
        elif 800 <= img_number <= 899:
            return 'mountain'
        elif 900 <= img_number <= 999:
            return 'dish'
        else:
            return 'unknown'
    except ValueError:
        return 'unknown'

# Map query names to expected categories
def get_query_category_mapping():
    return {
        'beach': 'beach',
        'mountain': 'mountain',
        'food': 'dish',
        'dinosaur': 'dinosaur',
        'flower': 'flower',
        'horse': 'horse',
        'elephant': 'elephant'
    }