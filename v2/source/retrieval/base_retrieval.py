from abc import ABC, abstractmethod
import cv2 as cv
import os
from glob import glob

# Abstract base class for all retrieval systems
class BaseRetrievalSystem(ABC):
    
    def __init__(self, database_dir="image.orig"):
        self.database_dir = database_dir
        self.database_paths = sorted(glob(database_dir + "/*.jpg"))
    
    @abstractmethod
    def find_best_match(self, query_img):
        """Find the best matching image for a query"""
        pass
    
    @abstractmethod
    def retrieve_similar(self, query_img, threshold):
        """Retrieve all images with similarity above threshold"""
        pass
    
    @abstractmethod
    def get_system_name(self):
        """Return system name for display"""
        pass
    
    def get_category_from_filename(self, filepath):
        """Extract category from filename"""
        filename = os.path.basename(filepath)
        try:
            img_number = int(filename.replace('.jpg', ''))
            if 0 <= img_number <= 99: return 'african'
            elif 100 <= img_number <= 199: return 'beach'
            elif 200 <= img_number <= 299: return 'building'
            elif 300 <= img_number <= 399: return 'bus'
            elif 400 <= img_number <= 499: return 'dinosaur'
            elif 500 <= img_number <= 599: return 'elephant'
            elif 600 <= img_number <= 699: return 'flower'
            elif 700 <= img_number <= 799: return 'horse'
            elif 800 <= img_number <= 899: return 'mountain'
            elif 900 <= img_number <= 999: return 'dish'
            else: return 'unknown'
        except: return 'unknown'