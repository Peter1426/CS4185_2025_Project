import cv2 as cv
import numpy as np
import os
import time
import pickle
import faiss
from retrieval.base_retrieval import BaseRetrievalSystem

class DeepRetrievalSystem(BaseRetrievalSystem):
    
    def __init__(self, database_dir="image.orig", use_gpu=False):
        super().__init__(database_dir)
        self.index = None
        self.image_paths = []
        self.index_dir = "faiss_index"
        self.use_gpu = use_gpu
        self.extractor = None
        self._init_extractor()
    
    def _init_extractor(self):
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self.MobileNetV2 = MobileNetV2
            self.preprocess_input = preprocess_input
        except ImportError:
            print("Error: TensorFlow not installed!")
            print("Please run: pip install tensorflow")
            raise
    
    def get_system_name(self):
        return "Deep Learning (MobileNetV2 + FAISS)"
    
    def _resize_if_large(self, img, max_size=1000):
        if img is None: return None
        h, w = img.shape[:2]
        if w > max_size or h > max_size:
            return cv.resize(img, (int(w*0.5), int(h*0.5)))
        return img
    
    def _get_model(self):
        if self.extractor is None:
            print("Loading MobileNetV2 model...")
            self.extractor = self.MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
        return self.extractor
    
    # Extract features from OpenCV image
    def _extract_features(self, img):
        model = self._get_model()
        
        # Convert BGR to RGB and resize
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_resized = cv.resize(img_rgb, (224, 224))
        
        # Preprocess and predict
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = self.preprocess_input(x)
        features = model.predict(x, verbose=0)
        
        return features.flatten().astype('float32')
    
    # Build or load FAISS index
    def build_index(self, rebuild=False):
        index_path = os.path.join(self.index_dir, 'faiss_index.bin')
        paths_path = os.path.join(self.index_dir, 'image_paths.pkl')
        
        if not rebuild and os.path.exists(index_path) and os.path.exists(paths_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
            with open(paths_path, 'rb') as f:
                self.image_paths = pickle.load(f)
            
            if self.use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            
            print(f"Loaded index with {len(self.image_paths)} vectors")
            return
        
        print("Building new FAISS index (this may take 2-5 minutes)...")
        features_list = []
        valid_paths = []
        
        for i, img_path in enumerate(self.database_paths):
            img = cv.imread(img_path)
            if img is not None:
                features = self._extract_features(img)
                if features is not None:
                    features_list.append(features)
                    valid_paths.append(img_path)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.database_paths)} images")
        
        features_array = np.array(features_list).astype('float32')
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(1280)
        self.index.add(features_array)
        self.image_paths = valid_paths
        
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        # Save index
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(paths_path, 'wb') as f:
            pickle.dump(self.image_paths, f)
        
        print(f"Index built with {len(valid_paths)} vectors")
    
    # Find best matching image
    def find_best_match(self, query_img):
        if self.index is None:
            self.build_index()
        
        query_features = self._extract_features(query_img)
        query = query_features.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query, 1)
        
        if indices[0][0] < len(self.image_paths):
            best_path = self.image_paths[indices[0][0]]
            best_img = cv.imread(best_path)
            best_score = distances[0][0]
            return best_path, best_img, best_score
        
        return None, None, float('inf')
    
    # Retrieve images with similarity above threshold
    def retrieve_similar(self, query_img, threshold, k=100):
        if self.index is None:
            self.build_index()
        
        query_features = self._extract_features(query_img)
        query = query_features.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query, k)
        
        # Convert distances to similarity scores
        if len(distances[0]) > 0:
            max_dist = max(distances[0])
            min_dist = min(distances[0])
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.image_paths):
                    if max_dist > min_dist:
                        similarity = 1 - (distances[0][i] - min_dist) / (max_dist - min_dist)
                    else:
                        similarity = 1.0
                    
                    if similarity >= threshold:
                        results.append((similarity, self.image_paths[idx]))
            
            return sorted(results, reverse=True, key=lambda x: x[0])
        
        return []