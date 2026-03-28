import cv2 as cv
import numpy as np
from glob import glob
import os
from retrieval.base_retrieval import BaseRetrievalSystem

class TraditionalRetrievalSystem(BaseRetrievalSystem):
    
    def __init__(self, database_dir="image.orig"):
        super().__init__(database_dir)
        self.database_imgs = None
    
    def get_system_name(self):
        return "Traditional (Edge + Color + CLD + Segmentation)"
        
    def resize_if_large(self, img):
        if img is None:
            return None
        height, width = img.shape[:2]
        if width > 1000 or height > 1000:
            return cv.resize(img, (int(width*0.5), int(height*0.5)))
        return img
    
    def compareImgs_edge(self, img1, img2):
        # Resize img2 to img1
        width, height = img1.shape[1], img1.shape[0]
        img2 = cv.resize(img2, (width, height))

        # Convert to grayscale
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Multi-scale processing
        scales = [1.0, 0.5]
        all_scores = []
        
        for scale in scales:
            if scale != 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                img1_scaled = cv.resize(img1_gray, (new_width, new_height))
                img2_scaled = cv.resize(img2_gray, (new_width, new_height))
            else:
                img1_scaled = img1_gray
                img2_scaled = img2_gray
            
            thresholds = [(30, 100), (50, 150), (70, 200)]
            
            for low, high in thresholds:
                edges1 = cv.Canny(img1_scaled, low, high)
                edges2 = cv.Canny(img2_scaled, low, high)
                
                density1 = np.sum(edges1) / (255 * edges1.size)
                density2 = np.sum(edges2) / (255 * edges2.size)
                density_diff = abs(density1 - density2)
                
                if edges1.size == edges2.size:
                    overlap = cv.bitwise_and(edges1, edges2)
                    overlap_ratio = np.sum(overlap) / max(np.sum(edges1), np.sum(edges2), 1)
                    structure_diff = 1 - overlap_ratio
                else:
                    structure_diff = 1.0
                
                sobelx1 = cv.Sobel(img1_scaled, cv.CV_64F, 1, 0, ksize=3)
                sobely1 = cv.Sobel(img1_scaled, cv.CV_64F, 0, 1, ksize=3)
                _, angle1 = cv.cartToPolar(sobelx1, sobely1, angleInDegrees=True)
                
                sobelx2 = cv.Sobel(img2_scaled, cv.CV_64F, 1, 0, ksize=3)
                sobely2 = cv.Sobel(img2_scaled, cv.CV_64F, 0, 1, ksize=3)
                _, angle2 = cv.cartToPolar(sobelx2, sobely2, angleInDegrees=True)
                
                hist1 = cv.calcHist([angle1.astype(np.float32)], [0], None, [36], [0, 360])
                hist2 = cv.calcHist([angle2.astype(np.float32)], [0], None, [36], [0, 360])
                
                hist1 = cv.normalize(hist1, hist1).flatten()
                hist2 = cv.normalize(hist2, hist2).flatten()
                
                orientation_diff = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
                
                score = 0.5 * density_diff + 0.3 * structure_diff + 0.2 * orientation_diff
                all_scores.append(score)
        
        return np.mean(all_scores)
    
    def compareImgs_Segmen(self, img1, img2):
        width, height = img1.shape[1], img1.shape[0]
        img2 = cv.resize(img2, (width, height))

        img1_blur = cv.GaussianBlur(img1, (3, 3), 0.8)
        img2_blur = cv.GaussianBlur(img2, (3, 3), 0.8)

        lab1 = cv.cvtColor(img1_blur, cv.COLOR_BGR2LAB)
        lab2 = cv.cvtColor(img2_blur, cv.COLOR_BGR2LAB)
        
        mean1, std1 = cv.meanStdDev(lab1)
        mean2, std2 = cv.meanStdDev(lab2)
        
        color_diff = 0
        for i in range(3):
            mean_diff = abs(mean1[i][0] - mean2[i][0]) / 255.0
            std_diff = abs(std1[i][0] - std2[i][0]) / 128.0
            color_diff += (mean_diff + std_diff) / 2
        
        color_diff /= 3.0
        
        edge_diff = self.compareImgs_edge(img1, img2)
        
        final_score = 0.7 * color_diff + 0.3 * edge_diff
        
        return final_score
    
    def compareImgs_hist_color(self, img1, img2):
        width, height = img1.shape[1], img1.shape[0]
        img2_resized = cv.resize(img2, (width, height))
        
        color_spaces = [
            ('BGR', img1, img2_resized),
            ('HSV', cv.cvtColor(img1, cv.COLOR_BGR2HSV), cv.cvtColor(img2_resized, cv.COLOR_BGR2HSV)),
            ('LAB', cv.cvtColor(img1, cv.COLOR_BGR2LAB), cv.cvtColor(img2_resized, cv.COLOR_BGR2LAB))
        ]
        
        total_diff = 0
        weights = [0.4, 0.3, 0.3]
        
        for idx, (space_name, img1_space, img2_space) in enumerate(color_spaces):
            channel_diffs = []
            
            for channel in range(3):
                hist1 = cv.calcHist([img1_space], [channel], None, [64], [0, 256])
                hist2 = cv.calcHist([img2_space], [channel], None, [64], [0, 256])
                
                cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
                cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
                
                channel_diff = 1 - cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
                channel_diffs.append(channel_diff)
            
            space_diff = np.mean(channel_diffs)
            total_diff += weights[idx] * space_diff
        
        return total_diff
    
    def ColorLayoutDescriptor(self, img):
        height, weight = img.shape[:2]

        grids = []
        for i in range(8):
            for j in range(8):
                grid = img[i*height//8 : (i+1)*height//8, j*weight//8 : (j+1)*weight//8]
                mean_color = cv.mean(grid)[:3] 
                grids.append(mean_color)

        grids_array = np.array(grids).reshape(8, 8, 3)

        cld_descriptor = []
        for channel in range(3):
            channel_data = grids_array[:, :, channel].astype(np.float32)
            dct_coeffs = cv.dct(channel_data)
                
            zigzag = []
            for sum_idx in range(16): 
                if sum_idx % 2 == 0: 
                    for i in range(min(sum_idx, 7), max(-1, sum_idx-7-1), -1):
                        j = sum_idx - i
                        if 0 <= j < 8:
                            zigzag.append(dct_coeffs[i, j])
                else: 
                    for i in range(max(0, sum_idx-7), min(sum_idx+1, 8)):
                        j = sum_idx - i
                        if 0 <= j < 8:
                            zigzag.append(dct_coeffs[i, j])

            zigzag_segment = zigzag[:10]
            if np.max(np.abs(zigzag_segment)) > 0:
                zigzag_segment = zigzag_segment / np.max(np.abs(zigzag_segment))

            quantized_coeffs = [int(coeff * 100) for coeff in zigzag_segment] 
            cld_descriptor.extend(quantized_coeffs)
        
        return np.array(cld_descriptor)
    
    def compareImgs_CLD(self, img1, img2):
        img1_YCrCb = cv.cvtColor(img1, cv.COLOR_BGR2YCrCb)
        img2_YCrCb = cv.cvtColor(img2, cv.COLOR_BGR2YCrCb)
        
        width, height = img1_YCrCb.shape[1], img1_YCrCb.shape[0]
        img2_YCrCb = cv.resize(img2_YCrCb, (width, height))
        
        cld1 = self.ColorLayoutDescriptor(img1_YCrCb)
        cld2 = self.ColorLayoutDescriptor(img2_YCrCb)

        cos_sim = np.dot(cld1, cld2) / (np.linalg.norm(cld1) * np.linalg.norm(cld2))
        distance = ((1 - cos_sim) / 2) * 100
        
        return distance
    
    def calculate_similarity(self, query_img, database_img):
        diff_edge = self.compareImgs_edge(query_img, database_img)
        diff_hist = self.compareImgs_hist_color(query_img, database_img)
        diff_CLD = self.compareImgs_CLD(query_img, database_img)
        diff_segmen = self.compareImgs_Segmen(query_img, database_img)
        
        edge_scaled = diff_edge * 5.0
        hist_scaled = diff_hist * 1.0  
        cld_scaled = diff_CLD * 1.0
        segmen_scaled = diff_segmen * 1.0
        
        weights = [0.25, 0.20, 0.35, 0.20]
        
        diff_total = (
            weights[0] * edge_scaled +
            weights[1] * hist_scaled + 
            weights[2] * cld_scaled +
            weights[3] * segmen_scaled
        )
        
        return diff_total
    
    def _load_database(self):
        """Load all database images into memory"""
        if self.database_imgs is None:
            print("Loading database images...")
            self.database_imgs = []
            for img_path in self.database_paths:
                img = self.resize_if_large(cv.imread(img_path))
                if img is not None:
                    self.database_imgs.append((img_path, img))
            print(f"Loaded {len(self.database_imgs)} images")
    
    def find_best_match(self, query_img):
        self._load_database()
        best_path, best_img, best_score = None, None, float('inf')
        
        for path, img in self.database_imgs:
            score = self.calculate_similarity(query_img, img)
            if score < best_score:
                best_score, best_path, best_img = score, path, img
        
        return best_path, best_img, best_score
    
    def retrieve_similar(self, query_img, threshold):
        self._load_database()
        
        # Calculate all scores first for normalization
        all_scores = []
        for _, img in self.database_imgs:
            all_scores.append(self.calculate_similarity(query_img, img))
        
        min_s = min(all_scores)
        max_s = max(all_scores)
        score_range = max_s - min_s
        
        results = []
        for i, (path, _) in enumerate(self.database_imgs):
            if score_range > 0:
                # Convert to normalized similarity (higher = more similar)
                normalized = 1 - ((all_scores[i] - min_s) / score_range)
            else:
                normalized = 1.0
            
            if normalized >= threshold:
                results.append((normalized, path))
        
        return sorted(results, reverse=True, key=lambda x: x[0])