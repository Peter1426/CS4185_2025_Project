# CS4185 Image Retrieval System

## Project Overview  
This is an *individual project* developed for **CS4185 Multimedia Technologies and Applications** at City University of Hong Kong. The system retrieves visually similar images from a database of 1,000 images using multiple feature comparison methods implemented with OpenCV. *The v2 version was implemented after the course ended using MobileNetV2 + FAISS.*

**Key Features:**
- **Dual Retrieval Systems:** Switch between Traditional (OpenCV) and Deep Learning (MobileNetV2 + FAISS) [for v2]
- **Single Image Retrieval:** Finds the most similar image from the database
- **Batch Testing:** Tests retrieval accuracy across 7 predefined query images
- **Precision/Recall Calculation:** Measures retrieval performance with a fixed similarity threshold
- **Custom Query Support:** Allows user-provided images for retrieval
- **System Comparison:** Side-by-side comparison of both retrieval methods

## System Comparison

| Feature | Traditional System (v1) | Deep Learning System (v2) |
|---------|-------------------|---------------------|
| **Feature Extraction** | Hand-crafted (Edge, Color, CLD, Segmentation) | MobileNetV2 CNN |
| **Search Method** | Linear scan O(n) | FAISS index O(log n) |
| **Search Speed** | ~10-30 seconds per query | ~50-200 ms per query |
| **Default Threshold** | 0.7 | 0.01 |
| **Memory Usage** | Low | Higher (1280-dim vectors) |
| **Pre-built Index** | No (recomputed each run) | Yes (saved for reuse) |
| **Accuracy** | Limited | Higher (deep features) |

## Project Performance  
The v1 system implements **multiple feature comparison methods** with weighted combination.
The v2 system implements a **deep learning-based** approach using MobileNetV2 + FAISS.

### Achieved Results:
- **v1 system**
    - **Task 1 (Correct Matches):** 5/7 correct matches → **15/20 marks**
    - **Task 3 (Precision):** Below 40% threshold → **0/20 marks**
    - **Task 4 (Recall):** Above 60% threshold → **20/20 marks**
- **v2 system**
    - **Task 1 (Correct Matches):** 6/7 correct matches → **20/20 marks**
    - **Task 3 (Precision):** Above 60% threshold → **20/20 marks**
    - **Task 4 (Recall):** Above 60% threshold → **20/20 marks**

*Note: The v1 system focused on implementing and combining basic OpenCV features rather than advanced machine learning or database optimization approaches.*

---

## QUICK START:
1. Go to `/binary` folder under v1 or v2 depends on which version to use
2. Double-click `"run_project.bat"` (Windows) or `"run_project.sh"` (MacOS/Linux)
3. Program will auto-install packages and run
4. Choose the system to run (for v2 only)

## MANUAL SETUP:
1. Install Python 3.9.12
2. Run: `pip install -r requirements.txt`
3. Go to `/source` folder
4. Run: `python "CS4185 Image retrieval system.py"` for v1 Or Run: `python main.py` for v2

## REQUIRED PACKAGES:
- OpenCV-python==4.6.0.66
- NumPy
- pillow
- tensorflow (for v2)
- faiss-cpu  (for v2)

## FOLDER STRUCTURE:
*v1*
```
Program/
├── source/
│   ├── CS4185 Image retrieval system.py   # Main entry point
│   ├── image.orig/   # Database images (1000 images)
│   └── image.query/  # Query images (7 images)
└── binary/
    ├── run_project.bat     # Windows launcher
    ├── run_project.sh  # MacOS launcher
    └── requirements.txt    # Dependencies
```

*v2*
```
Program/
├── source/
│   ├── main.py   # Main entry point
│   ├── retrieval/
│   │   ├── init.py
│   │   ├── base_retrieval.py   # Base class
│   │   ├── traditional_retrieval.py    # Original OpenCV system
│   │   └── deep_retrieval.py   # MobileNetV2 + FAISS system
│   ├── utils/
│   │   ├── init.py
│   │   ├── image_utils.py  # Image utilities
│   │   └── evaluation.py   # Precision/recall evaluation
│   ├── faiss_index/  # Saved FAISS index (auto-generated)
│   ├── image.orig/   # Database images (1000 images)
│   └── image.query/  # Query images (7 images)
└── binary/
    ├── run_project.bat     # Windows launcher
    ├── run_project.sh  # MacOS launcher
    └── requirements.txt    # Dependencies
```

## SYSTEM FEATURES:

### After selecting a system, you can:

1. **Single image retrieval** – Compare one query image against database
2. **Test all 7 queries** – Batch test with accuracy reporting
3. **Retrieve similar images** – Get all images above similarity threshold with precision/recall calculation
    - **Traditional System default threshold: 0.7** (press Enter to use)
    - **Deep Learning System default threshold: 0.01** (press Enter to use)
4. **Switch to different system** – Change between Traditional and Deep Learning
5. **Exit** – Terminate program

### System Comparison (Option 3):
- Runs both systems side by side
- Shows matched categories and scores for each query
- Displays correct match counts for both systems

## ⚠️ IMPORTANT NOTES:
- **First run of Deep Learning System:** Will take 2-5 minutes to build the FAISS index (extracting features from 1,000 images)
- **Subsequent runs:** Loads saved index instantly for fast retrieval
- **FAISS Index Location:** Saved in `/source/faiss_index/` - delete to rebuild
- **Memory:** Deep learning system requires ~500MB RAM for the index

---

## TECHNICAL DETAILS

### Traditional System Features:
- **Edge Detection:** Multi-scale Canny edge detection with orientation histograms
- **Color Analysis:** Histogram comparison in BGR, HSV, and LAB color spaces
- **Color Layout Descriptor (CLD):** DCT-based color layout extraction
- **Segmentation:** Color-based segmentation using LAB color space
- **Weighted Combination:** 25% Edge, 20% Color, 35% CLD, 20% Segmentation
- **Default Threshold:** 0.7 

### Deep Learning System:
- **Feature Extractor:** MobileNetV2 pre-trained on ImageNet (1280-dimensional features)
- **Similarity Search:** FAISS (Facebook AI Similarity Search) with L2 distance
- **Index Type:** Flat L2 index
- **Performance:** Pre-built index for instant retrieval after first build
- **Persistence:** Index saved to disk for future sessions
- **Default Threshold:** 0.01

---
## Sample Results

### Single Image Retrieval
| V1 | V2 |
|------------------------|---------------|
| <img width="1300" height="850" alt="Single Image Retrieval v1" src="https://github.com/user-attachments/assets/7382cea3-1768-4800-ad85-b0126a85d819" /> | <img width="1400" height="950" alt="Single Image Retrieval v2" src="https://github.com/Peter1426/CS4185_2025_Project/blob/main/screenshots/Single%20Image%20Retrieval%20v2.png" /> |

### Batch Testing
| V1 | V2 |
|------------------------|---------------|
| <img width="900" height="1000" alt="Batch Testing v1" src="https://github.com/user-attachments/assets/0d6959af-6341-4540-bdef-db14de55b25a" /> | <img width="800" height="900" alt="Batch Testing v1" src="https://github.com/Peter1426/CS4185_2025_Project/blob/main/screenshots/Batch%20Testing%20v2.png" /> |

### Precision/Recall Output
| V1 | V2 |
|-------------------------|-------------------------|
| <img width="400" height="584" alt="Precision and Recall Output v1" src="https://github.com/user-attachments/assets/7bd51ad1-ec97-4191-bba4-0b298905cb85" /> | <img width="500" height="684" alt="Precision and Recall Output v2" src="https://github.com/Peter1426/CS4185_2025_Project/blob/main/screenshots/Precision%20and%20Recall%20Output%20v2.png" />

### Compare Result
<img width="500" height="500" alt="Precision and Recall Output v1" src="https://github.com/Peter1426/CS4185_2025_Project/blob/main/screenshots/Compare%20Result.png" />

---

## Technologies Used
- **Language:** Python 3.9.12
- **Libraries:** OpenCV, NumPy, Pillow
- **Feature Extraction:** TensorFlow (MobileNetV2)
- **Similarity Search:** FAISS (Facebook AI Similarity Search)
- **Numerical Computing:** NumPy 1.24.3
- **Image Processing:** Pillow
- **Environment:** Cross-platform (Windows/macOS via batch/shell scripts)

---

## Academic Context
This project was completed as part of **CS4185 Multimedia Technologies and Applications** at City University of Hong Kong. The work focuses on:
- Implementing multiple image feature extraction techniques
- Experimenting with weighted feature combination
- Analyzing trade-offs between precision and recall in retrieval systems
