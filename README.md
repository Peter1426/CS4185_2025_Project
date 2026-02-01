# CS4185 Image Retrieval System

## Project Overview  
This is an *individual project* developed for **CS4185 Multimedia Technologies and Applications** at City University of Hong Kong. The system retrieves visually similar images from a database of 1,000 images using multiple feature comparison methods implemented with OpenCV.

**Key Features:**
- **Single Image Retrieval:** Finds the most similar image from the database
- **Batch Testing:** Tests retrieval accuracy across 7 predefined query images
- **Precision/Recall Calculation:** Measures retrieval performance with a fixed similarity threshold
- **Custom Query Support:** Allows user-provided images for retrieval

## Project Performance  
The system implements **multiple feature comparison methods** with weighted combination:

### Achieved Results:
- **Task 1 (Correct Matches):** 5/7 correct matches → **15/20 marks**
- **Task 3 (Precision):** Below 40% threshold → **0/20 marks**
- **Task 4 (Recall):** Above 60% threshold → **20/20 marks**

*Note: This project focused on implementing and combining basic OpenCV features rather than advanced machine learning or database optimization approaches.*

---

## QUICK START:
1. Go to `/binary` folder
2. Double-click `"run_project.bat"` (Windows) or `"run_project.sh"` (MacOS/Linux)
3. Program will auto-install packages and run

## MANUAL SETUP:
1. Install Python 3.9.12
2. Run: `pip install -r requirements.txt`
3. Go to `/source` folder
4. Run: `python "CS4185 Image retrieval system.py"`

## REQUIRED PACKAGES:
- OpenCV-python==4.6.0.66
- NumPy
- pillow

## FOLDER STRUCTURE:
```
/source    - Python code and images
/binary    - Executables and launcher scripts
```

## SYSTEM FEATURES:
1. **Single image retrieval** – Compare one query image against database
2. **Test all 7 queries** – Batch test with accuracy reporting
3. **Retrieve similar images with precision/recall** – Measure retrieval performance (threshold: 0.7)
4. **Exit** – Terminate program

**Note:** Features 1-3 may take a few minutes to compute. Feature 4 exits immediately.

## ⚠️ IMPORTANT NOTES:
- For feature 3, when run once, a folder containing the retrieval results will be created in the source folder.
- If required to run feature 3 again on the same category of image, **please delete the created folder of that category first**. Otherwise, an error will occur.

## Technologies Used
- **Language:** Python 3.9.12
- **Libraries:** OpenCV, NumPy, Pillow
- **Environment:** Cross-platform (Windows/macOS via batch/shell scripts)

## Sample Results


| Single Image Retrieval | Batch Testing |
|------------------------|---------------|
| <img width="1300" height="850" alt="Single Image Retrieval" src="https://github.com/user-attachments/assets/7382cea3-1768-4800-ad85-b0126a85d819" /> | <img width="900" height="1000" alt="Batch Testing" src="https://github.com/user-attachments/assets/0d6959af-6341-4540-bdef-db14de55b25a" /> |

| Precision/Recall Output |
|-------------------------|
| <img width="500" height="684" alt="Precision and Recall Output" src="https://github.com/user-attachments/assets/7bd51ad1-ec97-4191-bba4-0b298905cb85" /> |


## Academic Context
This project was completed as part of **CS4185 Multimedia Technologies and Applications** at City University of Hong Kong. The work focuses on:
- Implementing multiple image feature extraction techniques
- Experimenting with weighted feature combination
- Analyzing trade-offs between precision and recall in retrieval systems
