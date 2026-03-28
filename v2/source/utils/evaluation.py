import os
import shutil
from utils.image_utils import get_category_from_filename

# Calculate precision and recall from retrieval results
def calculate_precision_recall(results, expected_category):
    if not results:
        return 0, 0
    
    relevant_retrieved = 0
    for _, img_path in results:
        if get_category_from_filename(img_path) == expected_category:
            relevant_retrieved += 1
    
    total_relevant = 100
    precision = relevant_retrieved / len(results)
    recall = relevant_retrieved / total_relevant
    
    return precision, recall

# Save retrieved images to output directory
def save_retrieved_images(results, output_dir):
    # Delete existing folder if it exists
    if os.path.exists(output_dir):
        print(f"\nWarning: Output folder '{output_dir}' already exists!")
        choice = input("Delete existing folder? (y/n): ").strip().lower()
        if choice == 'y' or choice == 'yes':
            shutil.rmtree(output_dir)
            print(f"Deleted existing folder: {output_dir}")
        else:
            print("Cancelled. Please delete folder manually or choose a different threshold.")
            return None
            
    os.makedirs(output_dir)
    
    for i, (similarity, img_path) in enumerate(results):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{i+1:03d}_{similarity:.3f}_{img_name}")
        shutil.copy(img_path, output_path)
    
    return output_dir