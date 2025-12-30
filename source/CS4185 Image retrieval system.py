import cv2 as cv
import numpy as np
from glob import glob
import os
import shutil

# the directory of the image database
database_dir = "image.orig"

def compareImgs_edge(img1, img2):

    # Resize img2 to img1
    width, height = img1.shape[1], img1.shape[0]
    img2 = cv.resize(img2, (width, height))

    # Convert to grayscale
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Multi-scale processing
    scales = [1.0, 0.5]  # Original and half size
    all_scores = []
    
    for scale in scales:
        if scale != 1.0:
            # Resize images
            new_width = int(width * scale)
            new_height = int(height * scale)
            img1_scaled = cv.resize(img1_gray, (new_width, new_height))
            img2_scaled = cv.resize(img2_gray, (new_width, new_height))
        else:
            img1_scaled = img1_gray
            img2_scaled = img2_gray
        
        # Edge detection at multiple thresholds
        thresholds = [(30, 100), (50, 150), (70, 200)]
        
        for low, high in thresholds:
            edges1 = cv.Canny(img1_scaled, low, high)
            edges2 = cv.Canny(img2_scaled, low, high)
            
            # Edge density difference
            density1 = np.sum(edges1) / (255 * edges1.size)
            density2 = np.sum(edges2) / (255 * edges2.size)
            density_diff = abs(density1 - density2)
            
            # Edge overlap (structural similarity)
            if edges1.size == edges2.size:
                overlap = cv.bitwise_and(edges1, edges2)
                overlap_ratio = np.sum(overlap) / max(np.sum(edges1), np.sum(edges2), 1)
                structure_diff = 1 - overlap_ratio
            else:
                structure_diff = 1.0
            
            # Orientation comparison
            sobelx1 = cv.Sobel(img1_scaled, cv.CV_64F, 1, 0, ksize=3)
            sobely1 = cv.Sobel(img1_scaled, cv.CV_64F, 0, 1, ksize=3)
            magnitude1, angle1 = cv.cartToPolar(sobelx1, sobely1, angleInDegrees=True)
            
            sobelx2 = cv.Sobel(img2_scaled, cv.CV_64F, 1, 0, ksize=3)
            sobely2 = cv.Sobel(img2_scaled, cv.CV_64F, 0, 1, ksize=3)
            magnitude2, angle2 = cv.cartToPolar(sobelx2, sobely2, angleInDegrees=True)
            
            # Orientation histograms
            hist1 = cv.calcHist([angle1.astype(np.float32)], [0], None, [36], [0, 360])
            hist2 = cv.calcHist([angle2.astype(np.float32)], [0], None, [36], [0, 360])
            
            hist1 = cv.normalize(hist1, hist1).flatten()
            hist2 = cv.normalize(hist2, hist2).flatten()
            
            orientation_diff = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
            
            # Combine scores
            score = 0.5 * density_diff + 0.3 * structure_diff + 0.2 * orientation_diff
            all_scores.append(score)
    
    return np.mean(all_scores)

def compareImgs_Segmen(img1, img2):
    
    # Resize img2 to img1
    width, height = img1.shape[1], img1.shape[0]
    img2 = cv.resize(img2, (width, height))

    # Apply blur for better color analysis
    img1_blur = cv.GaussianBlur(img1, (3, 3), 0.8)
    img2_blur = cv.GaussianBlur(img2, (3, 3), 0.8)

    # 1. Color analysis
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
    
    # Use compare edge function
    edge_diff = compareImgs_edge(img1, img2)
    
    # Combine scores
    final_score = 0.7 * color_diff + 0.3 * edge_diff
    
    return final_score

def compareImgs_hist_color(img1, img2):

    # Resize img2 to img1
    width, height = img1.shape[1], img1.shape[0]
    img2_resized = cv.resize(img2, (width, height))
    
    # Convert to multiple color spaces
    color_spaces = [
        ('BGR', img1, img2_resized),
        ('HSV', cv.cvtColor(img1, cv.COLOR_BGR2HSV), cv.cvtColor(img2_resized, cv.COLOR_BGR2HSV)),
        ('LAB', cv.cvtColor(img1, cv.COLOR_BGR2LAB), cv.cvtColor(img2_resized, cv.COLOR_BGR2LAB))
    ]
    
    total_diff = 0
    weights = [0.4, 0.3, 0.3]  # Weight for each color space
    
    for idx, (space_name, img1_space, img2_space) in enumerate(color_spaces):
        channel_diffs = []
        
        for channel in range(3):
            # Calculate histogram for this channel
            hist1 = cv.calcHist([img1_space], [channel], None, [64], [0, 256])
            hist2 = cv.calcHist([img2_space], [channel], None, [64], [0, 256])
            
            # Normalize
            cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
            cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
            
            # Use correlation distance
            channel_diff = 1 - cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
            channel_diffs.append(channel_diff)
        
        # Average channel differences for this color space
        space_diff = np.mean(channel_diffs)
        total_diff += weights[idx] * space_diff
    
    return total_diff

def ColorLayoutDescriptor(img):
	height, weight = img.shape[:2]

	# Find grids
	grids = []
	for i in range(8):
		for j in range(8):
			grid = img[i*height//8 : (i+1)*height//8, j*weight//8 : (j+1)*weight//8]
			mean_color = cv.mean(grid)[:3] 
			grids.append(mean_color)

	grids_array = np.array(grids).reshape(8, 8, 3)

	# Apply DCT for each channels
	cld_descriptor = []
	for channel in range(3):
		channel_data = grids_array[:, :, channel].astype(np.float32)
		dct_coeffs = cv.dct(channel_data)
            
        # Take zigzag scan of coefficients 
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

		# Normalize the segment
		zigzag_segment = zigzag[:10]
		if np.max(np.abs(zigzag_segment)) > 0:
			zigzag_segment = zigzag_segment / np.max(np.abs(zigzag_segment))

        # Quantized
		quantized_coeffs = [int(coeff * 100) for coeff in zigzag_segment] 
		cld_descriptor.extend(quantized_coeffs)
	
	return np.array(cld_descriptor)

def compareImgs_CLD(img1, img2):

    # Convert image to YCrCb scale
	img1_YCrCb = cv.cvtColor(img1, cv.COLOR_BGR2YCrCb)
	img2_YCrCb = cv.cvtColor(img2, cv.COLOR_BGR2YCrCb)
	
	width, height = img1_YCrCb.shape[1], img1_YCrCb.shape[0]
	img2_YCrCb = cv.resize(img2_YCrCb, (width, height))
    
	# Extract discriptor
	cld1 = ColorLayoutDescriptor(img1_YCrCb)
	cld2 = ColorLayoutDescriptor(img2_YCrCb)

	# Caluculate cosine distance and normalized
	cos_sim = np.dot(cld1, cld2) / (np.linalg.norm(cld1) * np.linalg.norm(cld2))
	distance = ((1 - cos_sim) / 2) *100
	
	return distance

def calculate_similarity(query_img, database_img):

    # Calculate features of img
    diff_edge = compareImgs_edge(query_img, database_img)
    diff_hist = compareImgs_hist_color(query_img, database_img)
    diff_CLD = compareImgs_CLD(query_img, database_img)
    diff_segmen = compareImgs_Segmen(query_img, database_img)
    
    # Apply scaling
    edge_scaled = diff_edge * 5.0
    hist_scaled = diff_hist * 1.0  
    cld_scaled = diff_CLD * 1.0
    segmen_scaled = diff_segmen * 1.0
    
    # Apply weight
    weights = [0.25, 0.20, 0.35, 0.20]
    
    diff_total = (
        weights[0] * edge_scaled +
        weights[1] * hist_scaled + 
        weights[2] * cld_scaled +
        weights[3] * segmen_scaled
    )
    
    return {
        'total': diff_total,
        'individual': {
            'edge': diff_edge, 
            'hist': diff_hist, 
            'CLD': diff_CLD,
            'segmen': diff_segmen
        }
    }

def resize_if_large(img):
    
    if img is None:
        return None
    height, width = img.shape[:2]
    if width > 1000 or height > 1000:
        return cv.resize(img, (int(width*0.5), int(height*0.5)))
    return img

def retrieval():
    print("1: beach")
    print("2: mountain")
    print("3: food")
    print("4: dinosaur")
    print("5: flower")
    print("6: horse")
    print("7: elephant")
    print("8: Choose a non-default image")
    choice = input("Type in the number to choose a category and type enter to confirm\n")
    category_map = {
        '1': ('beach', "image.query/beach.jpg"),
        '2': ('mountain', "image.query/mountain.jpg"),
        '3': ('dish', "image.query/food.jpg"),
        '4': ('dinosaur', "image.query/dinosaur.jpg"),
        '5': ('flower', "image.query/flower.jpg"),
        '6': ('horse', "image.query/horse.jpg"),
        '7': ('elephant', "image.query/elephant.jpg")
    }
    
    if choice in category_map:
        query_category, query_path = category_map[choice]
        src_input = resize_if_large(cv.imread(query_path))
        print(f"You choose: {choice} - {query_category}\n")
    elif choice == '8':
        # Custom image selection
        filename = input("Enter the filename of your image (e.g., 'myimage.jpg'): ").strip()
        query_path = f"image.query/{filename}"
        
        if not os.path.exists(query_path):
            print(f"Error: File '{query_path}' not found!")
            print("Please make sure the image is in the 'image.query' folder.")
            return

        src_input = resize_if_large(cv.imread(query_path))
        if src_input is None:
            print(f"Error: Could not load image '{query_path}'!")
            print("Please check if the file is a valid image format.")
            return
            
        query_category = "custom"
        print(f"You choose: custom image - {filename}\n")
    else:
        print("Invalid choice")
        return
   
    cv.imshow("Input", src_input)

    # read image database
    database = sorted(glob(database_dir + "/*.jpg"))
    print(f"Searching through {len(database)} images...")

    closest_img = None
    result = None
    best_score = float('inf')

    # Progress tracking
    print("Comparing images: ", end="", flush=True)
    
    for i, img_path in enumerate(database):
        # Show progress every 100 images
        if i % 100 == 0:
            print(f"{i}...", end="", flush=True)
            
        img_rgb = cv.imread(img_path)
        if img_rgb is None:
            continue
            
        # Calculate similarity
        similarity_result = calculate_similarity(src_input, img_rgb)
        diff_total = similarity_result['total']
        
        # Find the best match
        if diff_total < best_score:
            best_score = diff_total
            closest_img = img_rgb
            result = img_path
    
    # Complete progress
    print("1000")  
    print("Search completed!\n")

    # Final result
    matched_category = get_category_from_filename(result)
    if query_category == "custom":
        print(f"The most similar image is: {os.path.basename(result)} ({matched_category})")
        print(f"Total difference score: {best_score:.4f}")
    else:
        is_correct = "✓ CORRECT" if matched_category == query_category else "✗ WRONG"
        print(f"The most similar image is: {os.path.basename(result)} ({matched_category}), {is_correct}")
        print(f"Total difference score: {best_score:.4f}")

    cv.imshow("Best Match", closest_img)
    print("Press any key to close images...")
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_category_from_filename(filepath):
    
    filename = os.path.basename(filepath)
    try:
        # Extract number from filename
        img_number = int(filename.replace('.jpg', ''))
        
        # Map numbers to categories based on image database
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

# test queries fast
def test_all_queries():
    """Faster version with same resize logic as retrieval()"""
    query_images = {
        'beach': "image.query/beach.jpg",
        'mountain': "image.query/mountain.jpg", 
        'food': "image.query/food.jpg",
        'dinosaur': "image.query/dinosaur.jpg",
        'flower': "image.query/flower.jpg",
        'horse': "image.query/horse.jpg",
        'elephant': "image.query/elephant.jpg"
    }
    
    category_mapping = get_query_category_mapping()
    
    print("Fast testing all 7 query images (with resize)...")
    print("Loading database images into memory...")
    
    # Pre-load all database images 
    database_paths = sorted(glob(database_dir + "/*.jpg"))
    database_imgs = []
    failed_loads = 0
    
    # Resize if too large
    for img_path in database_paths:
        img = resize_if_large(cv.imread(img_path))
        if img is not None:
            database_imgs.append((img_path, img))
        else:
            failed_loads += 1
    
    print(f"Successfully loaded {len(database_imgs)}/{len(database_paths)} database images")
    if failed_loads > 0:
        print(f"Warning: Failed to load {failed_loads} images")
    
    # Test each query against pre-loaded database
    correct_matches = 0
    results = {}
    
    print("\n" + "=" * 60)
    print(f"{'Query':<12} {'Expected':<12} {'Matched':<12} {'Status':<10} {'Score':<8}")
    print("-" * 60)
    
    for query_name, query_path in query_images.items():
        if not os.path.exists(query_path):
            print(f"{query_name:<12} {'ERROR':<12} {'File missing':<12} {'✗':<10} {'N/A':<8}")
            continue
            
        expected_category = category_mapping[query_name]
        query_img = resize_if_large(cv.imread(query_path))
        
        if query_img is None:
            print(f"{query_name:<12} {'ERROR':<12} {'Load failed':<12} {'✗':<10} {'N/A':<8}")
            continue
                
        best_match_path = None
        min_diff = float('inf')
        best_score = 0
        
        # Compare against all pre-loaded database images
        for img_path, img_rgb in database_imgs:
            similarity_result = calculate_similarity(query_img, img_rgb)
            diff_total = similarity_result['total']
            
            if diff_total < min_diff:
                min_diff = diff_total
                best_match_path = img_path
                best_score = diff_total
        
        # Check result
        if best_match_path:
            matched_category = get_category_from_filename(best_match_path)
            is_correct = (matched_category == expected_category)
            
            if is_correct:
                correct_matches += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"
            
            print(f"{query_name:<12} {expected_category:<12} {matched_category:<12} {status:<10} {best_score:.4f}")
            results[query_name] = (is_correct, matched_category, best_match_path, best_score)
        else:
            print(f"{query_name:<12} {'ERROR':<12} {'No match':<12} {'✗':<10} {'N/A':<8}")
            results[query_name] = (False, "none", None, float('inf'))
    
    print("=" * 60)
    print(f"Task 1 Result: {correct_matches}/7 correct matches")
    
    if correct_matches >= 6:
        marks = "20/20"
    elif correct_matches == 5:
        marks = "15/20" 
    elif correct_matches == 4:
        marks = "10/20"
    else:
        marks = "0/20"
    
    print(f"Marks for Task 1: {marks}")
    return correct_matches, results

def calculate_precision_recall(query_img_path, threshold):
    
    # Get expected category from query filename
    query_filename = os.path.basename(query_img_path).lower()
    category_mapping = get_query_category_mapping()
    
    expected_category = None
    for query_name, category in category_mapping.items():
        if query_name in query_filename:
            expected_category = category
            break
    
    if not expected_category:
        # For custom images, ask for expected category
        print("\nAvailable categories:")
        categories = ['african', 'beach', 'building', 'bus', 'dinosaur', 'elephant', 'flower', 'horse', 'mountain', 'dish']
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")
        cat_choice = input("Select expected category number: ").strip()
        try:
            expected_category = categories[int(cat_choice) - 1]
        except:
            print("Invalid category selection!")
            return 0, 0
    
    query_img = resize_if_large(cv.imread(query_img_path))
    database = sorted(glob(database_dir + "/*.jpg"))
    
    # Create output directory
    output_dir = f"retrieved_{expected_category}_{os.path.basename(query_img_path).split('.')[0]}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    results = []
    
    print(f"\nRetrieving similar images for {os.path.basename(query_img_path)}...")
    print(f"Threshold: {threshold}, Expected category: {expected_category}")
    
    # Find the range of scores to normalize
    all_scores = []
    for img_path in database:
        img_rgb = resize_if_large(cv.imread(img_path))
        similarity_result = calculate_similarity(query_img, img_rgb)
        diff_total = similarity_result['total']
        all_scores.append(diff_total)
    
    # Normalize scores to 0-1 range (0 = most similar, 1 = least similar)
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score
    
    print(f"Score range: {min_score:.4f} to {max_score:.4f}")
    
    for img_path in database:
        img_rgb = resize_if_large(cv.imread(img_path))
        
        # Calculate similarity
        similarity_result = calculate_similarity(query_img, img_rgb)
        diff_total = similarity_result['total']
        
        # Convert to NORMALIZED similarity score (0 = perfect match, 1 = no match)
        normalized_similarity = 1.0 - ((diff_total - min_score) / score_range) if score_range > 0 else 1.0
        
        # Debug output for first few images
        if len(results) < 3:
            print(f"Debug: {os.path.basename(img_path)} - Raw: {diff_total:.4f}, Normalized: {normalized_similarity:.4f}")
        
        if normalized_similarity >= threshold:
            results.append((normalized_similarity, img_path))
    
    # Sort by similarity (descending)
    results.sort(reverse=True, key=lambda x: x[0])
    
    # Save results to output folder
    for i, (similarity, img_path) in enumerate(results):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{i+1:03d}_{similarity:.3f}_{img_name}")
        shutil.copy(img_path, output_path)
    
    print(f"Found {len(results)} similar images, saved to '{output_dir}' folder")
    
    # Calculate precision and recall
    if not results:
        print("No images retrieved! Try lowering the threshold.")
        return 0, 0
    
    # Count relevant images in results
    relevant_retrieved = 0
    for similarity, img_path in results:
        if get_category_from_filename(img_path) == expected_category:
            relevant_retrieved += 1
    
    # Total relevant images in database (100 per category)
    total_relevant = 100
    
    precision = relevant_retrieved / len(results) if results else 0
    recall = relevant_retrieved / total_relevant
    
    print(f"\n--- RETRIEVAL RESULTS ---")
    print(f"Total retrieved: {len(results)}")
    print(f"Relevant retrieved: {relevant_retrieved}")
    print(f"Total relevant in database: {total_relevant}")
    print(f"Precision: {precision:.2%} ({relevant_retrieved}/{len(results)})")
    print(f"Recall: {recall:.2%} ({relevant_retrieved}/{total_relevant})")
    
    return precision, recall

def main_menu():
    while True:
        print("\n" + "=" * 50)
        print("IMAGE RETRIEVAL SYSTEM")
        print("=" * 50)
        print("1. Test single query image")
        print("2. Test all 7 queries - Count correct matches")
        print("3. Retrieve similar images with threshold + Precision/Recall")
        print("4. Exit")
        
        choice = input("\nChoose option: ").strip()
        
        if choice == '1':
            retrieval() 
        elif choice == '2':
            test_all_queries()
        elif choice == '3':
            query_images = {
                'beach': "image.query/beach.jpg",
                'mountain': "image.query/mountain.jpg", 
                'food': "image.query/food.jpg",
                'dinosaur': "image.query/dinosaur.jpg",
                'flower': "image.query/flower.jpg",
                'horse': "image.query/horse.jpg",
                'elephant': "image.query/elephant.jpg"
            }
            
            print("\nAvailable query images:")
            for i, (name, path) in enumerate(query_images.items(), 1):
                print(f"{i}. {name}")
            print("8. Custom image")
            
            query_choice = input("\nChoose query image: ").strip()
            
            if query_choice == '8':
                filename = input("Enter custom image filename: ").strip()
                query_path = f"image.query/{filename}"
                if not os.path.exists(query_path):
                    print(f"Error: File '{query_path}' not found!")
                    continue
            else:
                query_names = list(query_images.keys())
                try:
                    query_idx = int(query_choice) - 1
                    if 0 <= query_idx < len(query_names):
                        query_name = query_names[query_idx]
                        query_path = query_images[query_name]
                    else:
                        print("Invalid choice!")
                        continue
                except:
                    print("Invalid choice!")
                    continue
            
            # FIXED THRESHOLD OF 0.7
            threshold = 0.7
            print(f"\nUsing fixed similarity threshold: {threshold}")
            print("Retrieving similar images and calculating precision/recall...")
            
            # Perform retrieval and calculate precision/recall
            precision, recall = calculate_precision_recall(query_path, threshold)
            
            print(f"\n--- FINAL RESULTS ---")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            
            # Grading based on results
            precision_marks = 20 if precision >= 0.6 else (5 if precision >= 0.4 else 0)
            recall_marks = 20 if recall >= 0.6 else (5 if recall >= 0.4 else 0)
            
            print(f"Marks for Precision: {precision_marks}/20")
            print(f"Marks for Recall: {recall_marks}/20")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()