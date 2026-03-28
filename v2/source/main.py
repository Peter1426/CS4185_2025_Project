import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disable TensorFlow oneDNN custom operations warnings

from retrieval.traditional_retrieval import TraditionalRetrievalSystem
from retrieval.deep_retrieval import DeepRetrievalSystem
from utils.image_utils import resize_if_large, get_category_from_filename, get_query_category_mapping
from utils.evaluation import calculate_precision_recall, save_retrieved_images

class ImageRetrievalApp:
    
    def __init__(self):
        self.traditional_system = None
        self.deep_system = None
        self.current_system = None
        self.database_dir = "image.orig"
    
    def select_system(self):
        print("\n" + "=" * 60)
        print("SELECT IMAGE RETRIEVAL SYSTEM")
        print("=" * 60)
        print("1. Traditional System (Edge + Color + CLD + Segmentation)")
        print("2. Deep Learning System (MobileNetV2 + FAISS)")
        print("3. Compare Both Systems")
        print("4. Exit")
        
        choice = input("\nChoose system (1-4): ").strip()
        
        if choice == '1':
            if self.traditional_system is None:
                print("\nInitializing Traditional System...")
                self.traditional_system = TraditionalRetrievalSystem(self.database_dir)
            self.current_system = self.traditional_system
            print(f"\nUsing: {self.current_system.get_system_name()}")
            return True
            
        elif choice == '2':
            if self.deep_system is None:
                print("\nInitializing Deep Learning System...")
                print("This may take a moment for first-time setup...")
                self.deep_system = DeepRetrievalSystem(self.database_dir, use_gpu=False)
                self.deep_system.build_index()
            self.current_system = self.deep_system
            print(f"\nUsing: {self.current_system.get_system_name()}")
            return True
            
        elif choice == '3':
            self.compare_systems()
            return True
            
        elif choice == '4':
            return False
            
        else:
            print("Invalid choice!")
            return True
    
    def compare_systems(self):
        print("\n" + "=" * 60)
        print("COMPARING BOTH SYSTEMS")
        print("=" * 60)
        
        if self.traditional_system is None:
            print("\nInitializing Traditional System...")
            self.traditional_system = TraditionalRetrievalSystem(self.database_dir)
        
        if self.deep_system is None:
            print("\nInitializing Deep Learning System...")
            self.deep_system = DeepRetrievalSystem(self.database_dir, use_gpu=False)
            self.deep_system.build_index()
        
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
        
        print("\n" + "=" * 70)
        print(f"{'Query':<12} {'Traditional Match':<18} {'Deep Match':<18}")
        print("-" * 70)
        
        traditional_correct = 0
        deep_correct = 0
        
        for query_name, query_path in query_images.items():
            if not os.path.exists(query_path):
                continue
            
            expected = category_mapping[query_name]
            query_img = resize_if_large(cv.imread(query_path))
            
            if query_img is None:
                continue
            
            # Traditional system
            trad_path, _, _ = self.traditional_system.find_best_match(query_img)
            trad_match = get_category_from_filename(trad_path) if trad_path else "N/A"
            if trad_match == expected:
                traditional_correct += 1
            
            # Deep system
            deep_path, _, _ = self.deep_system.find_best_match(query_img)
            deep_match = get_category_from_filename(deep_path) if deep_path else "N/A"
            if deep_match == expected:
                deep_correct += 1
            
            print(f"{query_name:<12} {trad_match:<18} {deep_match:<18}")
        
        print("-" * 70)
        print(f"\nCorrect Matches (Task 1):")
        print(f"  Traditional System: {traditional_correct}/7")
        print(f"  Deep Learning System: {deep_correct}/7")
        
        input("\nPress Enter to continue...")
    
    def test_single_query(self):
        if self.current_system is None:
            print("Please select a system first!")
            return
        
        print("\n1: beach")
        print("2: mountain")
        print("3: food")
        print("4: dinosaur")
        print("5: flower")
        print("6: horse")
        print("7: elephant")
        print("8: Choose a non-default image")
        
        choice = input("Type in the number to choose a category: ").strip()
        
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
            query_img = resize_if_large(cv.imread(query_path))
            print(f"You choose: {choice} - {query_category}\n")
        elif choice == '8':
            filename = input("Enter the filename of your image: ").strip()
            query_path = f"image.query/{filename}"
            if not os.path.exists(query_path):
                print(f"Error: File '{query_path}' not found!")
                return
            query_img = resize_if_large(cv.imread(query_path))
            if query_img is None:
                print(f"Error: Could not load image '{query_path}'!")
                return
            query_category = "custom"
            print(f"You choose: custom image - {filename}\n")
        else:
            print("Invalid choice")
            return
        
        cv.imshow("Input", query_img)
        
        print(f"Searching using {self.current_system.get_system_name()}...")
        best_match_path, best_match_img, best_score = self.current_system.find_best_match(query_img)
        
        matched_category = get_category_from_filename(best_match_path) if best_match_path else "unknown"
        
        if query_category == "custom":
            print(f"The most similar image is: {os.path.basename(best_match_path)} ({matched_category})")
            print(f"Distance/Score: {best_score:.4f}")
        else:
            is_correct = "✓ CORRECT" if matched_category == query_category else "✗ WRONG"
            print(f"The most similar image is: {os.path.basename(best_match_path)} ({matched_category}), {is_correct}")
            print(f"Distance/Score: {best_score:.4f}")
        
        cv.imshow("Best Match", best_match_img)
        print("Press any key to close images...")
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def test_all_queries(self):
        if self.current_system is None:
            print("Please select a system first!")
            return
        
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
        
        print(f"\nTesting all 7 queries using {self.current_system.get_system_name()}...")
        
        correct_matches = 0
        
        print("\n" + "=" * 70)
        print(f"{'Query':<12} {'Expected':<12} {'Matched':<12} {'Status':<10} {'Score':<12}")
        print("-" * 70)
        
        for query_name, query_path in query_images.items():
            if not os.path.exists(query_path):
                continue
            
            expected = category_mapping[query_name]
            query_img = resize_if_large(cv.imread(query_path))
            
            if query_img is None:
                continue
            
            best_match_path, _, best_score = self.current_system.find_best_match(query_img)
            matched_category = get_category_from_filename(best_match_path) if best_match_path else "unknown"
            is_correct = (matched_category == expected)
            
            if is_correct:
                correct_matches += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"
            
            print(f"{query_name:<12} {expected:<12} {matched_category:<12} {status:<10} {best_score:<12.4f}")
        
        print("-" * 70)
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
        input("\nPress Enter to continue...")
    
    def retrieve_similar_images(self):
        if self.current_system is None:
            print("Please select a system first!")
            return
        
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
        for i, (name, _) in enumerate(query_images.items(), 1):
            print(f"{i}. {name}")
        print("8. Custom image")
        
        query_choice = input("\nChoose query image: ").strip()
        
        if query_choice == '8':
            filename = input("Enter custom image filename: ").strip()
            query_path = f"image.query/{filename}"
            if not os.path.exists(query_path):
                print(f"Error: File '{query_path}' not found!")
                return
        else:
            query_names = list(query_images.keys())
            try:
                query_idx = int(query_choice) - 1
                if 0 <= query_idx < len(query_names):
                    query_name = query_names[query_idx]
                    query_path = query_images[query_name]
                else:
                    print("Invalid choice!")
                    return
            except:
                print("Invalid choice!")
                return
        
        # Set different default thresholds based on system
        system_name = self.current_system.get_system_name()
        
        if "Deep Learning" in system_name:
            default_threshold = 0.01
            recommendation = f"\n[Recommended threshold for Deep Learning System: {default_threshold}]"
        else:
            default_threshold = 0.7
            recommendation = f"\n[Recommended threshold for Traditional System: {default_threshold}]"
        
        print(recommendation)
        threshold_input = input(f"Enter similarity threshold (0.0-1.0, default={default_threshold}): ").strip()
        
        if threshold_input == "":
            threshold = default_threshold
            print(f"Using default threshold: {threshold}")
        else:
            threshold = float(threshold_input)
        
        query_img = resize_if_large(cv.imread(query_path))
        
        print(f"\nRetrieving similar images using {self.current_system.get_system_name()}...")
        print(f"Threshold: {threshold}")
        
        results = self.current_system.retrieve_similar(query_img, threshold)
        
        if not results:
            print("No images retrieved! Try lowering the threshold.")
            return
        
        # Get expected category
        query_filename = os.path.basename(query_path).lower()
        category_mapping = get_query_category_mapping()
        expected_category = None
        for q_name, category in category_mapping.items():
            if q_name in query_filename:
                expected_category = category
                break
        
        if not expected_category:
            print("\nAvailable categories:")
            categories = ['african', 'beach', 'building', 'bus', 'dinosaur', 
                         'elephant', 'flower', 'horse', 'mountain', 'dish']
            for i, cat in enumerate(categories, 1):
                print(f"{i}. {cat}")
            cat_choice = input("Select expected category number: ").strip()
            try:
                expected_category = categories[int(cat_choice) - 1]
            except:
                print("Invalid category selection!")
                return
        
        # Save results
        output_dir = f"retrieved_{expected_category}_{os.path.basename(query_path).split('.')[0]}"
        save_retrieved_images(results, output_dir)
        
        # Calculate precision and recall
        precision, recall = calculate_precision_recall(results, expected_category)
        
        print(f"\n--- RETRIEVAL RESULTS ---")
        print(f"Total retrieved: {len(results)}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"Images saved to: {output_dir}")
        
        # Grading
        precision_marks = 20 if precision >= 0.6 else (5 if precision >= 0.4 else 0)
        recall_marks = 20 if recall >= 0.6 else (5 if recall >= 0.4 else 0)
        
        print(f"\nMarks for Precision: {precision_marks}/20")
        print(f"Marks for Recall: {recall_marks}/20")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        while True:
            if self.current_system is None:
                if not self.select_system():
                    break
                continue
            
            print("\n" + "=" * 60)
            print(f"CURRENT SYSTEM: {self.current_system.get_system_name()}")
            print("=" * 60)
            print("1. Test single query image")
            print("2. Test all 7 queries - Count correct matches")
            print("3. Retrieve similar images with threshold + Precision/Recall")
            print("4. Switch to different system")
            print("5. Exit")
            
            choice = input("\nChoose option (1-5): ").strip()
            
            if choice == '1':
                self.test_single_query()
            elif choice == '2':
                self.test_all_queries()
            elif choice == '3':
                self.retrieve_similar_images()
            elif choice == '4':
                self.current_system = None
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice, please try again.")

if __name__ == "__main__":
    app = ImageRetrievalApp()
    app.run()