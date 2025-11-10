import cv2
import numpy as np
import os
from pathlib import Path
import shutil

def add_dirt_spots(image, num_spots=15, intensity=0.6):
    """Add dirty spots to simulate food residue/stains"""
    contaminated = image.copy()
    height, width = image.shape[:2]
    
    for _ in range(num_spots):
        # Random position and size
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(10, 40)
        
        # Brown/yellow/gray colors for contamination
        color_options = [
            (50, 70, 30),    # Dark brown (food residue)
            (40, 90, 100),   # Yellow-brown (grease)
            (60, 60, 60),    # Gray (dirt)
            (30, 50, 80)     # Orange-brown (rust/stains)
        ]
        color = color_options[np.random.randint(0, len(color_options))]
        
        # Create soft edges with Gaussian blur
        overlay = contaminated.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        contaminated = cv2.addWeighted(contaminated, 1-intensity*0.3, overlay, intensity*0.3, 0)
    
    return contaminated

def add_texture_noise(image, intensity=0.3):
    """Add grainy texture to simulate dirt"""
    noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
    noisy = np.clip(image.astype(np.int16) + noise * intensity, 0, 255).astype(np.uint8)
    return noisy

def add_liquid_stain(image, intensity=0.5):
    """Simulate liquid spills"""
    contaminated = image.copy()
    height, width = image.shape[:2]
    
    # Create irregular stain shape
    center_x = np.random.randint(width//4, 3*width//4)
    center_y = np.random.randint(height//4, 3*height//4)
    
    # Create mask for irregular stain
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(5):
        x = center_x + np.random.randint(-50, 50)
        y = center_y + np.random.randint(-50, 50)
        radius = np.random.randint(30, 80)
        cv2.circle(mask, (x, y), radius, 255, -1)
    
    # Blur for realistic edges
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Apply brownish/yellowish tint
    stain_color = np.array([40, 80, 90], dtype=np.uint8)
    stain = np.zeros_like(contaminated)
    stain[:] = stain_color
    
    # Blend
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    contaminated = (contaminated * (1 - mask_3channel * intensity) + 
                   stain * mask_3channel * intensity).astype(np.uint8)
    
    return contaminated

def create_contaminated_version(image_path, contamination_level='moderate'):
    """Apply multiple contamination effects"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    contaminated = image.copy()
    
    if contamination_level == 'light':
        contaminated = add_dirt_spots(contaminated, num_spots=5, intensity=0.3)
        contaminated = add_texture_noise(contaminated, intensity=0.2)
    
    elif contamination_level == 'moderate':
        contaminated = add_dirt_spots(contaminated, num_spots=12, intensity=0.5)
        contaminated = add_liquid_stain(contaminated, intensity=0.4)
        contaminated = add_texture_noise(contaminated, intensity=0.3)
    
    elif contamination_level == 'heavy':
        contaminated = add_dirt_spots(contaminated, num_spots=20, intensity=0.7)
        contaminated = add_liquid_stain(contaminated, intensity=0.6)
        contaminated = add_texture_noise(contaminated, intensity=0.4)
        # Add extra large stain
        contaminated = add_liquid_stain(contaminated, intensity=0.5)
    
    return contaminated

def process_dataset(base_dir='final_dataset'):
    """
    Process entire dataset:
    1. Rename existing folders to *_clean
    2. Create contaminated versions
    """
    
    categories_to_process = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_dir, split)
        
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} set...")
        print(f"{'='*50}")
        
        for category in categories_to_process:
            original_folder = os.path.join(split_path, category)
            clean_folder = os.path.join(split_path, f'{category}_clean')
            contaminated_folder = os.path.join(split_path, f'{category}_contaminated')
            
            if not os.path.exists(original_folder):
                print(f"⚠️  Skipping {category} - folder not found")
                continue
            
            # Step 1: Rename original folder to *_clean
            if os.path.exists(clean_folder):
                print(f"✓ {category}_clean already exists")
            else:
                os.rename(original_folder, clean_folder)
                print(f"✓ Renamed {category} → {category}_clean")
            
            # Step 2: Create contaminated folder
            os.makedirs(contaminated_folder, exist_ok=True)
            
            # Step 3: Process images
            clean_images = [f for f in os.listdir(clean_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  Creating contaminated versions for {len(clean_images)} images...")
            
            for i, img_name in enumerate(clean_images):
                clean_path = os.path.join(clean_folder, img_name)
                
                # Randomly choose contamination level
                level = np.random.choice(['light', 'moderate', 'heavy'], 
                                        p=[0.3, 0.5, 0.2])
                
                contaminated_img = create_contaminated_version(clean_path, level)
                
                if contaminated_img is not None:
                    # Save with prefix indicating contamination level
                    new_name = f'cont_{level}_{img_name}'
                    save_path = os.path.join(contaminated_folder, new_name)
                    cv2.imwrite(save_path, contaminated_img)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(clean_images)} images processed")
            
            print(f"✓ {category}: {len(clean_images)} contaminated images created")
    
    print("\n" + "="*50)
    print("✅ CONTAMINATION GENERATION COMPLETE!")
    print("="*50)
    
    # Print final summary
    print("\nFINAL DATASET STRUCTURE:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        split_path = os.path.join(base_dir, split)
        for folder in sorted(os.listdir(split_path)):
            folder_path = os.path.join(split_path, folder)
            if os.path.isdir(folder_path):
                count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {folder}: {count} images")

# Run the script
if __name__ == "__main__":
    print("🚀 Starting contamination generation...")
    print("This will:")
    print("1. Rename your existing folders to *_clean")
    print("2. Create *_contaminated folders with synthetic contamination")
    print("\nPress Ctrl+C within 5 seconds to cancel...")
    
    import time
    time.sleep(5)
    
    process_dataset('final_dataset')
    
    print("\n✅ Done! Your dataset now has contamination detection capability.")