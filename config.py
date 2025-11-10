"""
Configuration file for Contamination-Aware Waste Classification
SIMPLE & PROVEN CONFIG - Back to basics
"""

import os

# ==================== PATHS ====================
BASE_DIR = 'final_dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODEL_SAVE_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== MODEL SETTINGS ====================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

CATEGORIES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
NUM_CATEGORIES = len(CATEGORIES)

CATEGORY_MAPPING = {
    'cardboard_clean': 0,
    'cardboard_contaminated': 0,
    'glass_clean': 1,
    'glass_contaminated': 1,
    'metal_clean': 2,
    'metal_contaminated': 2,
    'organic': 3,
    'paper_clean': 4,
    'paper_contaminated': 4,
    'plastic_clean': 5,
    'plastic_contaminated': 5,
    'trash': 6
}

# ==================== TRAINING SETTINGS - SIMPLE ====================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001  # Standard rate, don't change

CATEGORY_LOSS_WEIGHT = 1.0
CONTAMINATION_LOSS_WEIGHT = 0.5

EARLY_STOPPING_PATIENCE = 8

# ==================== DATA AUGMENTATION - REDUCED ====================
TRAIN_AUGMENTATION = {
    'rotation_range': 10,  # REDUCED from 20
    'width_shift_range': 0.1,  # REDUCED from 0.2
    'height_shift_range': 0.1,  # REDUCED from 0.2
    'horizontal_flip': True,
    'zoom_range': 0.1,  # REDUCED from 0.2
    'fill_mode': 'nearest'
    # REMOVED brightness_range - can cause issues
}

VAL_AUGMENTATION = {}

# ==================== MODEL ARCHITECTURE - SIMPLE ====================
BASE_MODEL_NAME = 'MobileNetV2'  # CHANGED: Simpler, more stable than EfficientNet
FREEZE_BASE_MODEL = True  # Keep frozen
DENSE_UNITS = 128  # REDUCED from 256
DROPOUT_RATE = 0.3  # REDUCED from 0.5

VERBOSE = 1

print("✓ SIMPLIFIED Configuration loaded")
print(f"  Model: {BASE_MODEL_NAME} (simpler & stable)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Reduced augmentation for stability")