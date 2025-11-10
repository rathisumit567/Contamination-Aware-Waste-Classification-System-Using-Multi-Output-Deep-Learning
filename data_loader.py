"""
Data loader for contamination-aware waste classification
Handles dual labels: category + contamination status
FIXED VERSION - No errors
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from config import *

class DualLabelDataGenerator:
    """
    Custom data generator that provides both category and contamination labels
    """
    
    def __init__(self, directory, batch_size=32, shuffle=True, augment=True):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Setup image data generator
        if augment:
            self.datagen = ImageDataGenerator(
                rescale=1./255,
                **TRAIN_AUGMENTATION
            )
        else:
            self.datagen = ImageDataGenerator(rescale=1./255)
        
        # Create base generator
        self.base_generator = self.datagen.flow_from_directory(
            directory,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=shuffle
        )
        
        # Store class information
        self.class_indices = self.base_generator.class_indices
        self.folder_names = list(self.class_indices.keys())
        
        # Calculate steps per epoch
        self.steps_per_epoch = len(self.base_generator)
        
        print(f"\n✓ Data generator created for: {directory}")
        print(f"  Found {self.base_generator.samples} images")
        print(f"  Folders: {len(self.folder_names)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get batch from base generator
        batch_x, batch_y_categorical = next(self.base_generator)
        
        batch_size_actual = batch_x.shape[0]
        
        # Initialize labels
        category_labels = np.zeros((batch_size_actual, NUM_CATEGORIES))
        contamination_labels = np.zeros((batch_size_actual, 1))
        
        # Process each image in batch
        for i in range(batch_size_actual):
            # Get folder index from one-hot encoding
            folder_idx = np.argmax(batch_y_categorical[i])
            folder_name = self.folder_names[folder_idx]
            
            # Map to category
            if folder_name in CATEGORY_MAPPING:
                category_idx = CATEGORY_MAPPING[folder_name]
                category_labels[i, category_idx] = 1
            else:
                # Fallback: try to match base name
                for mapped_name, cat_idx in CATEGORY_MAPPING.items():
                    if mapped_name.startswith(folder_name) or folder_name.startswith(mapped_name):
                        category_labels[i, cat_idx] = 1
                        break
            
            # Determine contamination status
            if 'contaminated' in folder_name.lower():
                contamination_labels[i, 0] = 1.0  # Contaminated
            else:
                contamination_labels[i, 0] = 0.0  # Clean
        
        # Return in format expected by multi-output model
        return batch_x, {
            'category': category_labels,
            'contamination': contamination_labels
        }


def create_data_generators():
    """
    Create train and validation data generators
    Returns: train_gen, val_gen, steps_per_epoch_train, steps_per_epoch_val
    """
    
    print("\n" + "="*60)
    print("CREATING DATA GENERATORS")
    print("="*60)
    
    # Training generator (with augmentation)
    train_gen = DualLabelDataGenerator(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )
    
    # Validation generator (without augmentation)
    val_gen = DualLabelDataGenerator(
        VAL_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    print("\n" + "="*60)
    print("DATA GENERATORS READY")
    print("="*60)
    
    return train_gen, val_gen, train_gen.steps_per_epoch, val_gen.steps_per_epoch


def verify_data_generator():
    """
    Test function to verify data generator works correctly
    """
    print("\n" + "="*60)
    print("VERIFYING DATA GENERATOR")
    print("="*60)
    
    # Create a test generator
    test_gen = DualLabelDataGenerator(TRAIN_DIR, batch_size=4, shuffle=False, augment=False)
    
    # Get one batch
    batch_x, batch_y = next(test_gen)
    
    print(f"\n✓ Batch loaded successfully!")
    print(f"  Images shape: {batch_x.shape}")
    print(f"  Category labels shape: {batch_y['category'].shape}")
    print(f"  Contamination labels shape: {batch_y['contamination'].shape}")
    
    print(f"\n  Sample category labels:\n{batch_y['category']}")
    print(f"\n  Sample contamination labels:\n{batch_y['contamination'].flatten()}")
    
    # Verify values
    assert batch_x.shape[1:] == INPUT_SHAPE, "Image shape mismatch"
    assert batch_y['category'].shape[1] == NUM_CATEGORIES, "Category labels mismatch"
    assert batch_y['contamination'].shape[1] == 1, "Contamination labels mismatch"
    
    print("\n✅ Data generator verification PASSED!")
    print("="*60)


if __name__ == "__main__":
    # Test the data generator
    verify_data_generator()