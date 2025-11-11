"""
Simple prediction demo
Test model on individual images
FIXED: Uses stable BASE model
"""

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import cv2
import os
from config import *

def load_model_for_prediction():
    """Load the trained model (BASE or FINE-TUNED)"""
    print("\nLoading model...")
    
    # Check which model exists
    base_path = 'models/best_model.weights.h5'
    finetuned_path = 'models/finetuned_model.weights.h5'
    
    # Use BASE model (more stable)
    if os.path.exists(base_path):
        weights_path = base_path
        model_type = "BASE"
    elif os.path.exists(finetuned_path):
        weights_path = finetuned_path
        model_type = "FINE-TUNED"
    else:
        raise FileNotFoundError("No trained model found! Run train_complete.py first.")
    
    print(f"✓ Using {model_type} model: {weights_path}")
    
    # Build architecture (SIMPLIFIED - matches training)
    input_layer = layers.Input(shape=INPUT_SHAPE)
    
    if BASE_MODEL_NAME == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    elif BASE_MODEL_NAME == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    else:
        raise ValueError(f"Unknown base model: {BASE_MODEL_NAME}")
    
    base_model.trainable = False
    
    x = base_model(input_layer, training=False)
    x = layers.GlobalAveragePooling2D(name='global_pooling')(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='shared_dense')(x)
    x = layers.Dropout(DROPOUT_RATE, name='shared_dropout')(x)
    
    # SIMPLIFIED outputs (no intermediate branches)
    category_output = layers.Dense(NUM_CATEGORIES, activation='softmax', name='category')(x)
    contamination_output = layers.Dense(1, activation='sigmoid', name='contamination')(x)
    
    model = models.Model(inputs=input_layer, outputs=[category_output, contamination_output])
    
    # Load weights
    try:
        model.load_weights(weights_path)
        print(f"✓ Model loaded successfully ({model_type})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    return model, model_type


def predict_image(model, image_path):
    """Predict single image"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    cat_pred, cont_pred = model.predict(img, verbose=0)
    
    # Parse results
    category_idx = np.argmax(cat_pred[0])
    category = CATEGORIES[category_idx]
    category_conf = float(cat_pred[0][category_idx] * 100)
    
    contamination_score = float(cont_pred[0][0])
    is_contaminated = contamination_score > 0.5
    cont_conf = contamination_score * 100 if is_contaminated else (1 - contamination_score) * 100
    
    return {
        'category': category,
        'category_confidence': category_conf,
        'is_contaminated': is_contaminated,
        'contamination_confidence': cont_conf,
        'contamination_score': contamination_score * 100,
        'all_categories': {cat: float(score*100) for cat, score in zip(CATEGORIES, cat_pred[0])}
    }


def print_prediction(result, image_path):
    """Pretty print prediction results"""
    print("\n" + "="*70)
    print(" PREDICTION RESULTS")
    print("="*70)
    print(f"\n📷 Image: {os.path.basename(image_path)}")
    print(f"   Path: {image_path}")
    
    print(f"\n📦 CATEGORY: {result['category'].upper()}")
    print(f"   Confidence: {result['category_confidence']:.1f}%")
    
    # Contamination status with color/emoji
    if result['is_contaminated']:
        print(f"\n⚠️  STATUS: CONTAMINATED")
        print(f"   Contamination Score: {result['contamination_score']:.1f}%")
        print(f"   Confidence: {result['contamination_confidence']:.1f}%")
        print("\n   ❌ NOT RECYCLABLE - Please clean before recycling")
        print("   ⚠️  May contaminate entire recycling batch")
    else:
        print(f"\n✅ STATUS: CLEAN")
        print(f"   Contamination Score: {result['contamination_score']:.1f}%")
        print(f"   Confidence: {result['contamination_confidence']:.1f}%")
        print("\n   ✓ RECYCLABLE - Ready for recycling")
    
    # Top 3 categories
    print("\n" + "-"*70)
    print(" Top 3 Category Predictions:")
    print("-"*70)
    sorted_cats = sorted(result['all_categories'].items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (cat, prob) in enumerate(sorted_cats, 1):
        bar = "█" * int(prob / 5)  # Simple bar chart
        print(f"  {i}. {cat:12s}: {prob:5.1f}% {bar}")
    
    # All categories (detailed)
    print("\n" + "-"*70)
    print(" All Category Probabilities:")
    print("-"*70)
    for cat, prob in sorted(result['all_categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:12s}: {prob:5.1f}%")
    
    print("="*70 + "\n")


def predict_from_path(model, image_path):
    """Predict and display results for a single image"""
    try:
        result = predict_image(model, image_path)
        print_prediction(result, image_path)
        return result
    except Exception as e:
        print(f"\n❌ Error processing {image_path}: {e}")
        return None


def interactive_mode(model):
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print(" INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("\nEnter image path (or 'quit' to exit)")
    print("Example: final_dataset/test/plastic_clean/image.jpg\n")
    
    while True:
        user_input = input("Image path: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        if os.path.exists(user_input):
            predict_from_path(model, user_input)
        else:
            print(f"❌ File not found: {user_input}\n")


def demo_mode(model):
    """Demo with sample images from test set"""
    print("\n" + "="*70)
    print(" DEMO MODE - Testing Random Images")
    print("="*70)
    
    # Find sample images
    sample_images = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                sample_images.append(os.path.join(root, file))
                if len(sample_images) >= 5:  # Get 5 samples
                    break
        if len(sample_images) >= 5:
            break
    
    if not sample_images:
        print("\n❌ No test images found!")
        return
    
    print(f"\nFound {len(sample_images)} sample images\n")
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n{'='*70}")
        print(f" SAMPLE {i}/{len(sample_images)}")
        print('='*70)
        predict_from_path(model, img_path)
        
        if i < len(sample_images):
            input("Press Enter for next image...")


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print(" WASTE CLASSIFICATION - PREDICTION DEMO")
    print("="*70)
    
    # Load model
    try:
        model, model_type = load_model_for_prediction()
        print(f"\n✓ Ready for predictions!")
        print(f"  Model: {model_type} {BASE_MODEL_NAME}")
        print(f"  Categories: {', '.join(CATEGORIES)}")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    # Choose mode
    print("\n" + "="*70)
    print(" SELECT MODE")
    print("="*70)
    print("\n1. Demo Mode    - Test on sample images from test set")
    print("2. Interactive  - Enter custom image paths")
    print("3. Single Image - Predict one specific image\n")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        demo_mode(model)
    elif choice == '2':
        interactive_mode(model)
    elif choice == '3':
        img_path = input("\nEnter image path: ").strip()
        if os.path.exists(img_path):
            predict_from_path(model, img_path)
        else:
            print(f"❌ File not found: {img_path}")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()