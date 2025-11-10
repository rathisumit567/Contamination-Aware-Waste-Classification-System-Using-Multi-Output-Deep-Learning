"""
Helper script to load the trained contamination-aware waste classification model
Use this after training completes
"""

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os

def load_trained_model(weights_path='models/best_model.weights.h5', 
                       architecture_path='models/model_architecture.json'):
    """
    Load the trained model from saved weights and architecture
    
    Args:
        weights_path: Path to saved weights file (.weights.h5)
        architecture_path: Path to saved architecture JSON file
    
    Returns:
        Loaded and compiled model ready for predictions
    """
    
    print("="*60)
    print("LOADING TRAINED MODEL")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    if not os.path.exists(architecture_path):
        raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
    
    # Load model architecture from JSON
    print(f"\n✓ Loading architecture from: {architecture_path}")
    with open(architecture_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    # Load weights
    print(f"✓ Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    
    # Compile the model (needed for predictions)
    print("✓ Compiling model...")
    model.compile(
        optimizer='adam',  # Optimizer doesn't matter for inference
        loss={
            'category': 'categorical_crossentropy',
            'contamination': 'binary_crossentropy'
        },
        metrics={
            'category': ['accuracy'],
            'contamination': ['accuracy']
        }
    )
    
    print("\n✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Outputs: {len(model.outputs)}")
    print("="*60)
    
    return model


def predict_single_image(model, image_path, categories):
    """
    Make prediction on a single image
    
    Args:
        model: Loaded model
        image_path: Path to image file
        categories: List of category names
    
    Returns:
        Dictionary with predictions
    """
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    category_pred, contamination_pred = model.predict(img_array, verbose=0)
    
    # Get results
    category_idx = np.argmax(category_pred[0])
    category_confidence = float(category_pred[0][category_idx])
    contamination_score = float(contamination_pred[0][0])
    
    is_contaminated = contamination_score > 0.5
    
    return {
        'category': categories[category_idx],
        'category_confidence': category_confidence,
        'is_contaminated': is_contaminated,
        'contamination_confidence': contamination_score if is_contaminated else (1 - contamination_score),
        'all_category_scores': {cat: float(score) for cat, score in zip(categories, category_pred[0])}
    }


# Example usage
if __name__ == "__main__":
    # Define categories (must match training)
    CATEGORIES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
    
    # Load model
    model = load_trained_model()
    
    # Example prediction (replace with your image path)
    image_path = 'test_image.jpg'
    
    if os.path.exists(image_path):
        print(f"\n{'='*60}")
        print("MAKING PREDICTION")
        print('='*60)
        print(f"Image: {image_path}\n")
        
        result = predict_single_image(model, image_path, CATEGORIES)
        
        print(f"Category: {result['category']} ({result['category_confidence']*100:.1f}% confident)")
        print(f"Contaminated: {'YES' if result['is_contaminated'] else 'NO'} ({result['contamination_confidence']*100:.1f}% confident)")
        
        print(f"\nAll category scores:")
        for cat, score in sorted(result['all_category_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {score*100:.1f}%")
    else:
        print(f"\n⚠️  Example image not found: {image_path}")
        print("   Replace with your own image path to test predictions")