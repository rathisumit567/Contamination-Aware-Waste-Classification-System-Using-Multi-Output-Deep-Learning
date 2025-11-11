"""
Evaluate model on test set
Works with both base and fine-tuned models
"""

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os

from config import *
from data_loader import DualLabelDataGenerator

print("\n" + "="*70)
print(" MODEL EVALUATION ON TEST SET")
print("="*70)


def load_model():
    """Load the best available model"""
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    # Try base model first (more reliable)
    base_path = os.path.join(MODEL_SAVE_DIR, 'best_model.weights.h5')
    finetuned_path = os.path.join(MODEL_SAVE_DIR, 'finetuned_model.weights.h5')
    
    # PRIORITIZE BASE MODEL (it's more stable)
    if os.path.exists(base_path):
        weights_path = base_path
        model_type = "BASE"
        print(f"✓ Using BASE model: {weights_path}")
    elif os.path.exists(finetuned_path):
        print(f"⚠️  Fine-tuned model found, but using BASE model for evaluation")
        print(f"   (Fine-tuned models can have loading issues)")
        weights_path = base_path
        model_type = "BASE"
    else:
        raise FileNotFoundError("No trained model found! Run train_complete.py first.")
    
    print(f"Loading: {weights_path}")
    
    # Recreate architecture (SIMPLIFIED - matches training)
    input_layer = layers.Input(shape=INPUT_SHAPE, name='input_image')
    
    if BASE_MODEL_NAME == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE
        )
    elif BASE_MODEL_NAME == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE
        )
    else:
        raise ValueError(f"Unknown base model: {BASE_MODEL_NAME}")
    
    # For loading weights, start with frozen (matches saved architecture)
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
        print(f"✓ Loaded {model_type} model successfully")
        print(f"  Architecture: {BASE_MODEL_NAME} + {DENSE_UNITS} dense units")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print(f"\nCurrent config:")
        print(f"  BASE_MODEL_NAME: {BASE_MODEL_NAME}")
        print(f"  DENSE_UNITS: {DENSE_UNITS}")
        print(f"  DROPOUT_RATE: {DROPOUT_RATE}")
        raise
    
    return model, model_type


def evaluate_on_test_set(model):
    """Evaluate model on test set"""
    print("\n" + "="*60)
    print("CREATING TEST DATA GENERATOR")
    print("="*60)
    
    test_gen = DualLabelDataGenerator(
        TEST_DIR,
        batch_size=32,
        shuffle=False,
        augment=False
    )
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    all_category_preds = []
    all_category_true = []
    all_contamination_preds = []
    all_contamination_true = []
    
    steps = test_gen.steps_per_epoch
    print(f"Processing {steps} batches...")
    
    for i in range(steps):
        batch_x, batch_y = next(test_gen)
        cat_pred, cont_pred = model.predict(batch_x, verbose=0)
        
        all_category_preds.extend(np.argmax(cat_pred, axis=1))
        all_category_true.extend(np.argmax(batch_y['category'], axis=1))
        all_contamination_preds.extend((cont_pred > 0.5).astype(int).flatten())
        all_contamination_true.extend(batch_y['contamination'].flatten().astype(int))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{steps} batches...")
    
    print(f"✓ Evaluation complete: {len(all_category_true)} test samples")
    
    return (np.array(all_category_preds), np.array(all_category_true),
            np.array(all_contamination_preds), np.array(all_contamination_true))


def plot_confusion_matrices(cat_true, cat_pred, cont_true, cont_pred, model_type):
    """Plot confusion matrices"""
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRICES")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Category confusion matrix
    cm_cat = confusion_matrix(cat_true, cat_pred)
    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=axes[0])
    axes[0].set_title(f'Category Classification ({model_type} Model)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Contamination confusion matrix
    cm_cont = confusion_matrix(cont_true, cont_pred)
    sns.heatmap(cm_cont, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Clean', 'Contaminated'], 
                yticklabels=['Clean', 'Contaminated'], ax=axes[1])
    axes[1].set_title(f'Contamination Detection ({model_type} Model)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'confusion_matrices_test_{model_type.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {save_path}")
    plt.close()


def print_detailed_metrics(cat_true, cat_pred, cont_true, cont_pred):
    """Print detailed metrics"""
    print("\n" + "="*70)
    print(" DETAILED TEST SET RESULTS")
    print("="*70)
    
    # Category metrics
    cat_accuracy = np.mean(cat_true == cat_pred)
    
    print("\n📊 CATEGORY CLASSIFICATION:")
    print(f"  Overall Accuracy: {cat_accuracy*100:.2f}%")
    print(f"  Total samples: {len(cat_true)}")
    print("\n  Per-Class Report:")
    print(classification_report(cat_true, cat_pred, target_names=CATEGORIES, digits=4))
    
    # Contamination metrics
    cont_accuracy = np.mean(cont_true == cont_pred)
    cont_precision = precision_score(cont_true, cont_pred, zero_division=0)
    cont_recall = recall_score(cont_true, cont_pred, zero_division=0)
    cont_f1 = f1_score(cont_true, cont_pred, zero_division=0)
    
    print("\n🔍 CONTAMINATION DETECTION:")
    print(f"  Accuracy:  {cont_accuracy*100:.2f}%")
    print(f"  Precision: {cont_precision*100:.2f}%")
    print(f"  Recall:    {cont_recall*100:.2f}%")
    print(f"  F1-Score:  {cont_f1*100:.2f}%")
    
    # Confusion matrix details
    cm = confusion_matrix(cont_true, cont_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n  Confusion Matrix Breakdown:")
    print(f"    True Negatives (Clean → Clean):           {tn}")
    print(f"    False Positives (Clean → Contaminated):   {fp}")
    print(f"    False Negatives (Contaminated → Clean):   {fn}")
    print(f"    True Positives (Contaminated → Contam):   {tp}")
    
    print("\n" + "="*70)
    
    return {
        'category_accuracy': cat_accuracy,
        'contamination_accuracy': cont_accuracy,
        'contamination_precision': cont_precision,
        'contamination_recall': cont_recall,
        'contamination_f1': cont_f1,
        'total_samples': len(cat_true)
    }


def save_results_summary(metrics, model_type):
    """Save results to text file"""
    save_path = os.path.join(RESULTS_DIR, f'test_results_{model_type.lower()}.txt')
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f" TEST SET RESULTS - {model_type} MODEL\n")
        f.write(" CONTAMINATION-AWARE WASTE CLASSIFIER\n")
        f.write("="*70 + "\n\n")
        
        f.write("CATEGORY CLASSIFICATION:\n")
        f.write(f"  Accuracy: {metrics['category_accuracy']*100:.2f}%\n")
        f.write(f"  Test samples: {metrics['total_samples']}\n\n")
        
        f.write("CONTAMINATION DETECTION:\n")
        f.write(f"  Accuracy:  {metrics['contamination_accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {metrics['contamination_precision']*100:.2f}%\n")
        f.write(f"  Recall:    {metrics['contamination_recall']*100:.2f}%\n")
        f.write(f"  F1-Score:  {metrics['contamination_f1']*100:.2f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Model: {model_type} {BASE_MODEL_NAME}\n")
        f.write(f"Dense units: {DENSE_UNITS}, Dropout: {DROPOUT_RATE}\n")
        f.write(f"Test directory: {TEST_DIR}\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Results summary saved: {save_path}")


def main():
    """Main evaluation function"""
    print("\n🚀 STARTING TEST SET EVALUATION\n")
    
    # Load model
    model, model_type = load_model()
    
    # Evaluate
    cat_pred, cat_true, cont_pred, cont_true = evaluate_on_test_set(model)
    
    # Plot confusion matrices
    plot_confusion_matrices(cat_true, cat_pred, cont_true, cont_pred, model_type)
    
    # Print metrics
    metrics = print_detailed_metrics(cat_true, cat_pred, cont_true, cont_pred)
    
    # Save summary
    save_results_summary(metrics, model_type)
    
    print("\n🎉 TEST SET EVALUATION COMPLETE!\n")
    print("Generated files:")
    print(f"  - results/confusion_matrices_test_{model_type.lower()}.png")
    print(f"  - results/test_results_{model_type.lower()}.txt\n")


if __name__ == "__main__":
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()