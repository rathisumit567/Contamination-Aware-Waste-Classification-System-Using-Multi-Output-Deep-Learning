"""
Fine-tuning script for contamination-aware waste classification
FIXED: Matches exact architecture from train_complete.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from config import *
from data_loader import create_data_generators

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*70)
print(" FINE-TUNING: CONTAMINATION-AWARE WASTE CLASSIFIER")
print("="*70)


class BestModelSaver(Callback):
    """Custom callback to save best model weights"""
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super().__init__()
        self.filepath = filepath.replace('.h5', '.weights.h5')
        self.monitor = monitor
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return
            
        if hasattr(current_loss, 'numpy'):
            current_loss = float(current_loss.numpy())
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current_loss:.5f}, saving weights...")


def load_base_model():
    """
    Load model with EXACT same architecture as train_complete.py
    """
    print("\n" + "="*60)
    print("LOADING BASE MODEL FOR FINE-TUNING")
    print("="*60)
    
    # Input layer
    input_layer = layers.Input(shape=INPUT_SHAPE, name='input_image')
    
    # Base model (feature extractor)
    if BASE_MODEL_NAME == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE
        )
    elif BASE_MODEL_NAME == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE
        )
    elif BASE_MODEL_NAME == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=INPUT_SHAPE
        )
    else:
        raise ValueError(f"Unknown base model: {BASE_MODEL_NAME}")
    
    # IMPORTANT: Keep base frozen initially to match trained model architecture!
    base_model.trainable = False
    
    print(f"✓ Base model: {BASE_MODEL_NAME}")
    print(f"  All layers FROZEN (matches training architecture)")
    print(f"  Total layers: {len(base_model.layers)}")
    
    # Connect input to base model
    x = base_model(input_layer, training=False)
    
    # IMPORTANT: SIMPLIFIED architecture - matches train_complete.py that gave 66% accuracy
    x = layers.GlobalAveragePooling2D(name='global_pooling')(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='shared_dense')(x)
    x = layers.Dropout(DROPOUT_RATE, name='shared_dropout')(x)
    
    # Direct outputs - NO intermediate branches (SIMPLIFIED version)
    category_output = layers.Dense(
        NUM_CATEGORIES, 
        activation='softmax', 
        name='category'
    )(x)
    
    contamination_output = layers.Dense(
        1, 
        activation='sigmoid', 
        name='contamination'
    )(x)
    
    # Create model
    model = models.Model(
        inputs=input_layer,
        outputs=[category_output, contamination_output],
        name='ContaminationAwareWasteClassifier'
    )
    
    print(f"\n✓ Model architecture recreated (SIMPLIFIED - matches your 66% model)")
    
    return model


def load_pretrained_weights(model, weights_path):
    """
    Load weights from previously trained model, then unfreeze for fine-tuning
    """
    print("\n" + "="*60)
    print("LOADING PRE-TRAINED WEIGHTS")
    print("="*60)
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        print("   Make sure you have trained the base model first!")
        print(f"   Expected location: {weights_path}")
        return False
    
    try:
        # Load weights with frozen architecture
        model.load_weights(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
        
        # NOW unfreeze top layers for fine-tuning
        base_model = model.layers[1]  # Get the base model layer
        base_model.trainable = True
        
        total_layers = len(base_model.layers)
        UNFREEZE_LAYERS = 30
        freeze_until = max(0, total_layers - UNFREEZE_LAYERS)
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        
        print(f"✓ Unfroze top {trainable_count} layers for fine-tuning")
        print("✓ Model ready for fine-tuning")
        return True
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure train_complete.py finished successfully")
        print("  2. Check that best_model.weights.h5 exists in models/ folder")
        print("  3. Verify config.py settings match your training run")
        print(f"\nCurrent config:")
        print(f"  BASE_MODEL_NAME: {BASE_MODEL_NAME}")
        print(f"  DENSE_UNITS: {DENSE_UNITS}")
        print(f"  DROPOUT_RATE: {DROPOUT_RATE}")
        return False


def compile_model_for_finetuning(model, learning_rate=5e-5):
    """
    Compile model with VERY LOW learning rate for fine-tuning
    """
    print("\n" + "="*60)
    print("COMPILING MODEL FOR FINE-TUNING")
    print("="*60)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss={
            'category': 'categorical_crossentropy',
            'contamination': 'binary_crossentropy'
        },
        loss_weights={
            'category': CATEGORY_LOSS_WEIGHT,
            'contamination': CONTAMINATION_LOSS_WEIGHT
        },
        metrics={
            'category': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')],
            'contamination': ['accuracy', tf.keras.metrics.AUC(name='auc')]
        }
    )
    
    print(f"✓ Model compiled for fine-tuning")
    print(f"  Learning rate: {learning_rate} (very low for stability)")
    print(f"  Optimizer: Adam")
    
    return model


def create_callbacks_finetuning():
    """
    Create callbacks for fine-tuning
    """
    print("\n" + "="*60)
    print("SETTING UP CALLBACKS")
    print("="*60)
    
    callbacks_list = [
        BestModelSaver(
            filepath=os.path.join(MODEL_SAVE_DIR, 'finetuned_model.h5'),
            monitor='val_loss',
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured for fine-tuning")
    print("  - Best model saver")
    print("  - Early stopping (patience=10)")
    print("  - Learning rate reduction")
    
    return callbacks_list


def plot_finetuning_history(history, save_path='results/finetuning_history.png'):
    """
    Plot fine-tuning history
    """
    print("\n" + "="*60)
    print("GENERATING FINE-TUNING PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Fine-Tuning History - Contamination-Aware Waste Classification', 
                 fontsize=16, fontweight='bold')
    
    # Category accuracy
    axes[0, 0].plot(history.history['category_accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_category_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Category Classification Accuracy', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # Contamination accuracy
    axes[0, 1].plot(history.history['contamination_accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_contamination_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Contamination Detection Accuracy', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    # Total loss
    axes[0, 2].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 2].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 2].set_title('Total Loss', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    
    # Category loss
    axes[1, 0].plot(history.history['category_loss'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_category_loss'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Category Loss', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    # Contamination loss
    axes[1, 1].plot(history.history['contamination_loss'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_contamination_loss'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Contamination Loss', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    # Contamination AUC
    if 'contamination_auc' in history.history:
        axes[1, 2].plot(history.history['contamination_auc'], label='Train', linewidth=2)
        axes[1, 2].plot(history.history['val_contamination_auc'], label='Validation', linewidth=2)
        axes[1, 2].set_title('Contamination AUC', fontweight='bold')
        axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Fine-tuning history saved: {save_path}")
    plt.close()


def evaluate_initial_performance(model, val_gen, steps_val):
    """
    Evaluate the loaded model before fine-tuning
    """
    print("\n" + "="*60)
    print("EVALUATING INITIAL PERFORMANCE")
    print("="*60)
    
    print("Running evaluation on validation set...")
    results = model.evaluate(val_gen, steps=steps_val, verbose=0)
    
    # Extract metrics (order: loss, cat_loss, cont_loss, cat_acc, cat_top2, cont_acc, cont_auc)
    metrics_dict = {
        'loss': results[0],
        'category_loss': results[1],
        'contamination_loss': results[2],
        'category_accuracy': results[3],
        'category_top_2_accuracy': results[4],
        'contamination_accuracy': results[5],
        'contamination_auc': results[6]
    }
    
    print("\n✓ Initial Performance (Before Fine-Tuning):")
    print(f"  Category Accuracy:      {metrics_dict['category_accuracy']*100:.2f}%")
    print(f"  Top-2 Accuracy:         {metrics_dict['category_top_2_accuracy']*100:.2f}%")
    print(f"  Contamination Accuracy: {metrics_dict['contamination_accuracy']*100:.2f}%")
    print(f"  Contamination AUC:      {metrics_dict['contamination_auc']:.4f}")
    print(f"  Total Loss:             {metrics_dict['loss']:.4f}")
    
    return metrics_dict


def print_comparison(initial, final):
    """
    Print before/after comparison
    """
    print("\n" + "="*70)
    print(" FINE-TUNING RESULTS COMPARISON")
    print("="*70)
    
    print("\n📊 BEFORE FINE-TUNING:")
    print(f"  Category Accuracy:      {initial['category_accuracy']*100:.2f}%")
    print(f"  Contamination Accuracy: {initial['contamination_accuracy']*100:.2f}%")
    print(f"  Contamination AUC:      {initial['contamination_auc']:.4f}")
    
    print("\n🚀 AFTER FINE-TUNING:")
    print(f"  Category Accuracy:      {final['category_accuracy']*100:.2f}%")
    print(f"  Contamination Accuracy: {final['contamination_accuracy']*100:.2f}%")
    print(f"  Contamination AUC:      {final['contamination_auc']:.4f}")
    
    print("\n📈 IMPROVEMENT:")
    cat_imp = (final['category_accuracy'] - initial['category_accuracy']) * 100
    cont_imp = (final['contamination_accuracy'] - initial['contamination_accuracy']) * 100
    auc_imp = (final['contamination_auc'] - initial['contamination_auc'])
    
    print(f"  Category:      {cat_imp:+.2f} percentage points")
    print(f"  Contamination: {cont_imp:+.2f} percentage points")
    print(f"  AUC:           {auc_imp:+.4f}")
    
    if cat_imp > 0 or cont_imp > 0:
        print("\n✅ Fine-tuning improved the model!")
    else:
        print("\n⚠️ No improvement - base model was already well-trained")
    
    print("="*70)


def main():
    """
    Main fine-tuning function
    """
    print("\n🚀 STARTING FINE-TUNING PIPELINE\n")
    
    # Step 1: Create data generators
    train_gen, val_gen, steps_train, steps_val = create_data_generators()
    
    # Step 2: Load model with exact architecture
    model = load_base_model()
    
    # Step 3: Load pre-trained weights
    weights_path = os.path.join(MODEL_SAVE_DIR, 'best_model.weights.h5')
    if not load_pretrained_weights(model, weights_path):
        print("\n❌ Cannot proceed without pre-trained weights!")
        print("   Run train_complete.py first to train the base model.")
        return
    
    # Step 4: Compile with very low learning rate
    model = compile_model_for_finetuning(model, learning_rate=5e-5)
    
    # Step 5: Evaluate initial performance
    initial_perf = evaluate_initial_performance(model, val_gen, steps_val)
    
    # Step 6: Setup callbacks
    callbacks = create_callbacks_finetuning()
    
    # Step 7: Fine-tune
    print("\n" + "="*70)
    print(" STARTING FINE-TUNING")
    print("="*70)
    print(f"\nFine-tuning for up to 25 epochs...")
    print(f"Learning rate: 0.00005 (very low)")
    print(f"Unfrozen: Top 30 layers")
    print("\n" + "-"*70 + "\n")
    
    FINETUNING_EPOCHS = 25
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=FINETUNING_EPOCHS,
        validation_data=val_gen,
        validation_steps=steps_val,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print(" FINE-TUNING COMPLETE! ✅")
    print("="*70)
    
    # Step 8: Plot results
    plot_finetuning_history(history)
    
    # Step 9: Get final performance
    final_perf = {
        'category_accuracy': history.history['val_category_accuracy'][-1],
        'contamination_accuracy': history.history['val_contamination_accuracy'][-1],
        'contamination_auc': history.history['val_contamination_auc'][-1]
    }
    
    # Step 10: Print comparison
    print_comparison(initial_perf, final_perf)
    
    # Step 11: Save final model architecture
    try:
        model_json = model.to_json()
        json_path = os.path.join(MODEL_SAVE_DIR, 'finetuned_architecture.json')
        with open(json_path, 'w') as f:
            f.write(model_json)
        print(f"\n✓ Model architecture saved: {json_path}")
    except:
        pass
    
    print("\n🎉 FINE-TUNING COMPLETE!\n")
    print("✓ Fine-tuned weights: models/finetuned_model.weights.h5")
    print("✓ Training plots: results/finetuning_history.png")
    print("\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Compare with base model")
    print("  3. Deploy best performing model\n")


if __name__ == "__main__":
    # Check GPU
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU detected")
    
    print(f"✓ TensorFlow version: {tf.__version__}")
    print("="*60)
    
    # Run fine-tuning
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Fine-tuning interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()