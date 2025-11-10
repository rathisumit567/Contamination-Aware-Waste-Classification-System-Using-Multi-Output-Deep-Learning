"""
Complete training script for contamination-aware waste classification
FIXED: JSON serialization error resolved - Use save_weights_only=True
Compatible with TensorFlow 2.10+ and GPU support
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*70)
print(" CONTAMINATION-AWARE WASTE CLASSIFICATION SYSTEM")
print("="*70)


def create_model():
    """
    Create multi-output model - SIMPLIFIED VERSION
    """
    print("\n" + "="*60)
    print("BUILDING MODEL - SIMPLIFIED")
    print("="*60)
    
    # Input layer
    input_layer = layers.Input(shape=INPUT_SHAPE, name='input_image')
    
    # Base model - SIMPLE & STABLE
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
    
    # Keep base model FROZEN - it works!
    base_model.trainable = False
    
    print(f"✓ Base model: {BASE_MODEL_NAME}")
    print(f"  All layers FROZEN (using pretrained features)")
    print(f"  Total layers: {len(base_model.layers)}")
    
    # Connect input to base model
    x = base_model(input_layer, training=False)
    
    # Simple top layers - NO batch norm, NO L2 reg
    x = layers.GlobalAveragePooling2D(name='global_pooling')(x)
    x = layers.Dense(DENSE_UNITS, activation='relu', name='shared_dense')(x)
    x = layers.Dropout(DROPOUT_RATE, name='shared_dropout')(x)
    
    # Category branch - SIMPLE
    category_output = layers.Dense(
        NUM_CATEGORIES, 
        activation='softmax', 
        name='category'
    )(x)
    
    # Contamination branch - SIMPLE
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
    
    print(f"\n✓ SIMPLE model architecture created")
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Output 1 (category): {NUM_CATEGORIES} classes")
    print(f"  Output 2 (contamination): binary")
    
    return model

def compile_model(model):
    """
    Compile model with appropriate losses and metrics
    """
    print("\n" + "="*60)
    print("COMPILING MODEL")
    print("="*60)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
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
    
    print("✓ Model compiled successfully")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Category loss weight: {CATEGORY_LOSS_WEIGHT}")
    print(f"  Contamination loss weight: {CONTAMINATION_LOSS_WEIGHT}")
    
    return model


class BestModelSaver(Callback):
    """
    Custom callback to save best model WITHOUT JSON serialization issues
    Saves only weights (not the full model) to avoid EagerTensor serialization bug
    """
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super().__init__()
        # Change .h5 to .weights.h5 for clarity
        self.filepath = filepath.replace('.h5', '.weights.h5')
        self.monitor = monitor
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return
            
        # Convert to float if it's a tensor
        if hasattr(current_loss, 'numpy'):
            current_loss = float(current_loss.numpy())
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            # Save weights immediately (weights-only saves don't have JSON issues)
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current_loss:.5f}, saving weights...")
    
    def on_train_end(self, logs=None):
        if self.verbose:
            print(f"\n✓ Best model weights saved to: {self.filepath}")
            print(f"  Best epoch: {self.best_epoch}, Best {self.monitor}: {self.best_loss:.5f}")


def create_callbacks():
    """
    Create training callbacks
    FIXED: Using custom BestModelSaver instead of ModelCheckpoint
    """
    print("\n" + "="*60)
    print("SETTING UP CALLBACKS")
    print("="*60)
    
    callbacks_list = [
        # Custom model saver (avoids JSON serialization)
        BestModelSaver(
            filepath=os.path.join(MODEL_SAVE_DIR, 'best_model.h5'),
            monitor='val_loss',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,  # Our custom saver handles this
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured:")
    print("  - Custom model saver (best model)")
    print(f"  - Early stopping (patience={EARLY_STOPPING_PATIENCE})")
    print("  - Learning rate reduction")
    
    return callbacks_list


def plot_training_history(history):
    """
    Plot and save training history
    """
    print("\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - Contamination-Aware Waste Classification', fontsize=16, fontweight='bold')
    
    # Category accuracy
    axes[0, 0].plot(history.history['category_accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_category_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Category Classification Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Contamination accuracy
    axes[0, 1].plot(history.history['contamination_accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_contamination_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Contamination Detection Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total loss
    axes[0, 2].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 2].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 2].set_title('Total Loss', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Category loss
    axes[1, 0].plot(history.history['category_loss'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_category_loss'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Category Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Contamination loss
    axes[1, 1].plot(history.history['contamination_loss'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_contamination_loss'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Contamination Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Contamination AUC
    if 'contamination_auc' in history.history:
        axes[1, 2].plot(history.history['contamination_auc'], label='Train', linewidth=2)
        axes[1, 2].plot(history.history['val_contamination_auc'], label='Validation', linewidth=2)
        axes[1, 2].set_title('Contamination AUC', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved: {save_path}")
    plt.close()


def print_training_summary(history):
    """
    Print final training statistics
    """
    print("\n" + "="*70)
    print(" TRAINING SUMMARY")
    print("="*70)
    
    final_epoch = len(history.history['loss'])
    
    print(f"\nTotal Epochs Trained: {final_epoch}")
    print(f"\nFinal Metrics:")
    print(f"  Category Accuracy (Train):      {history.history['category_accuracy'][-1]:.4f}")
    print(f"  Category Accuracy (Val):        {history.history['val_category_accuracy'][-1]:.4f}")
    print(f"  Contamination Accuracy (Train): {history.history['contamination_accuracy'][-1]:.4f}")
    print(f"  Contamination Accuracy (Val):   {history.history['val_contamination_accuracy'][-1]:.4f}")
    
    # Find best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\nBest Epoch: {best_epoch}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Category Accuracy: {history.history['val_category_accuracy'][best_epoch-1]:.4f}")
    print(f"  Contamination Accuracy: {history.history['val_contamination_accuracy'][best_epoch-1]:.4f}")
    
    print(f"\n✓ Best model saved to: {os.path.join(MODEL_SAVE_DIR, 'best_model.h5')}")
    print("="*70)


def main():
    """
    Main training function
    """
    print("\n🚀 STARTING TRAINING PIPELINE\n")
    
    # Step 1: Create data generators
    train_gen, val_gen, steps_train, steps_val = create_data_generators()
    
    # Step 2: Build model
    model = create_model()
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    # Step 3: Compile model
    model = compile_model(model)
    
    # Step 4: Setup callbacks
    callbacks = create_callbacks()
    
    # Step 5: Train model
    print("\n" + "="*70)
    print(" STARTING TRAINING")
    print("="*70)
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_train} (train), {steps_val} (val)")
    print("\n" + "-"*70 + "\n")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=steps_val,
        callbacks=callbacks,
        verbose=VERBOSE
    )
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE! ✅")
    print("="*70)
    
    # Step 6: Plot results
    plot_training_history(history)
    
    # Step 7: Print summary
    print_training_summary(history)
    
    print("\n🎉 ALL DONE! Your model is ready to use!\n")
    print("Next steps:")
    print("  1. Run evaluation script to test on test set")
    print("  2. Use the model for predictions")
    print("  3. Deploy in your application\n")


if __name__ == "__main__":
    # Check GPU availability
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU detected - training will use CPU (slower)")
    
    print(f"✓ TensorFlow version: {tf.__version__}")
    print("="*60)
    
    # Run main training
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial progress has been saved in checkpoints")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()