#!/usr/bin/env python3
"""
Bugs of Buffalo - Model Training Script
Train a deep learning model for cattle breed classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import argparse

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Create training and validation data generators with augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names

def create_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create the transfer learning model using EfficientNetB0.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_model(model, train_generator, validation_generator, epochs=50, callbacks=None):
    """
    Train the model with callbacks.
    """
    if callbacks is None:
        callbacks = []
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_generator, validation_generator, fine_tune_epochs=20):
    """
    Fine-tune the model by unfreezing some layers.
    """
    base_model = model.layers[1]
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    fine_tune_callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]
    
    print("\nStarting fine-tuning...")
    fine_tune_history = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        callbacks=fine_tune_callbacks,
        verbose=1
    )
    
    return fine_tune_history

def save_model_and_artifacts(model, class_names, train_generator, history, fine_tune_history=None):
    """
    Save the trained model and training artifacts.
    """
    save_dir = '../saved_model'
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = os.path.join(save_dir, 'bugs_of_buffalo_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    class_mapping = {i: class_name for i, class_name in enumerate(class_names)}
    mapping_path = os.path.join(save_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")
    
    history_path = os.path.join(save_dir, f'training_history_{timestamp}.json')
    history_dict = {
        'training_history': history.history,
        'class_names': class_names,
        'timestamp': timestamp
    }
    
    if fine_tune_history:
        history_dict['fine_tune_history'] = fine_tune_history.history
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return model_path, mapping_path

def plot_training_history(history, fine_tune_history=None):
    """
    Plot training history.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if fine_tune_history:
        ax1.plot(range(len(history.history['accuracy']), 
                     len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])),
                 fine_tune_history.history['accuracy'], label='Fine-tuning Accuracy')
        ax1.plot(range(len(history.history['val_accuracy']), 
                     len(history.history['val_accuracy']) + len(fine_tune_history.history['val_accuracy'])),
                 fine_tune_history.history['val_accuracy'], label='Fine-tuning Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    if fine_tune_history:
        ax2.plot(range(len(history.history['loss']), 
                     len(history.history['loss']) + len(fine_tune_history.history['loss'])),
                 fine_tune_history.history['loss'], label='Fine-tuning Loss')
        ax2.plot(range(len(history.history['val_loss']), 
                     len(history.history['val_loss']) + len(fine_tune_history.history['val_loss'])),
                 fine_tune_history.history['val_loss'], label='Fine-tuning Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    ax3.plot(history.history['precision'], label='Training Precision')
    ax3.plot(history.history['val_precision'], label='Validation Precision')
    ax3.set_title('Model Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    
    ax4.plot(history.history['recall'], label='Training Recall')
    ax4.plot(history.history['val_recall'], label='Validation Recall')
    ax4.set_title('Model Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../saved_model/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train cattle breed classification model')
    parser.add_argument('--data_dir', type=str, default='../data/train',
                       help='Path to training data directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("🐃 Bugs of Buffalo - Model Training")
    print("=" * 50)
    
    print("Creating data generators...")
    train_gen, val_gen, class_names = create_data_generators(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    print("Creating model...")
    model = create_model(
        input_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=len(class_names)
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            '../saved_model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    print("Starting initial training...")
    history = train_model(
        model=model,
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print("Starting fine-tuning...")
    fine_tune_history = fine_tune_model(
        model=model,
        train_generator=train_gen,
        validation_generator=val_gen,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    print("Saving model and artifacts...")
    model_path, mapping_path = save_model_and_artifacts(
        model=model,
        class_names=class_names,
        train_generator=train_gen,
        history=history,
        fine_tune_history=fine_tune_history
    )
    
    print("Plotting training history...")
    plot_training_history(history, fine_tune_history)
    
    final_loss, final_accuracy, final_precision, final_recall = model.evaluate(val_gen, verbose=0)
    print(f"Validation Loss: {final_loss:.4f}")
    print(f"Validation Accuracy: {final_accuracy:.4f}")
    print(f"Validation Precision: {final_precision:.4f}")
    print(f"Validation Recall: {final_recall:.4f}")
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Class mapping saved to: {mapping_path}")

if __name__ == "__main__":
    main()
