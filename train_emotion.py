import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

def create_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Create a transfer learning model using MobileNetV2 base.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), # Lower learning rate for fine-tuning/transfer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    """
    Train the model.
    """
    # Parameters
    IMG_HEIGHT, IMG_WIDTH = 224, 224 # Upgraded from 48x48
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Train directory '{TRAIN_DIR}' not found.")
        return

    # Advanced Data Generators (Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print(f"Loading data from {TRAIN_DIR}...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )
    
    val_generator = None
    if VAL_DIR and os.path.exists(VAL_DIR):
        print(f"Loading validation data from {VAL_DIR}...")
        val_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb'
        )
    
    num_classes = train_generator.num_classes
    print(f"Detected {num_classes} classes: {list(train_generator.class_indices.keys())}")
    
    # Create Model
    model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('custom_emotion_model_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    
    # Train
    validation_steps = None
    if val_generator:
        validation_steps = val_generator.samples // BATCH_SIZE
        if validation_steps == 0: validation_steps = 1
        
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    
    # Save Final Model
    save_path = 'custom_emotion_model.h5'
    model.save(save_path)
    print(f"Final Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom emotion detection model.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory (with class subfolders)')
    parser.add_argument('--val_dir', type=str, help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    main(args)

