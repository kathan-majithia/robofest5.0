import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def create_gesture_model(input_shape=(128, 128, 3), num_classes=6):
    """Create a lightweight CNN model for Raspberry Pi"""
    
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block  
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(class_idx)
    
    return np.array(images), np.array(labels), class_names

def train_model():
    # Load data
    print("Loading dataset...")
    X, y, class_names = load_and_preprocess_data('dataset')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_gesture_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        batch_size=32
    )
    
    # Save model
    model.save('gesture_model.h5')
    print("Model saved as 'gesture_model.h5'")
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, class_names

if __name__ == "__main__":
    train_model()