import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
 
print("Libraries imported successfully!")
dataset_dir = r'C:\Users\USER\Desktop\Deepfake Detection\FF++'

print("Dataset directory set to:", dataset_dir)
def load_video_frames(dataset_dir, frame_skip=30):
    images = []
    labels = []
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")  # Debugging output
            
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp4'):  # Process only video files
                    video_path = os.path.join(folder_path, filename)
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        print(f"Error opening video: {video_path}")
                        continue
                    
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break  # Stop when the video ends
                        
                        # Process every 'frame_skip' frame to reduce redundancy
                        if frame_count % frame_skip == 0:
                            # Convert to grayscale for face detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            # Detect faces
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            # If faces are found, process them
                            if len(faces) > 0:
                                # Process the largest face
                                x, y, w, h = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
                                
                                # Add some margin
                                margin = int(w * 0.2)
                                x = max(0, x - margin)
                                y = max(0, y - margin)
                                w = min(frame.shape[1] - x, w + 2 * margin)
                                h = min(frame.shape[0] - y, h + 2 * margin)
                                
                                # Extract face ROI
                                face_roi = frame[y:y+h, x:x+w]
                                
                                # Resize face to standard size
                                face_roi = cv2.resize(face_roi, (224, 224))
                                
                                # Apply data augmentation for fake samples to increase dataset diversity
                                if 'fake' in folder:
                                    # Original face
                                    images.append(face_roi)
                                    labels.append(1)  # 1 for fake
                                    
                                    # Slightly rotated face (5 degrees)
                                    M = cv2.getRotationMatrix2D((face_roi.shape[1] / 2, face_roi.shape[0] / 2), 5, 1)
                                    rotated = cv2.warpAffine(face_roi, M, (face_roi.shape[1], face_roi.shape[0]))
                                    images.append(rotated)
                                    labels.append(1)
                                else:
                                    images.append(face_roi)
                                    labels.append(0)  # 0 for real
                        
                        frame_count += 1
                    
                    cap.release()
    
    print(f"Total frames extracted: {len(images)}")  # Debugging output
    return np.array(images), np.array(labels)

print("Function to extract video frames with face detection is ready!")
images, labels = load_video_frames(dataset_dir)

# Check if images are loaded correctly
print(f"Images shape: {images.shape}")  # Should be (num_samples, 224, 224, 3)
print(f"Labels shape: {labels.shape}")  # Should be (num_samples,)
images = images / 255.0  # Scale pixel values to [0, 1]
print("Images normalized successfully!")
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Check dataset shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
def build_improved_model():
    # Use a pre-trained model (EfficientNetB0) as the base
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Create the model
model = build_improved_model()
print("Improved CNN model created successfully!")

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]
print("Training callbacks defined!")
history = model.fit(
    X_train, y_train,
    epochs=20,  
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
print("Model training completed!")
results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
print(f"Test Precision: {results[2]:.4f}")
print(f"Test Recall: {results[3]:.4f}")
print(f"Test AUC: {results[4]:.4f}")
model.save('improved_deepfake_model.keras')
print("Model saved as 'improved_deepfake_model.keras'")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
model.save('saveModel.keras')
