import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Load the face dataset
face_dir = "human_faces"  # Update with the path to your dataset
face_files = os.listdir(face_dir)
X_face = []
for file in face_files:
    img = cv2.imread(os.path.join(face_dir, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # resize image to (64, 64)
    X_face.append(img)
y_face = [1] * len(X_face)

# Convert the data to numpy arrays and normalize pixel values
X_face = np.array(X_face) / 255.0
y_face = np.array(y_face)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_face, y_face, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Define the CNN model for face detection
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(datagen.flow(X_train.reshape(-1, 64, 64, 1), y_train, batch_size=32),
          validation_data=(X_test.reshape(-1, 64, 64, 1), y_test),
          epochs=10,
          callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test.reshape(-1, 64, 64, 1), y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("face_detection_model.h5")
