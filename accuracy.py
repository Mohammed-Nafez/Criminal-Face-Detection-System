import os
import cv2
import numpy as np
import re
import tensorflow as tf

# Load your model (with the updated input layer)
model = tf.keras.models.load_model('face_recognition_model.h5')

# Define the desired input shape for your model
img_width, img_height = 64, 64

# Directory containing your test images
test_dir = "human_faces"

accurate_predictions = 0
total_images = 0

for filename in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, filename))

    if img is not None:
        img = cv2.resize(img, (img_width, img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img / 255.0  # Normalize the pixel values to the range [0, 1]

        # Extract the label from the filename using regular expressions
        label_match = re.search(r'(\d+)', filename)
        if label_match:
            ground_truth_label = int(label_match.group())
        else:
            continue  # Skip files without valid labels

        # Expand dimensions to match the expected input shape (1 channel)
        img = np.expand_dims(img, axis=-1)
        
        # Get the predicted label from your model
        prediction = model.predict(np.expand_dims(img, axis=0))
        predicted_label = 1 if prediction > 0.5 else 0

        if predicted_label == ground_truth_label:
            accurate_predictions += 1

        total_images += 1

accuracy = accurate_predictions / total_images

print(f"Accuracy: {accuracy:.2f}")
