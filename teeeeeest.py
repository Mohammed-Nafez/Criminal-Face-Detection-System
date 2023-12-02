import os
import cv2
import numpy as np
import tensorflow as tf
import face_recognition

class FaceCounter:
    def __init__(self):
        self.load_models()

    def load_models(self):
        self.face_recognition_model = tf.keras.models.load_model('face_recognition_model.h5')

    def preprocess_face(self, face_image):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (128, 128))
        face_image = np.reshape(face_image, (128, 128, 1))
        face_image = face_image / 255.0
        return face_image

    def predict_face(self, face_encoding):
        processed_encoding = self.preprocess_face(face_encoding)
        predictions = self.face_recognition_model.predict(np.array([processed_encoding]))
        return predictions

if __name__ == '__main':
    fc = FaceCounter()
    dataset_folder = r'C:\Users\Mohammd Nafez Aloul\PycharmProjects\pythonProject2'  # Replace with the path to your dataset folder

    true_labels = []  # Ground truth labels (1 for faces, 0 for non-faces)
    predictions = []  # Model's predictions (1 for faces, 0 for non-faces)

    # Specify the paths to "human-faces" and "non-faces" folders
    human_faces_folder = os.path.join(dataset_folder, 'human_faces')
    non_faces_folder = os.path.join(dataset_folder, 'Non-faces')

    for class_name, folder_path in [('human-faces', human_faces_folder), ('Non-faces', non_faces_folder)]:
        is_face = 1 if class_name == 'human-faces' else 0
        print(f"Class: {class_name}, Label: {is_face}")

        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            print(f"Processing image: {image_path}")

            # Process the image with your face recognition model
            predictions = fc.predict_face(image)

            true_labels.append(is_face)
            predictions.append(int(predictions[0][0] > 0.5))  # Assuming a threshold of 0.5 for classification

    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"Accuracy: {accuracy*100:.2f}%")
