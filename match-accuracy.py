import sys
import face_recognition
import pygame
import os
import cv2
import numpy as np
import math
import tensorflow as tf
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import concurrent.futures
import time

pygame.mixer.init()
sound = pygame.mixer.Sound(r'Alarms\wrong-answer-129254.mp3')

class FaceRecognition:
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        self.load_model()
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_frames = 0
        self.face_match_threshold = 0.8  # Adjust this threshold
        self.ground_truth = {}  # Dictionary to store ground truth labels
        self.manual_labeling_mode = False
        self.manual_labeling_name = None
        self.text_input = ""
        self.prev_face_names = []
        self.consecutive_matches = 0

    def encode_faces(self):
        for image in os.listdir('images'):
            face_image = face_recognition.load_image_file(f"images/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def load_model(self):
        self.model = tf.keras.models.load_model('face_recognition_model.h5')

    def preprocess_face(self, face_image):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (128, 128))
        face_image = np.reshape(face_image, (128, 128, 1))
        face_image = face_image / 255.0
        return face_image

    def predict_face(self, face_encoding):
        processed_encoding = self.preprocess_face(face_encoding)
        print(self.model.input_shape)
        print(processed_encoding.shape)
        predictions = self.model.predict(np.array([processed_encoding]))

    def detect_faces_caffe(self, frame):
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)

        detections = net.forward()

        face_locations = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.16:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                face_location = (startY, endX, endY, startX)
                face_locations.append(face_location)

        return face_locations

    def process_frame(self, frame):
        face_locations_caffe = self.detect_faces_caffe(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations_caffe)

        detected_face_names = []

        for face_location, face_encoding in zip(face_locations_caffe, face_encodings):
            top, right, bottom, left = face_location
            face_height = bottom - top
            face_width = right - left

            min_face_height_threshold = 0.2
            min_face_width_threshold = 0.2

            name = " "
            confidence = " "

            if face_height >= min_face_height_threshold and face_width >= min_face_width_threshold:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.face_match_threshold)

                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index].split(".")[0]
                            confidence = self.face_confidence(face_distances[best_match_index], self.face_match_threshold)

                            image_folder = os.path.join('matches', name)
                            if not os.path.exists(image_folder):
                                os.makedirs(image_folder)

                            screenshot = frame.copy()
                            screenshot_filename = os.path.join(image_folder, f"{name}_{confidence}.jpg")
                            cv2.imwrite(screenshot_filename, screenshot)

                            sound.play()

                            detected_face_names.append(name)

        # Check for true positives based on ground truth
        if self.manual_labeling_name is not None:
            for name in detected_face_names:
                if name in self.ground_truth and self.ground_truth[name] == self.manual_labeling_name:
                    self.true_positives += 1
                    # Manually label the ground truth for this frame
                    self.manual_label_ground_truth([name])

        if self.manual_labeling_mode and self.manual_labeling_name is not None:
            # Manually label the ground truth for the clicked face
            self.manual_label_ground_truth(detected_face_names)

        if hasattr(self, 'prev_face_names'):
            if set(self.prev_face_names) == set(detected_face_names):
                self.consecutive_matches += 1
                if self.consecutive_matches == 2:
                    self.show_match(frame, self.prev_face_names)
                    # Manually label the ground truth for this frame
                    self.manual_label_ground_truth(self.prev_face_names)
                else:
                    self.consecutive_matches = 0
            else:
                self.consecutive_matches = 0
                # Calculate false negatives
                for name in self.prev_face_names:
                    if name not in detected_face_names:
                        self.false_negatives += 1
                        # Manually label the ground truth for this frame
                        self.manual_label_ground_truth([name])
        else:
            # Calculate false positives
            self.false_positives += len(detected_face_names)

        self.prev_face_names = detected_face_names
        self.total_frames += 1

    def manual_label_ground_truth(self, names):
        # Manually label the ground truth for each person
        for name in names:
            if name not in self.ground_truth:
                if self.manual_labeling_name is not None:
                    self.ground_truth[name] = self.manual_labeling_name
                elif name not in self.ground_truth:
                    # Manually label as unknown if no name is provided
                    self.ground_truth[name] = "unknown"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.manual_labeling_mode = True
            self.manual_labeling_name = ""
            self.text_input = ""

    def show_match(self, frame, face_names):
        # Add your logic for displaying the match on the frame
        pass

    def run_recognition(self):
        Tk().withdraw()

        video_file = askopenfilename(filetypes=[("Video Files", "*.mp4")])
        video_capture = cv2.VideoCapture(video_file)

        window_width = 1920
        window_height = 1080

        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', window_width, window_height)
        cv2.setMouseCallback('Face Recognition', self.mouse_callback)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        frame_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            while True:
                ret, frame = video_capture.read()

                if self.process_current_frame:
                    if frame_count % 6 == 0:
                        self.process_frame(frame)

                frame_count += 1

                # Display the name input box on the frame
                if self.manual_labeling_mode:
                    cv2.putText(frame, "Enter name and press Enter:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, self.text_input, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                resized_frame = cv2.resize(frame, (window_width, window_height))
                cv2.imshow('Face Recognition', resized_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif self.manual_labeling_mode and key != -1:
                    if key == 13:  # Enter key pressed
                        self.manual_labeling_mode = False
                        self.manual_labeling_name = self.text_input
                        # Manually label the ground truth for the clicked face
                        self.manual_label_ground_truth(self.prev_face_names)
                    elif key == 27:  # Esc key pressed
                        self.manual_labeling_mode = False
                    else:
                        self.text_input += chr(key)

        # Calculate true negatives
        self.true_negatives = self.total_frames - (self.true_positives + self.false_positives + self.false_negatives)

        self.calculate_metrics()

        video_capture.release()
        cv2.destroyAllWindows()

    def calculate_accuracy(self):
        if self.true_positives + self.false_negatives > 0:
            accuracy = self.true_positives / (self.true_positives + self.false_negatives)
            print(f"Accuracy based on True Positives and False Negatives: {accuracy:.2%}")
        else:
            print("No true positives or false negatives to calculate accuracy.")

    def calculate_metrics(self):
        accuracy = (self.true_positives + self.true_negatives) / self.total_frames if self.total_frames > 0 else 0
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"True Positives: {self.true_positives}")
        print(f"False Negatives: {self.false_negatives}")
        

        self.calculate_accuracy()
        

    @staticmethod
    def face_confidence(face_distance, face_match_threshold):
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
