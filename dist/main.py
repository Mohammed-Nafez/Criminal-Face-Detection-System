import sys
import face_recognition
import pyautogui
import pygame
import os
import cv2
import numpy as np
import math
import tensorflow as tf

pygame.mixer.init()
sound = pygame.mixer.Sound(r'C:\Users\Mohammd Nafez Aloul\PycharmProjects\pythonProject2\Alarms\wrong-answer-129254.mp3')

# Helper function to preprocess face images
def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (64, 64))
    face_image = np.reshape(face_image, (64, 64, 1))
    face_image = face_image / 255.0
    return face_image

# Class for face recognition
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        self.load_model()
        # Create the "matches" folder if it doesn't exist
        if not os.path.exists('matches'):
            os.makedirs('matches')

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
        face_image = cv2.resize(face_image, (64, 64))
        face_image = np.reshape(face_image, (64, 64, 1))
        face_image = face_image / 255.0
        return face_image

    def predict_face(self, face_encoding):
        processed_encoding = self.preprocess_face(face_encoding)
        print(self.model.input_shape)
        print(processed_encoding.shape)
        predictions = self.model.predict(np.array([processed_encoding]))
        # Perform further processing on the predictions to determine the recognized face
        # ...

    def run_recognition(self):
        video_capture = cv2.VideoCapture(1)

        # Set the desired window width and height
        window_width = 1080
        window_height = 720

        # Create a named window with the specified width and height
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', window_width, window_height)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 0/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = " "
                    confidence = " "

                    if True in matches:
                        # Calculate the face distances to find the best match
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                        if len(face_distances) > 0:
                            # Find the index of the face with the minimum distance
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index].split(".")[0]
                                confidence = self.face_confidence(face_distances[best_match_index])

                                # Create a folder for the image if it doesn't exist
                                image_folder = os.path.join('matches', name)
                                if not os.path.exists(image_folder):
                                    os.makedirs(image_folder)

                                # Take a screenshot of the frame
                                screenshot = frame.copy()
                                screenshot_filename = os.path.join(image_folder, f"{name}_{confidence}.jpg")
                                cv2.imwrite(screenshot_filename, screenshot)
                                # Play the sound
                                sound.play()

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                # Calculate the text position
                text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)[0]
                text_x = left + int((right - left - text_width) / 2)
                text_y = bottom - 6

                # Draw the text
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # Resize the frame to the desired window width and height
            resized_frame = cv2.resize(frame, (window_width, window_height))

            # Display the resulting image
            cv2.imshow('Face Recognition', resized_frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def face_confidence(face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
