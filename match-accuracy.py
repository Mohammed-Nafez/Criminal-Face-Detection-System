import cv2
import face_recognition
import numpy as np
import os

def test_face_recognition_video(video_path, known_face_encodings, known_face_names, confidence_threshold=0.8, skip_frames=5, frame_limit=None):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Dictionary to store vertical offsets for each face
    vertical_offsets = {}

    while True:
        ret, frame = video_capture.read()

        if not ret or (frame_limit is not None and frame_count >= frame_limit):
            break

        # Skip frames if necessary
        if frame_count % skip_frames == 0:
            # Perform face detection and recognition
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if not face_locations:
                # No face detected in the frame
                if all(name == "Unknown" for name in known_face_names):
                    true_negatives += 1
                else:
                    false_negatives += 1

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare faces with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    # Calculate the face distances to find the best match
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        # Find the index of the face with the minimum distance
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            confidence = 1 - face_distances[best_match_index]

                # Consider a detection as positive only if confidence is above the threshold
                if confidence > confidence_threshold:
                    if name != "Unknown":
                        true_positives += 1
                    else:
                        false_positives += 1

                    # Print predicted value and confidence
                    print(f"Frame {frame_count}: Predicted: {name}, Confidence: {confidence:.2f}")

                # Determine the vertical position for the text
                top, _, _, _ = face_location
                vertical_offset = vertical_offsets.get(name, 0)
                text_y = top + vertical_offset

                # Update the vertical offset for the next frame
                vertical_offsets[name] = vertical_offset + 30  # Adjust this value as needed

                # Display the result on the frame
                text = f"Predicted: {name}, Confidence: {confidence:.2f}"
                cv2.putText(frame, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Face Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()
    print("True Positives:", true_positives)
    print("False Positives:", false_positives)
    print("True Negatives:", true_negatives)
    print("False Negatives:", false_negatives)
    

if __name__ == "__main__":
    # Replace 'path_to_test_video.mp4' with the path to your test video file
    test_video_path = r'C:\Users\Mohammd Nafez Aloul\Downloads\Videos\10000000_6457683190953863_2884099765419198804_n.mp4'

    # Replace 'path_to_known_faces_directory' with the path to the directory containing images of known faces
    known_faces_directory = 'images'

    known_face_encodings = []
    known_face_names = []

    # Load known face encodings and names
    for image_file in os.listdir(known_faces_directory):
        image_path = os.path.join(known_faces_directory, image_file)
        face_image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(face_image)[0]

        known_face_encodings.append(encoding)
        known_face_names.append(image_file.split(".")[0])

    # Call the testing function for video with skipping frames, confidence threshold, and frame limit
    test_face_recognition_video(test_video_path, known_face_encodings, known_face_names, confidence_threshold=0.4, skip_frames=5, frame_limit=100)
