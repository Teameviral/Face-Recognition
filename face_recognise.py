import face_recognition
import cv2
import json

# Load the face encodings from index.json
index_file_path = "index.json"  # Replace with the actual path
with open(index_file_path, "r") as index_file:
    face_encodings_dict = json.load(index_file)

# Create arrays to store face encodings and corresponding labels
known_face_encodings = list(face_encodings_dict.values())
known_face_names = list(face_encodings_dict.keys())

# Initialize some variables
face_encodings = []
face_names = []
face_locations=[] #Track Movement
process_this_frame = True

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = video_capture.read() #Return Value - ret "Boolean Function" - Frame is read or not.

    # Resize frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 
    '''
    frame -  real frame Size || Output size sets zero, zero to modify chamges 
    ''' 
    # Only process every other frame to save time
    if process_this_frame:
        # Find all face locations and face encodings in the current frame
        face_locations =face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name below the face
        cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

