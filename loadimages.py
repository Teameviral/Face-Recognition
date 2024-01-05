import face_recognition
import os
import json

# Current working directory is assumed to be where your images are located
image_directory = ""

# Image filenames and corresponding names to be used in index.json
image_data = {
    "Avishek Bhattacharjee": "avishek.jpg",
    "Selfie 1": "selfie 1.jpg"
}

# Create an empty dictionary to store face encodings
face_encodings_dict = {}

# Loop through each image and compute face encodings
for name, image_file in image_data.items():
    image_path = os.path.join(image_directory, image_file)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        # Take the first face encoding (assuming only one face per image)
        face_encodings_dict[name] = face_encodings[0].tolist()

# Save the face encodings to index.json
index_file_path = os.path.join(image_directory, "index.json")
with open(index_file_path, "w") as index_file:
    json.dump(face_encodings_dict, index_file)

print(f"Face encodings saved to {index_file_path}")
