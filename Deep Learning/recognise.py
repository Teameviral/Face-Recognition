import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

# Load the trained model
model = load_model('image_recognition_model.h5')

# Define the class labels based on your training data
class_labels = ["APJ Abdul Kalam", "Apple", "Avishek Bhattacharjee", "HETC", "Mango"]

# Function to predict the class of an image
def predict_class(image):
    # Preprocess the image
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(normalized_image.reshape(1, 224, 224, 3))

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label


# Function to open webcam and recognize in real-time
def open_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Preprocess frame for prediction (resize, normalize, etc.)

        # Make prediction
        predicted_class = predict_class(frame)

        # Display the predicted class on the frame
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Webcam Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to open file explorer and recognize an image
def open_file_explorer():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    if file_path:
        # Load the image
        image = cv2.imread(file_path)

        # Preprocess image for prediction (resize, normalize, etc.)

        # Make prediction
        predicted_class = predict_class(image)

        # Display the predicted class on the image
        cv2.putText(image, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Image Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# GUI for recognize.py
def recognize_gui():
    root = Tk()
    root.title("Image Recognition")

# Set the initial size of the window
    root.geometry("300x200")

    # Increase button font size based on the current size of the GUI
    button_font = ("Helvetica", int(root.winfo_reqheight() / 10))

    # Button to open webcam
    webcam_button = Button(root, text="Open Webcam", command=open_webcam, font=button_font)
    webcam_button.pack(pady=10)

    # Button to open file explorer
    explorer_button = Button(root, text="Open Explorer", command=open_file_explorer, font=button_font)
    explorer_button.pack(pady=10)

    root.mainloop()

# Run the GUI
recognize_gui()
