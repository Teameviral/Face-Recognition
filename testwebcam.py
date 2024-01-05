import cv2
import json
import imutils

def get_available_camera_index():
    # Check for the first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return None

# Find an available camera index
camera_index = get_available_camera_index()

if camera_index is not None:
    print(f"Using webcam with index: {camera_index}")
else:
    print("No available webcam found.")
    exit()

# Open the selected webcam
video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

while True:
    # Capture each frame
    ret, frame = video_capture.read()

    # Check if the frame is valid
    if not ret or frame is None:
        print("Error capturing frame. Exiting.")
        break

    # Resize frame to display it better
    frame = imutils.resize(frame, width=800)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check if the user pressed the 'q' key or closed the window
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

