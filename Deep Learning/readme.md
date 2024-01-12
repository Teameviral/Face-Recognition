# Train our Model

# Train a best neural network model based on the images directory (D:\Facial Emotion Detection\Images) here.
# Now understand Images directory - it has different sub folders
## Images Directory Structure

```plaintext
Images
│
├── APJ Abdul Kalam
│   ├── AP1.jpg
│   ├── AP2.jpg
│   ├── AP3.jpg
│   └── AP4.jpg
│
├── Apple
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
│
├── Avishek Bhattacharjee
│   ├── A1.jpg
│   ├── A2.jpg
│   ├── A3.jpg
│   └── A4.jpg
│
├── HETC
│   ├── H1.jpg
│   ├── H2.jpg
│   ├── H3.jpg
│   └── H4.jpg
│
└── Mango
    ├── M1.jpg
    ├── M2.jpg
    ├── M3.jpg
    └── M4.jpg

```

# Different sub folders have different images in the form of .jpg, .jpeg, .png with different size.
# Each and every subfolder belongs to a category
# There are 3 categories 
# Which sub folder has two or more words with spaces like APJ ABDUL KALAM - It's a huaman name
# Which has only single word like Apple, Mango are fruit category.
# Which has CAPSLOCK format like HETC - It's a college category.
# Train like this we can recognize real time objects using webcam or opening image through the sub-folder name and is category.
# Training model must predict and classify all images via its sub folder name just using its model path
# That means if model stores in .hf or .tf or .csv or any format - just take model path in different file (recognise.py)


# In recognize.py, there is simple recognize gui that has two buttons 
# One is open webcam and another is open file explorer
# When we open webcam, it tries to predict the images from the training model and say its realtime name
# Like if the person is APJ Abdul Kalam - it prints it on webcam until the person present
# If person leaves webcam it should not print anything
# When new person come, it prints new name if it's on the model like Avishek Bhattacharjee
# Quit Webcam -Pressing Q
# Open Explorer button while clicking and open any image - it checks image from training model and check size of image , resize it to fit name on image and show
# Can be closed
