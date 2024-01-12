import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Define the path to the folder containing your images
input_folder = "Images"

# Create an ImageDataGenerator with desired augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create the output folder if it doesn't exist
output_folder = "outputimages"
os.makedirs(output_folder, exist_ok=True)

# Perform data augmentation for each class (subfolder)
for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)

    # Skip if it's not a directory
    if not os.path.isdir(class_path):
        continue

    # Create a subfolder in the output directory for each class
    output_class_folder = os.path.join(output_folder, class_folder)
    os.makedirs(output_class_folder, exist_ok=True)

    # List all files in the class folder
    image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    # Perform data augmentation for each image in the class
    for image_file in image_files:
        img_path = os.path.join(class_path, image_file)
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target size as needed
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_folder, save_prefix=image_file.split('.')[0], save_format='jpg'):
            i += 1
            if i > 4:  # Generate a few augmented images per original image
                break
