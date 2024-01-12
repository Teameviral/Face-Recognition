import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint


# Set the path to your Images directory
images_directory = "outputimages"

# Set the path for saving the trained model
model_path = "image_recognition_model.h5"

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Define ImageDataGenerator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
# Generate batches of training data
train_generator = train_datagen.flow_from_directory(
    images_directory,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generate validation data
validation_generator = train_datagen.flow_from_directory(
    images_directory,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Calculate steps per epoch dynamically
steps_per_epoch = train_generator.samples // batch_size
if train_generator.samples % batch_size != 0:
    steps_per_epoch += 1

# Calculate validation steps dynamically
validation_steps = validation_generator.samples // batch_size
if validation_generator.samples % batch_size != 0:
    validation_steps += 1


# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add your custom classifier on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Use ModelCheckpoint to save the best model during training
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=[checkpoint]
)




