import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil

# Parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32

OBJECT_DIR = 'images/raw_object'
BACKGROUND_DIR = 'images/raw_background'

# Prepare folder structure for training and testing
base_dir = 'images/dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_obj_dir = os.path.join(train_dir, 'object')
train_bg_dir = os.path.join(train_dir, 'background')
test_obj_dir = os.path.join(test_dir, 'object')
test_bg_dir = os.path.join(test_dir, 'background')

# Remove old dataset folders if they exist
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

# Create new folders
os.makedirs(train_obj_dir)
os.makedirs(train_bg_dir)
os.makedirs(test_obj_dir)
os.makedirs(test_bg_dir)

# Function to split data into train/test folders
def split_data(source_dir, train_dest, test_dest, split_size=0.8):
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    train_size = int(len(files) * split_size)

    train_files = files[:train_size]
    test_files = files[train_size:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), train_dest)
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), test_dest)

# Split object and background data
split_data(OBJECT_DIR, train_obj_dir, test_obj_dir)
split_data(BACKGROUND_DIR, train_bg_dir, test_bg_dir)

print("Dataset preparation done.")

# Image data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
print("Starting training...")
history = model.fit(train_generator, epochs=15, validation_data=test_generator)

# Save the trained model
model.save('object_detector.h5')
print("Training complete and model saved as object_detector.h5")
