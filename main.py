from roboflow import Roboflow
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import pandas as pd
import tensorflow as tf

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

# === Download dataset ===
rf = Roboflow(api_key="ncXyX85amkWFvK30pEat")
project = rf.workspace("ali-rostami").project("labeled-mri-brain-tumor-dataset")
version = project.version(1)
dataset = version.download("tensorflow")

# Make sure this path is correct (matches downloaded folder name)
DATASET_DIR = os.path.join(os.getcwd(), "Labeled-MRI-Brain-Tumor-Dataset-1")

# === Class name mapping based on filename prefixes ===
class_mapping = {
    "Tr-gl_": "Glioma",
    "Tr-me_": "Meningioma",
    "Tr-pi_": "Pituitary",
    "Tr-no_": "NoTumor",
    "Glioma": "Glioma",
    "Meningioma": "Meningioma",
    "Pituitary": "Pituitary",
    "NoTumor": "NoTumor"
}

# === Organize images into class folders ===
for split in ["train", "valid", "test"]:
    split_dir = os.path.join(DATASET_DIR, split)

    if not os.path.exists(split_dir):
        print(f" Directory not found: {split_dir}")
        continue

    for img_name in os.listdir(split_dir):
        img_path = os.path.join(split_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        class_name = None
        for key in class_mapping:
            if img_name.startswith(key):
                class_name = class_mapping[key]
                break

        if class_name:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(img_path, os.path.join(class_dir, img_name))
            #print(f"Moved {img_name} -> {class_name}")

        #else:
            #print(f"Warning: No matching class for file: {img_name}")

print("All images have been organized into class folders.")

import shutil
import os

base_path = r"C:\Users\Supriya Gupta\Desktop\Labeled-MRI-Brain-Tumor-Dataset-1\train"

# Move Glioma
src_glioma = os.path.join(base_path, "Pituitary", "Glioma")
dst_glioma = os.path.join(base_path, "Glioma")
if os.path.exists(src_glioma):
    shutil.move(src_glioma, dst_glioma)

# Move Meningioma
src_meningioma = os.path.join(base_path, "Pituitary", "Meningioma")
dst_meningioma = os.path.join(base_path, "Meningioma")
if os.path.exists(src_meningioma):
    shutil.move(src_meningioma, dst_meningioma)

#Model Build

#CNN Model
import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape=(224,224,3)))
model.add(tf.keras.layers.Conv2D(filters = 16 , kernel_size = (3,3), activation = 'relu'))

model.add(tf.keras.layers.Conv2D(filters = 36 , kernel_size = (3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(filters = 64 , kernel_size = (3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(filters = 128 , kernel_size = (3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

model.add(tf.keras.layers.Dropout(rate = 0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(rate = 0.25))
model.add(tf.keras.layers.Dense(units = 4, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics = ['accuracy'])

#Processing Images

def preprocessingImages1(path):
  """
  imput : Path
  output : Pre processed images
  """
  # data augmemtation
  image_data = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, rescale = 1/255, horizontal_flip = True)
  image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'categorical')

  return image

path = r"C:\Users\Supriya Gupta\Desktop\VS CODE\ML 2\Labeled-MRI-Brain-Tumor-Dataset-1\test"
train_data = preprocessingImages1(path)

def preprocessingImages2(path):
  """
  imput : Path
  output : Pre processed images
  """

  image_data = ImageDataGenerator(rescale = 1/255)
  image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'categorical')

  return image

path = r"C:\Users\Supriya Gupta\Desktop\VS CODE\ML 2\Labeled-MRI-Brain-Tumor-Dataset-1\train"
test_data = preprocessingImages2(path)

path = r"C:\Users\Supriya Gupta\Desktop\VS CODE\ML 2\Labeled-MRI-Brain-Tumor-Dataset-1\valid"
val_data = preprocessingImages2(path)

# Early stopping and model check point

from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping

es = EarlyStopping(monitor="val_accuracy", min_delta = 0.01, patience = 3, verbose = 1, mode = 'auto')

#model check point
mc = ModelCheckpoint(monitor="val_accuracy",filepath=r"C:\Users\Supriya Gupta\Desktop\VS CODE\ML 2\bestmodel.h5", verbose = 3, save_best_only = True, mode = 'auto')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cd = [es,mc]

hs = model.fit(train_data,
                steps_per_epoch = 8,
                epochs = 30,
                verbose = 1,
                validation_data = val_data,
                validation_steps = 16,
                callbacks = cd)

# Model Graphical Interpretation

h = hs.history
h.keys()


plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c = "red")

plt.title("acc vs val-acc")
#plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c = "red")

plt.title("loss vs val-loss")
#plt.show()

# Model Accuracy
from keras.models import load_model

model = load_model(r"C:\Users\Supriya Gupta\Desktop\VS CODE\ML 2\bestmodel.h5")

model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics = ['accuracy'])

acc = model.evaluate(test_data)[1]

print(f"The accuracy of model is {acc*100} %")
