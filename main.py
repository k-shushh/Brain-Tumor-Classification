from roboflow import Roboflow
rf = Roboflow(api_key="ncXyX85amkWFvK30pEat")
project = rf.workspace("ali-rostami").project("labeled-mri-brain-tumor-dataset")
version = project.version(1)
dataset = version.download("tensorflow")

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob
import pandas as pd
import tensorflow as tf

import os
import shutil

# Dataset directory
DATASET_DIR = "/content/Labeled-MRI-Brain-Tumor-Dataset-1"

# Mapping prefixes to class names
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

# Organizing images into respective classes
for split in ["train", "valid", "test"]:
    split_dir = os.path.join(DATASET_DIR, split)

    # Loop through each file in the current split directory
    for img_name in os.listdir(split_dir):
        # Debugging: print the filename to ensure it's being processed correctly
        print(f"Processing: {img_name}")

        # Check if the item is a file before processing
        if os.path.isfile(os.path.join(split_dir, img_name)):
            for key in class_mapping:
                if img_name.startswith(key):
                    class_name = class_mapping[key]
                    break
                else:
                    print(f"Warning: No matching class for file: {img_name}")
                    continue  # This 'continue' skips the current file if class isn't found

            # Debugging: print the extracted prefix and class name
            if class_name:
                print(f"Found class: {class_name} for file: {img_name}")
            else:
                print(f"Warning: No matching class for file: {img_name}")

            if class_name:  # If a valid class is found
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)  # Create class folder if it doesn't exist
                shutil.move(os.path.join(split_dir, img_name), os.path.join(class_dir, img_name))  # Move the image to the appropriate folder

import shutil
import os

base_path = "/content/Labeled-MRI-Brain-Tumor-Dataset-1/train"

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

#Model Building
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

#CNN Model
import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 16 , kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3) ))

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

#Preparing data Generator

def preprocessingImages1(path):
  """
  imput : Path
  output : Pre processed images
  """
  # data augmemtation
  image_data = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, rescale = 1/255, horizontal_flip = True)
  image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32, class_mode = 'categorical')

  return image

# Early stopping and model check point

from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping

es = EarlyStopping(monitor="val_accuracy", min_delta = 0.01, patience = 3, verbose = 1, mode = 'auto')

#model check point
mc = ModelCheckpoint(monitor="val_accuracy",filepath="./bestmodel.h5", verbose = 3, save_best_only = True, mode = 'auto')

cd = [es,mc]

#Model Training
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
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c = "red")

plt.title("loss vs val-loss")
plt.show()

# Model Accuracy
from keras.models import load_model

model = load_model("/content/bestmodel.h5")

model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics = ['accuracy'])

acc = model.evaluate(test_data)[1]

print(f"The accuracy of model is {acc*100} %")

#Test Case
from tensorflow.keras.utils import load_img, img_to_array

path = "/content/Labeled-MRI-Brain-Tumor-Dataset-1/test/NoTumor/Tr-no_0466_jpg.rf.abeeb51c69ba93bf3e6f6844dd4155df.jpg"

img = load_img(path, target_size = (224,224))
input_arr = img_to_array(img)/255

input_arr.shape

input_arr = np.expand_dims(input_arr, axis = 0)

pred_probs = model.predict(input_arr)  # Get prediction probabilities
pred = np.argmax(pred_probs, axis=1)[0]

pred

class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']  # Based on train_data.class_indices
predicted_class = class_names[pred]
print("Predicted class:", predicted_class)
