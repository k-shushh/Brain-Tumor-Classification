# 🧠 Brain Tumor Classification using Deep Learning

This project focuses on classifying brain tumors using labeled MRI scan data and deep learning techniques in TensorFlow. It leverages the **Roboflow** platform for dataset management and applies a structured pipeline to prepare, train, and evaluate a classification model on four tumor types: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

## 🚀 Project Overview

The notebook walks through:

- Dataset import using Roboflow API.
- Data preprocessing and cleaning.
- Organizing MRI images into class-specific folders.
- Building and training a convolutional neural network (CNN) in TensorFlow.
- Evaluating the model with accuracy and performance metrics.
- Visualizing results through plots.

## 📁 Dataset

The dataset used is:
**Labeled MRI Brain Tumor Dataset** (sourced via Roboflow)

Classes:
- `Glioma`
- `Meningioma`
- `Pituitary`
- `No Tumor`

Structure:
```
/train
  /Glioma
  /Meningioma
  /Pituitary
  /NoTumor
/valid
/test
```

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Roboflow (for dataset import)

## 📦 Installation

Before running the notebook, install dependencies:

```bash
pip install roboflow
```

Ensure TensorFlow is installed and GPU is enabled for faster training.

## 🔑 Roboflow API

The dataset is pulled using:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="ncXyX85amkWFvK30pEat")
project = rf.workspace("ali-rostami").project("labeled-mri-brain-tumor-dataset")
version = project.version(1)
dataset = version.download("tensorflow")
```

## 📊 Results

- Model Accuracy: 76.82%

## 📈 Visualizations

Sample visualizations include:
- Training/Validation Loss and Accuracy plots
- Image samples for each tumor class

## ✨ Future Improvements

- Hyperparameter tuning
- Model optimization and deployment
- Integration with web or mobile applications

## 📌 Acknowledgements

- Dataset provided by [Roboflow](https://roboflow.com)
- Colab GPU runtime for model training
