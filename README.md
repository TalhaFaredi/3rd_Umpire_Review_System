3D Umpire Review System
This project implements a 3D Umpire Review System using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The goal of the system is to classify decisions as "out" or "not out" based on cricket imagery. The model utilizes a binary classification approach and processes images of various situations during a cricket match.

Features
Binary Image Classification: The system classifies cricket scenarios into two categories: "Out" or "Not Out."
Deep Learning Model: Utilizes CNN for efficient image processing.
Data Augmentation: The model applies transformations like rescaling, shearing, zooming, and flipping to generalize better.
Custom Model Training: A custom-built CNN model trained with TensorFlow/Keras.
Model Architecture
The system uses a CNN model with the following layers:

Convolutional layers followed by MaxPooling.
Dense layers with ReLU activation.
A sigmoid output layer for binary classification.
Model Summary
Input: 224x224 image with 3 channels (RGB).
Output: Binary classification (1 for "out", 0 for "not out").
Key Libraries Used:
python
Copy code
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import models, layers
Training & Validation
Data Augmentation
Training Data Augmentation: Images are rescaled, randomly sheared, zoomed, and flipped horizontally.
Validation Data: Only rescaled to ensure consistency during testing.
Model Compilation
The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

Training the Model:
The training is done over 30 epochs with the following results:

Training Accuracy: Reached up to 94.74%.
Validation Accuracy: Achieved around 50% due to a small validation dataset.
