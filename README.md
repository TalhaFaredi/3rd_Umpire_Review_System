# 3D Umpire Review System

This project implements a 3D Umpire Review System using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The goal of the system is to classify decisions as "out" or "not out" based on cricket imagery. The model utilizes a binary classification approach and processes images of various situations during a cricket match.

## Features

- **Binary Image Classification**: The system classifies cricket scenarios into two categories: "Out" or "Not Out."
- **Deep Learning Model**: Utilizes CNN for efficient image processing.
- **Data Augmentation**: The model applies transformations like rescaling, shearing, zooming, and flipping to generalize better.
- **Custom Model Training**: A custom-built CNN model trained with TensorFlow/Keras.

## Model Architecture

The system uses a CNN model with the following layers:

1. Convolutional layers followed by MaxPooling.
2. Dense layers with ReLU activation.
3. A sigmoid output layer for binary classification.

### Model Summary

- Input: 224x224 image with 3 channels (RGB).
- Output: Binary classification (1 for "out", 0 for "not out").

### Key Libraries Used:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import models, layers
```

### Training & Validation

#### Data Augmentation

- **Training Data Augmentation**: Images are rescaled, randomly sheared, zoomed, and flipped horizontally.
- **Validation Data**: Only rescaled to ensure consistency during testing.

#### Model Compilation

The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

### Training the Model:

The training is done over 30 epochs with the following results:
- **Training Accuracy**: Reached up to 94.74%.
- **Validation Accuracy**: Achieved around 50% due to a small validation dataset.

## Usage

### Training Data Generator:

```python
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

### Validation Data Generator:

```python
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'Valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

### Model Compilation:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Model Training:

```python
model.fit(train_generator, epochs=30)
```

### Validation Accuracy:

```python
validation_results = model.evaluate(val_generator)
print("Validation Loss:", validation_results[0])
print("Validation Accuracy:", validation_results[1])
```

### Manual Accuracy Check:

```python
accuracy = np.mean(np.equal(true_labels, predicted_labels))
print("Manual Validation Accuracy:", accuracy)
```

### Model Prediction:

```python
img_path = 'not4.jpg'  
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
prediction = model.predict(img_array)
predicted_class = int(round(prediction[0][0]))
class_labels = {0: 'not out', 1: 'out'}
predicted_label = class_labels[predicted_class]
print("Predicted Label:", predicted_label)
```

### Model Saving:

```python
model.save('path.h5')
```

## Results

- **Predicted Label**: The model predicts whether the situation in the input image is "out" or "not out."
- **Accuracy**: Achieved 94.74% accuracy during training, but validation accuracy was 50% due to the limited validation data.

## How to Run

1. Ensure that TensorFlow and Keras are installed.
2. Place your training images in the `Training` directory and validation images in the `Valid` directory.
3. Run the Python script to train the model and evaluate its performance on your dataset.
4. For custom image predictions, modify the `img_path` in the prediction block to point to your desired image.

## Future Improvements

- Expand the training and validation datasets to improve generalization.
- Fine-tune hyperparameters for optimal performance.
- Add more advanced augmentation techniques to improve robustness.

---

