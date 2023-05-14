# Urdu Sign Language Recognition using Deep Learning

This project focuses on recognizing Urdu sign language gestures using deep learning techniques. The main challenge encountered was the unavailability of a suitable dataset for Urdu sign language. To overcome this, a custom dataset was created, comprising 36 different signs, with 1000 images captured for each sign.

## Dataset Creation
The following steps were followed to create the dataset:

1. **Image Capture**: Images of the Urdu sign language signs were captured using an appropriate setup.
2. **Train-Test Split**: The dataset was divided into training and testing sets. Randomly, 200 images were selected and placed in the test folder, while the remaining images were used for training.
3. **Preprocessing**: To enhance the quality of the images, they were converted to grayscale and thresholding was applied.

## Convolutional Neural Network (CNN) Architecture
A Convolutional Neural Network (CNN) was employed to recognize the Urdu sign language gestures. The CNN architecture used in this project is described below:

```
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 1), padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.40),
    Dense(96, activation="relu"),
    Dropout(0.40),
    Dense(64, activation="relu"),
    Dense(36, activation="softmax") # softmax for more than 2
])
```

### Architecture Explanation:
- The model starts with three convolutional layers (`Conv2D`) with 32 filters each, followed by ReLU activation functions. This helps the model to learn meaningful features from the input images.
- Max pooling layers (`MaxPooling2D`) are used to downsample the spatial dimensions of the features, reducing complexity and extracting the most relevant information.
- The `Flatten()` layer converts the 2D feature maps into a 1D vector, preparing it for the fully connected layers.
- Three fully connected (`Dense`) layers with 128, 96, and 64 units, respectively, were employed to perform high-level feature learning and classification.
- Dropout layers were added after the first two fully connected layers to reduce overfitting.
- The final `Dense` layer with 36 units and softmax activation provides the output probabilities for the 36 different Urdu sign language signs.

## Model Evaluation
The trained model was evaluated on the test dataset, achieving an accuracy of over 98%. However, during real-time testing, it was observed that the model faced challenges due to the similarity between certain Urdu sign language signs and the influence of the background. These factors affected the model's prediction accuracy.

Further improvements could be made by considering techniques such as data augmentation, additional preprocessing, and fine-tuning the model architecture to enhance its performance in real-world scenarios.

For more details, please refer to the code and documentation provided in the project repository.
