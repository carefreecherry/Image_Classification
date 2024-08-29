# Fashion-CNN-Classifier
A Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify images from the Fashion MNIST dataset. The model is designed to accurately classify fashion items such as T-shirts, trousers, and shoes.

# Project Overview
This project implements a deep learning model using CNN architecture to recognize and classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images of 28x28 pixels, representing 10 different categories of clothing and accessories.

# Features
1. Convolutional Layers: Extract features from the images using multiple layers.
2. Max Pooling: Reduce the spatial dimensions of the feature maps.
3. Dropout: Prevent overfitting by randomly deactivating neurons during training.
4. Dense Layers: Fully connected layers for classification.
# Installation

Clone the repository:

git clone https://github.com/your-username/MNIST-CNN-Classifier.git

Navigate to the project directory:

cd MNIST-CNN-Classifier

# Usage

Load and preprocess the Fashion MNIST dataset.

Build and compile the CNN model.

Train the model on the training dataset.

Evaluate the model on the test dataset.

You can run the model training with the following command:

python train_model.py

# Model Architecture

Input Layer: 28x28 grayscale images

Conv2D Layers: Extract image features

MaxPooling2D Layer: Reduce dimensionality

Dropout Layer: Prevent overfitting

Flatten Layer: Convert 2D feature maps to 1D

Dense Layers: Classification layers

Output Layer: Softmax activation for 10 categories

# Results
The model is trained over 150 epochs with a batch size of 500. It achieves competitive accuracy in classifying fashion items.

# Acknowledgements
The Fashion MNIST dataset by Zalando Research.
Keras and TensorFlow for providing an accessible deep learning framework.
