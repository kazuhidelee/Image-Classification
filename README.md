# 🕸️ Neural Networks for CIFAR-10 Classification

This project involves implementing both a Fully Connected Neural Network (FCNN) and a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using TensorFlow. The main goal is to train and test these networks, while also performing hyperparameter search to optimize model performance.

## Table of Contents

- [Setup](#setup)
- [Problem Statement](#problem-statement)
- [Fully Connected Neural Network](#fully-connected-neural-network)
- [Convolutional Neural Network](#convolutional-neural-network)
- [File Structure](#file-structure)
- [Submission](#submission)

## Setup

### Prerequisites

This project uses Python for implementation. Ensure that you are working within a virtual environment to avoid dependency issues.

1. Create a new virtual environment:

   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install tensorflow
   ```
## Dataset
We use the CIFAR-10 dataset for this project, which contains 50,000 32x32 color images across 10 classes. You can find more information about the dataset here: CIFAR-10 Dataset.

## Problem Statement
In this project, I have implemented a Fully Connected Neural Network (FCNN) and a Convolutional Neural Network (CNN) to classify images from the CIFAR-100 dataset. 

1. Fully Connected Neural Network:

 - Implement a neural network with three linear layers.
 - Complete the forward pass and apply the activation function (e.g., ReLU) to the first two layers.
 - Train and test the model, then report the test accuracy.

2. Convolutional Neural Network:

 - Implement a CNN with three convolutional layers, followed by max pooling and a flatten layer.
 - Apply activation functions to the convolutional layers.
 - Complete the forward pass and ensure the model is correctly trained and tested.
 - 
3. Hyperparameter Search:
 - Explore various hyperparameters to optimize model performance.
