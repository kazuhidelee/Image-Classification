Neural Networks for CIFAR-10 Classification
This project involves implementing both a Fully Connected Neural Network (FCNN) and a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using PyTorch. The main goal is to train and test these networks, while also performing hyperparameter search to optimize model performance.

Table of Contents
Setup
Problem Statement
Fully Connected Neural Network
Convolutional Neural Network
File Structure
Submission
Setup
Prerequisites
This project uses Python for implementation. Ensure that you are working within a virtual environment to avoid dependency issues.

Create a new virtual environment:

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

For Windows:

bash
Copy
Edit
.\venv\Scripts\activate
For macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install the required dependencies:

bash
Copy
Edit
python -m pip install -r requirements.txt
Dataset
We use the CIFAR-10 dataset for this project, which contains 60,000 32x32 color images across 10 classes. You can find more information about the dataset here: CIFAR-10 Dataset.

PyTorch
You will primarily use PyTorch for building and training the neural networks. If you're new to PyTorch, refer to the official PyTorch Documentation.

Problem Statement
In this project, you will implement a Fully Connected Neural Network (FCNN) and a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. Your task includes:

Fully Connected Neural Network:

Implement a neural network with three linear layers.
Complete the forward pass and apply the activation function (e.g., ReLU) to the first two layers.
Train and test the model, then report the test accuracy.
Convolutional Neural Network:

Implement a CNN with two convolutional layers, followed by max pooling and a flatten layer.
Apply activation functions to the convolutional layers.
Complete the forward pass and ensure the model is correctly trained and tested.
Hyperparameter Search:

Explore various hyperparameters to optimize model performance.
Fully Connected Neural Network
Steps:
Define the model in models.py:

Implement a neural network with three layers: nn.Linear, nn.Linear, and a final classification layer.
The first layer should output 500 features, and the second should output 100 features.
Forward Pass:

Apply an activation function (e.g., ReLU) after the first and second layers.
Training and Testing:

Implement the train_and_test() function in run.py to train and test the model.
Ensure the test accuracy is above 50%.
Example Code:
python
Copy
Edit
if __name__ == "__main__":
    train_and_test(model_name="fcnet", dataset=CIFAR_10_dataset, num_epochs=5,
                   learning_rate=1e-3, activation_function_name="relu")
Convolutional Neural Network
Steps:
Define the CNN in models.py:

Implement two convolutional layers using nn.Conv2d.
Add nn.MaxPool2d for max pooling and nn.Flatten for flattening the output before the classification layer.
Forward Pass:

Apply activation functions to the convolutional layers only.
Flatten the output and pass it through the final classification layer.
Train and Test:

Use the same train_and_test() function to train and test the CNN.
File Structure
models.py: Contains the definition of the fully connected and convolutional neural networks.
data.py: Prepares the CIFAR-10 dataset and Dataloaders for training and testing.
run.py: Contains the train_and_test() function for training the models and evaluating test accuracy.
Submission
After completing the assignment, submit the following files to Gradescope:

models.py
run.py
cifar-10-fcn.pt (trained FCNN model)
cifar-10-cnn.pt (trained CNN model)
