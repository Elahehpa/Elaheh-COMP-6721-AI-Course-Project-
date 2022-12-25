# Elaheh-COMP-6721-AI-Course-Project-

Garbage Image Classification using Deep Learning
Contributors

Project Intro/Objective
Garbage classification is an important part of environmental protection. There are various laws regulating what can be considered garbage in different parts of the world. It can be difficult to accurately distinguish between different types of garbage. We are using deep learning to help us categorize garbage into different categories. We are using deep learning to prove that one of the biggest impediments to efficient waste management can finally be eliminated.

Methods Used
Data Preprocessing - We resized all training and test images to a size of 227x227 to best fit with our models. We incorporated data augmentation techniques including vertical flip, horizontal flip, width shift, height shift, zooming and rotation
Data Normalization - Using Mean and Standar Deviation to Normalize the image data and using tensors.
Deep Learning - Used deep cnn models to train the given dataset
Transfer Learning - Used pre-trained models with pre-trained weights to compare the metrics of our model.
Ablation Studies - Tried different parameters to see what results are acheived.
Hyperparameter Tuning - Used various learning rates and epochs to choose the best possible combination to generate accurate output.
Model Optimization Algorithms - Used Adam Optimation and Cross entropy loss function as optimization algorithms.
Technologies
Python
sklearn
numpy
matplotlib
torchvision, torch
Collab (To run the code)
Models
ResNet-50
AlexNet
VGG-16
Needs of this project
Understanding AI and Deep Learning
Learn to build models and solve a problem using CNN's
Model Optimization and Fine Tuning
Researching in AI and Deep Learning
Github File Structure
Scratch - Contains all the three models on three datasets trained from scratch.
ablation study - Contains the ablation studies and hyper parameter tuning perform using two scratch models (VGG16,AlexNet)
transfer learning = Contains all the three models on three datasets using pre-trained weights(performed transfer learning on ResNet-50 and AlexNet)
readme - Contains information about the project.
Getting Started
Clone this repo.

Upload datasets dataset1, dataset2, dataset3 to the drive.

(If using datasets offline then change the path in the code)

Install the library Numpy, torchvision, matplotlib

pip3 install numpy, torchvision, sklearn, torch, matplotlib
Run the project one after segment.

How to train/validate our models?
Run the code files on colab that you want to run.
Make sure if you are using another dataset, then change the path in data_dir variable.
If want to test the trained model, we have created checkpoint models that could be used for testing based on our trained models from scratch.
Run the pre-trained model on the provided sample test dataset
Change the line of code from the file
data_dir = "Garbage classification"
to
data_dir = "path to the sample test dataset"
You need to change the
