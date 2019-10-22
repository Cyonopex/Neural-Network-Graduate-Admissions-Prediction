# Neural-Network-Graduate-Admissions-Prediction
School project for Nanyang Technological University (CZ4042 Neural Networks) for predicting probability of graduate admission based on various factors

This is a neural network to create a model to predict the probability of a student entering a university program. The dataset can be found here:
https://www.kaggle.com/mohansacharya/graduate-admissions

## Dataset:

Feature vectors include:
1. GRE Score
2. TOEFL Score
3. University Rating
4. SOP
5. LOR
6. CGPA
7. Research

Regression model was made to predict:
- Chance of Admit

## Tasks

### Task 1 - Design a 3 layer Feed-forward neural network

Code is in **ProjectB1.py**. In this, I created a neural network of 10 neurons, L2 Regularisation of 10^-3, Learning rate of 10^-3 and batch size of 8.

Code is ran for 100k epochs, and the optimal epoch found was 45k.

### Task 2 - Plot an 8x8 Correlation Matrix

Code is in **ProjectB2.py**. Simple code using Pandas and Seaborn to plot a correlation matrix.

### Task 3 - Perform Recursive Feature Elimination

First, in **ProjectB3-6features.py**, one feature is eliminated at a time. Removing Feature 2 led to the best increase in performance.

Next, in **ProjectB3-5features.py**, one feature is eliminated at a time in addition to feature 2. Removing feature 7 led to the best increase in performance.

Lastly, in **ProjectB3-7-6-5.py**, I tested the neural network on 7 features vs 6 features (missing col2) vs 5 features (missing col2 and col7). 

Turns out that the 5 feature model had the best performance among the neural networks.

### Task 4 - Test against neural networks of 4 layers and 5 layers

Code is in **ProjectB4.py**. I compare the performance of the original 3 layer network with 5 features created in Task 3, against 4 and 5 layer networks with 50 neurons in the hidden layers. I also compare the performance of the presence of Dropout neurons and its effect on overfitting.

In this file, I also experimented with a multiprocessing approach to train the different models simultaneously, leading to much faster training times overall.

## Requirements

This code runs on Python 3.6 and above, Tensorflow 1.15 (although it should work fine on 1.14), Matplotlib 3.1.0, and the latest version of Seaborn/Pandas.

## References and Citations

This project is a school project for Nanyang Technological University's Neural Network course, AY19/20 Semester 1. 

The dataset is attributed to: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019
