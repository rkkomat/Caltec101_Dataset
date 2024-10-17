# Image Classification Using Caltech101 Dataset

This project is a part of my Data Mining course, where I implemented image classification using the Caltech101 dataset. The dataset contains images from 101 different classes, such as airplanes, cameras, and elephants. The classification is based on the *Edge Histogram* feature extracted from the images.

## Task Overview

The goal of this project was to classify images using different algorithms, evaluate their performance, and optimize parameters for the best accuracy. The algorithms used for classification were:
- Nearest Neighbor
- Support Vector Machine (SVM)
- Random Forest

### Dataset Files
- *Images.csv*: Contains the image IDs and their corresponding classes (e.g., 1;airplanes).
- *EdgeHistogram.csv*: Contains the Edge Histogram features for each image in the form of feature vectors.

### Training and Test Setup
- Different amounts of training images per class were used: 3, 5, 10, and 15.
- A suitable strategy for training and test image selection was chosen to evaluate the classifiers.
- The hyperparameters for each classifier were optimized to achieve the best accuracy.

## Classifiers Used

### 1. Nearest Neighbor
- Implemented using the k-NN algorithm, which classifies an image based on the majority class of its nearest neighbors.
  
### 2. Support Vector Machine (SVM)
- A classifier that separates data points using hyperplanes. Parameters such as kernel type and regularization were tuned for best performance.

### 3. Random Forest
- An ensemble method that uses multiple decision trees to classify images. Various parameters like the number of trees and max depth were optimized for accuracy.

## Results

- Experiments were conducted with different training set sizes (3, 5, 10, 15 images per class) and different classifiers.
- The hyperparameters for each algorithm were adjusted to achieve the best classification performance.
- A comprehensive analysis of accuracy and performance was performed for each classifier.

