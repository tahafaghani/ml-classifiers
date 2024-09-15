
# Machine Learning Project - Diabetes Classification with PyTorch and TensorFlow and SKlearn

## Author: Taha Faghani

This project focuses on implementing various machine learning models, including deep learning using both **PyTorch**,**SKLearn** **TensorFlow**, to classify and analyze the diabetes dataset. The goal is to evaluate multiple models and determine the best performing algorithm.

## Dataset

The dataset used in this project is `diabetes.csv`, which contains several health indicators for predicting the onset of diabetes.

## Project Structure

The project contains several Jupyter notebooks, each of which implements a different machine learning model, including PyTorch and TensorFlow for deep learning. Below is the list of files and their descriptions:

- `diabetes.csv`: The dataset used for training and testing the models.
- `dt_rf.ipynb`: Implements Decision Tree and Random Forest classifiers.
- `kmeans.ipynb`: Implements the K-Means clustering algorithm.
- `knn.ipynb`: Implements the K-Nearest Neighbors (KNN) algorithm.
- `libs.ipynb`: Contains library imports and common utility functions.
- `logistic_regression.ipynb`: Implements Logistic Regression for classification.
- `preprocess.ipynb`: Handles data preprocessing, including missing value imputation, scaling, and splitting.
- `svm.ipynb`: Implements Support Vector Machine (SVM) for classification.
- `ann_pytorch.ipynb`: Implements Artificial Neural Networks (ANN) using **PyTorch**.
- `ann_tensorflow.ipynb`: Implements Artificial Neural Networks (ANN) using **TensorFlow/Keras**.

## Running the Project

1. Ensure you have the required dependencies installed:

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn torch tensorflow
   ```

2. Open and run each Jupyter notebook individually using Jupyter Lab or Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   You can also run the `.ipynb` files in any compatible environment such as Google Colab.

3. Each notebook contains the necessary code to preprocess the dataset, train the model, and evaluate its performance.

## Models Implemented

### Supervised Learning Models

1. **Logistic Regression**: Implemented in `logistic_regression.ipynb`.
2. **Decision Tree and Random Forest**: Implemented in `dt_rf.ipynb`.
3. **K-Nearest Neighbors (KNN)**: Implemented in `knn.ipynb`.
4. **Support Vector Machine (SVM)**: Implemented in `svm.ipynb`.
5. **Artificial Neural Networks (ANN) - PyTorch**: Implemented in `ann_pytorch.ipynb`.
6. **Artificial Neural Networks (ANN) - TensorFlow**: Implemented in `ann_tensorflow.ipynb`.

### Unsupervised Learning Models

1. **K-Means Clustering**: Implemented in `kmeans.ipynb`.

## Data Preprocessing

The dataset is preprocessed in the `preprocess.ipynb` notebook. This includes:
- Handling missing values.
- Feature scaling using `StandardScaler`.
- Splitting the dataset into training and testing sets.

## Evaluation Metrics

For each classification model, the following metrics are calculated:
- **Accuracy**: Overall correctness of the model.
- **Precision**: The number of true positive results divided by the number of positive results predicted by the model.
- **Recall**: The number of true positive results divided by the number of positives that should have been predicted.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix to visualize the performance of the model.

## Artificial Neural Networks (ANN)

### PyTorch Implementation (`ann_pytorch.ipynb`)
In this notebook, an artificial neural network is implemented using the **PyTorch** deep learning framework. The model uses two hidden layers and ReLU activations, followed by a Sigmoid activation in the output layer to predict binary outcomes (diabetes or not).

### TensorFlow/Keras Implementation (`ann_tensorflow.ipynb`)
In this notebook, an artificial neural network is implemented using **TensorFlow/Keras**. Similar to the PyTorch implementation, this model uses two hidden layers with ReLU activations, followed by a Sigmoid activation in the output layer.


