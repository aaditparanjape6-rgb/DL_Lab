# Data Preprocessing and Visualization using TensorFlow/Keras

## Introduction

This practical focuses on setting up and using **TensorFlow/Keras in Google Colab** for basic deep learning workflows. It demonstrates the essential steps involved in preparing data for a machine learning model, including data preprocessing, normalization, train-test splitting, and visualization.

The practical provides a foundation for working with datasets in TensorFlow/Keras and understanding how raw data is transformed into a form suitable for training a deep learning model.

---

## Project Overview

The main objectives of this practical are:

- Install and configure TensorFlow/Keras in Google Colab.
- Load and inspect a sample dataset.
- Perform basic data preprocessing.
- Normalize the input data.
- Divide the dataset into training and testing sets.
- Visualize the dataset and its characteristics.
- Understand the importance of preprocessing before model training.

---

## Machine Learning Pipeline

```text
                 Sample Dataset
                       │
                       ▼
                  Load Dataset
                       │
                       ▼
                Inspect Dataset
                       │
                       ▼
              Data Preprocessing
                       │
             ┌─────────┴─────────┐
             │                   │
        Cleaning/Handling     Normalization
             │                   │
             └─────────┬─────────┘
                       ▼
                Train-Test Split
                       │
             ┌─────────┴─────────┐
             │                   │
        Training Data        Testing Data
             │                   │
             └─────────┬─────────┘
                       ▼
                  Visualization
                       │
                       ▼
               Prepared Dataset
