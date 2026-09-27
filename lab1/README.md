# Data Preprocessing and Visualization using TensorFlow/Keras

## Introduction

This practical focuses on installing and configuring **TensorFlow/Keras in Google Colab** and performing the fundamental data preprocessing steps required before developing a deep learning model.

The practical demonstrates how a dataset can be loaded, inspected, preprocessed, normalized, divided into training and testing sets, and visualized. These steps form the foundation of a typical machine learning and deep learning workflow.

---

## Project Overview

The main objectives of this practical are:

- Install and configure TensorFlow/Keras in Google Colab.
- Load and inspect a sample dataset.
- Perform data preprocessing.
- Normalize the input data.
- Split the dataset into training and testing sets.
- Visualize the dataset and its characteristics.
- Understand the importance of preprocessing before model training.
- Prepare the dataset for subsequent deep learning tasks.

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
             Cleaning           Normalization
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
```

---

# TensorFlow and Keras

## TensorFlow

**TensorFlow** is an open-source machine learning and deep learning framework developed by Google. It provides tools and libraries for creating, training, and evaluating machine learning and deep learning models.

TensorFlow supports numerical computation, neural networks, data processing, model optimization, and deployment.

## Keras

**Keras** is a high-level deep learning API integrated with TensorFlow. It provides a simple and user-friendly interface for building neural networks.

Keras allows models to be constructed using components such as:

- Layers
- Optimizers
- Loss functions
- Metrics
- Training methods

---

# Google Colab

The practical is designed to be executed using **Google Colab**.

Google Colab provides a cloud-based Python development environment that can be accessed through a web browser. It allows users to execute Python code without requiring a complete local machine learning environment.

The notebook can be opened in Google Colab and executed cell by cell.

---

# Data Preprocessing

Data preprocessing is one of the most important stages of a machine learning workflow.

Raw data may not always be directly suitable for training a neural network. Preprocessing converts the data into a form that can be efficiently used by a machine learning algorithm.

The general preprocessing workflow used in this practical is:

```text
Raw Dataset
     │
     ▼
Data Inspection
     │
     ▼
Data Preprocessing
     │
     ▼
Normalization
     │
     ▼
Train-Test Split
     │
     ▼
Visualization
     │
     ▼
Model-Ready Data
```

---

# Dataset Inspection

Before performing preprocessing, the dataset is inspected to understand its structure and characteristics.

Important properties that can be examined include:

- Number of samples
- Number of features
- Shape of the dataset
- Data types
- Input values
- Target values
- Distribution of the data

Dataset inspection helps determine which preprocessing techniques are appropriate.

---

# Data Normalization

Normalization is used to scale numerical values into a suitable range.

A common normalization approach is to scale values between:

```text
0 and 1
```

For example:

```text
x_normalized = x / max(x)
```

Normalization is useful because neural networks generally perform better when input values are on a consistent scale.

It can also help optimization algorithms converge more efficiently during training.

---

# Train-Test Split

The dataset is divided into separate training and testing sets.

```text
                    Dataset
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       Training Data        Testing Data
```

## Training Data

The training dataset is used to develop the machine learning model.

The model learns patterns and relationships between the input features and target values using this data.

## Testing Data

The testing dataset is kept separate from the training process.

It is used to evaluate how well the trained model performs on previously unseen data.

Keeping testing data separate helps provide a more realistic evaluation of model performance.

---

# Data Visualization

Visualization is used to understand the dataset and identify patterns in the data.

Graphs and plots can help analyze:

- Data distribution
- Relationships between variables
- Differences between classes
- Patterns in the dataset
- Possible anomalies

Visualization is an important part of exploratory data analysis because it allows the characteristics of a dataset to be understood more easily.

---

# Deep Learning Workflow

The practical demonstrates the basic workflow that can be followed before developing a deep learning model.

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Inspection
   │
   ▼
Preprocessing
   │
   ▼
Normalization
   │
   ▼
Train-Test Split
   │
   ▼
Visualization
   │
   ▼
Prepared Dataset
   │
   ▼
Deep Learning Model
```

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network API |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Google Colab | Development and execution environment |

---

# Key Concepts Demonstrated

This practical demonstrates the following concepts:

- TensorFlow/Keras configuration
- Google Colab
- Dataset loading
- Dataset inspection
- Data preprocessing
- Data normalization
- Train-test splitting
- Data visualization
- Exploratory data analysis
- Preparation of data for deep learning

---

# Project Structure

```text
lab1/
│
├── DL_A1.ipynb
│
└── README.md
```

## `DL_A1.ipynb`

The notebook contains the implementation of the practical, including the setup and data preprocessing workflow.

The notebook covers:

- TensorFlow/Keras setup
- Dataset loading
- Dataset inspection
- Data preprocessing
- Normalization
- Train-test splitting
- Data visualization

## `README.md`

This file contains the documentation for the practical, including the objective, workflow, preprocessing concepts, technologies used, and execution instructions.

---

# How to Run

## Using Google Colab

1. Open Google Colab.
2. Upload `DL_A1.ipynb`.
3. Run the notebook cells sequentially.
4. Allow the required libraries and dataset to load.
5. Execute the preprocessing operations.
6. Observe the generated outputs and visualizations.

---

## Using Jupyter Notebook

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
DL_A1.ipynb
```

Run the notebook cells sequentially.

---

# Results

The practical demonstrates the complete basic data preparation workflow required before training a deep learning model.

The workflow can be summarized as:

```text
Dataset Loading
       ↓
Dataset Inspection
       ↓
Data Preprocessing
       ↓
Normalization
       ↓
Train-Test Split
       ↓
Visualization
       ↓
Prepared Dataset
```

The processed dataset can subsequently be used for machine learning and deep learning model development.

---

# Limitations

The practical primarily focuses on data preparation rather than complete model development.

Some limitations include:

- The preprocessing techniques depend on the characteristics of the selected dataset.
- The practical focuses on basic preprocessing operations.
- Visualization is limited to the techniques implemented in the notebook.
- No extensive model training or hyperparameter optimization is performed.

---

# Future Improvements

The practical can be extended by:

- Building a neural network using the preprocessed data.
- Comparing different normalization techniques.
- Applying additional feature scaling methods.
- Performing more detailed exploratory data analysis.
- Implementing different machine learning models.
- Comparing model performance before and after preprocessing.
- Adding additional data visualizations.
- Applying advanced preprocessing techniques to larger datasets.

---

# Key Takeaway

Data preprocessing is a fundamental part of every machine learning and deep learning workflow.

Before a model can learn effectively, the input data needs to be properly inspected, processed, normalized, and divided into appropriate datasets.

The complete workflow demonstrated in this practical is:

```text
Raw Dataset
     ↓
Inspection
     ↓
Preprocessing
     ↓
Normalization
     ↓
Train-Test Split
     ↓
Visualization
     ↓
Model-Ready Data
```

Understanding these steps provides the foundation for developing more advanced deep learning models using TensorFlow and Keras.

---

# Author

**Aadit Paranjape**

Department of Computer Science and Engineering – Artificial Intelligence

Deep Learning Laboratory

Academic Year: 2026–27
