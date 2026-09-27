# Multilayer Perceptron for Classification using TensorFlow/Keras

## Introduction

This practical implements a **Multilayer Perceptron (MLP)** using TensorFlow/Keras for classification. A Multilayer Perceptron is a type of artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

The practical demonstrates the fundamental working of a neural network, including data preprocessing, model architecture, forward propagation, backpropagation, model training, validation, and evaluation.

The implementation provides an understanding of how a neural network learns patterns from input data and uses those learned patterns to perform classification.

---

# Project Overview

The main objectives of this practical are:

- Understand the concept of a Multilayer Perceptron.
- Build a neural network using TensorFlow/Keras.
- Prepare the dataset for classification.
- Perform data preprocessing.
- Divide the data into training and testing sets.
- Design a multilayer neural network.
- Understand forward propagation.
- Understand backpropagation.
- Train the neural network.
- Monitor training and validation performance.
- Evaluate the trained model.
- Analyze classification performance using evaluation metrics and visualizations.

---

# Machine Learning Pipeline

```text
                    Dataset
                       │
                       ▼
                Data Preprocessing
                       │
                       ▼
                 Data Splitting
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Training Data       Testing Data
              │                 │
              ▼                 │
      Multilayer Perceptron     │
              │                 │
              ▼                 │
       Forward Propagation      │
              │                 │
              ▼                 │
         Loss Calculation       │
              │                 │
              ▼                 │
        Backpropagation         │
              │                 │
              ▼                 │
        Weight Updates          │
              │                 │
              └────────┬────────┘
                       ▼
                Model Evaluation
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
          Accuracy    Loss   Confusion Matrix
```

---

# What is a Multilayer Perceptron?

A **Multilayer Perceptron (MLP)** is a feedforward artificial neural network consisting of multiple layers of interconnected neurons.

A basic MLP architecture can be represented as:

```text
Input Layer
     │
     ▼
Hidden Layer
     │
     ▼
Hidden Layer
     │
     ▼
Output Layer
     │
     ▼
Prediction
```

Unlike a single-layer perceptron, an MLP contains one or more hidden layers.

The hidden layers allow the network to learn more complex and nonlinear relationships in the data.

---

# Structure of an MLP

An MLP generally contains three types of layers:

## Input Layer

The input layer receives the features of the dataset.

Each input neuron corresponds to one input feature.

```text
Input Features
      ↓
Input Layer
```

## Hidden Layers

Hidden layers process the input data and learn patterns.

Each hidden layer contains multiple neurons.

```text
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
```

## Output Layer

The output layer produces the final prediction.

For a classification problem, the output layer typically contains neurons corresponding to the different classes.

```text
Hidden Layers
     ↓
Output Layer
     ↓
Class Prediction
```

---

# Neural Network Architecture

The practical uses fully connected `Dense` layers to construct the Multilayer Perceptron.

The general architecture is:

```text
Input Features
      │
      ▼
Dense Layer
      │
      ▼
Hidden Layer
      │
      ▼
Hidden Layer
      │
      ▼
Output Layer
      │
      ▼
Class Prediction
```

### Architecture Components

| Layer | Type | Purpose |
|------|------|---------|
| Input Layer | Input/Dense | Receives input features |
| Hidden Layer(s) | Dense | Learns patterns and representations |
| Output Layer | Dense | Produces the final classification |

The exact configuration of the model is implemented in the provided notebook.

---

# Data Preprocessing

Before training the neural network, the input data must be prepared.

The general preprocessing workflow is:

```text
Raw Dataset
      ↓
Data Inspection
      ↓
Data Preprocessing
      ↓
Normalization / Scaling
      ↓
Train-Test Split
      ↓
Model Training
```

Preprocessing ensures that the input data is in an appropriate format for the neural network.

---

# Data Splitting

The dataset is divided into training and testing datasets.

```text
                    Dataset
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Training Data       Testing Data
```

## Training Dataset

The training dataset is used by the neural network to learn patterns in the data.

During training, the model adjusts its weights based on the difference between its predictions and the actual target values.

## Testing Dataset

The testing dataset is not used for learning the model parameters.

Instead, it is used after training to evaluate how well the model performs on previously unseen data.

A validation split may also be used during training to monitor performance on a separate portion of the training data.

---

# Forward Propagation

**Forward propagation** is the process through which input data moves from the input layer through the hidden layers and finally reaches the output layer.

The process can be represented as:

```text
Input
  │
  ▼
Hidden Layer 1
  │
  ▼
Hidden Layer 2
  │
  ▼
Output Layer
  │
  ▼
Prediction
```

Each neuron performs a weighted calculation.

The basic equation is:

```text
z = Wx + b
```

where:

- `W` = weights
- `x` = input
- `b` = bias
- `z` = weighted sum

An activation function is then applied to the weighted sum.

---

# Activation Functions

Activation functions introduce non-linearity into neural networks.

Without activation functions, multiple layers of a neural network would behave similarly to a single linear transformation.

## ReLU

The **Rectified Linear Unit (ReLU)** is commonly used in hidden layers.

It is defined as:

```text
ReLU(x) = max(0, x)
```

ReLU allows the network to learn nonlinear relationships while being computationally efficient.

---

## Softmax

For multi-class classification, the output layer can use the **Softmax** activation function.

Softmax converts the output values into probabilities for the different classes.

For example:

```text
Class 0 → 0.02
Class 1 → 0.05
Class 2 → 0.81
Class 3 → 0.04
Class 4 → 0.08
```

The class with the highest probability becomes the predicted class.

---

# Backpropagation

**Backpropagation** is the process used to update the neural network's weights based on the error in its predictions.

The general process is:

```text
Prediction
     ↓
Loss Calculation
     ↓
Gradient Calculation
     ↓
Backpropagation
     ↓
Weight Updates
     ↓
Next Training Iteration
```

The gradients indicate how the model's weights should be changed to reduce the loss.

This allows the neural network to gradually improve its predictions during training.

---

# Loss Function

The loss function measures the difference between the predicted output and the actual target.

For a multi-class classification problem with integer class labels, a commonly used loss function is:

```text
Sparse Categorical Cross-Entropy
```

The model attempts to minimize this loss during training.

---

# Model Compilation

Before training, the neural network must be compiled.

The compilation step specifies:

- Optimizer
- Loss function
- Evaluation metrics

A typical configuration is:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

# Adam Optimizer

The practical uses the **Adam optimizer** for updating the model's weights.

Adam is an optimization algorithm that combines ideas from momentum-based optimization and adaptive learning rates.

The optimizer uses the gradients calculated during backpropagation to update the network parameters.

---

# Model Training

The model is trained using the training dataset.

A typical training operation is:

```python
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
```

During training, the neural network repeatedly performs:

```text
Forward Propagation
       ↓
Prediction
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Weight Update
```

This process is repeated over multiple epochs.

---

# Epochs

An **epoch** represents one complete pass through the training dataset.

For example:

```text
1 Epoch  → Dataset processed once
5 Epochs → Dataset processed five times
10 Epochs → Dataset processed ten times
```

Increasing the number of epochs can allow the model to learn more from the training data.

However, training for too many epochs can potentially lead to overfitting.

---

# Batch Size

The batch size determines the number of training samples processed before the model updates its weights.

For example:

```text
Batch Size = 32
```

means that the model processes 32 samples before performing a weight update.

Batch size can affect both training speed and model convergence.

---

# Training and Validation Accuracy

The training history can be used to visualize the model's accuracy.

```python
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.show()
```

The graph allows the training and validation performance to be compared across epochs.

---

# Training and Validation Loss

Loss values can also be visualized:

```python
plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
```

The loss curves provide information about how the prediction error changes during training.

---

# Model Evaluation

After training, the model is evaluated using the testing dataset.

```python
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

The main evaluation metrics are:

```text
Test Loss
Test Accuracy
```

## Accuracy

Accuracy represents the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy indicates that a larger proportion of samples were classified correctly.

---

# Confusion Matrix

A confusion matrix provides a detailed view of classification performance.

It compares:

```text
Actual Classes
      vs
Predicted Classes
```

The confusion matrix can be generated using Scikit-learn.

For example:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = np.argmax(
    model.predict(x_test),
    axis=1
)

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title('Confusion Matrix')
plt.show()
```

The confusion matrix helps identify which classes are correctly classified and which classes are being confused with each other.

---

# Performance Visualization

The practical can generate several visualizations to understand model performance.

### Main Visualizations

1. Training Accuracy
2. Validation Accuracy
3. Training Loss
4. Validation Loss
5. Confusion Matrix

These visualizations help analyze the learning behavior of the neural network.

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network API |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Google Colab/Jupyter | Development environment |

---

# Key Concepts Demonstrated

This practical demonstrates the following deep learning concepts:

- Artificial Neural Networks
- Multilayer Perceptron
- Dense Layers
- Input and Output Layers
- Hidden Layers
- Activation Functions
- ReLU
- Softmax
- Forward Propagation
- Backpropagation
- Loss Functions
- Adam Optimizer
- Gradient-based Optimization
- Epochs
- Batch Size
- Training
- Validation
- Classification
- Model Evaluation
- Accuracy
- Loss
- Confusion Matrix
- Data Visualization

---

# Project Structure

```text
lab2/
│
├── lab_2.ipynb
├── multilayer_perceptron.ipynb
└── README.md
```

## `lab_2.ipynb`

Contains the implementation of the Lab 2 practical and the associated neural network experiments.

## `multilayer_perceptron.ipynb`

Contains the implementation and demonstration of the Multilayer Perceptron model.

## `README.md`

Contains the complete documentation for the practical, including the MLP architecture, preprocessing, training process, evaluation, and key concepts.

---

# How to Run

## Using Google Colab

1. Open Google Colab.
2. Upload the required `.ipynb` notebook.
3. Run the cells sequentially.
4. Allow the required libraries and dataset to load.
5. Execute the preprocessing and model-building cells.
6. Train the model.
7. Observe the training and validation results.
8. Evaluate the model using the testing dataset.
9. Analyze the generated visualizations.

---

## Using Jupyter Notebook

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the required notebook and execute the cells sequentially.

---

# Results

The practical demonstrates the complete workflow for implementing a Multilayer Perceptron:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
MLP Architecture
   ↓
Forward Propagation
   ↓
Loss Calculation
   ↓
Backpropagation
   ↓
Weight Optimization
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Visualization
```

The trained model can be evaluated using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Test loss
- Confusion matrix

The exact numerical results depend on the dataset, architecture, hyperparameters, and execution of the notebook.

---

# Limitations

Some limitations of the Multilayer Perceptron approach include:

- Fully connected networks can become computationally expensive for high-dimensional inputs.
- MLPs do not explicitly preserve spatial relationships in image data.
- Model performance depends on the selected architecture and hyperparameters.
- A basic MLP may not perform as effectively as specialized architectures such as CNNs for image-based problems.
- Extensive hyperparameter tuning is not included in the basic implementation.

---

# Future Improvements

The project can be extended by:

- Experimenting with different numbers of hidden layers.
- Changing the number of neurons in each hidden layer.
- Testing different activation functions.
- Comparing different optimizers.
- Experimenting with different learning rates.
- Changing batch sizes.
- Applying dropout for regularization.
- Using batch normalization.
- Performing systematic hyperparameter tuning.
- Comparing the MLP with a Convolutional Neural Network.
- Testing the model on more complex datasets.

---

# Key Takeaway

A **Multilayer Perceptron** provides a fundamental understanding of how artificial neural networks learn patterns from data.

The practical demonstrates the complete learning process:

```text
Input Data
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradient Calculation
    ↓
Weight Updates
    ↓
Improved Predictions
```

Understanding this workflow provides a strong foundation for studying more advanced deep learning architectures and techniques.

---

# Author

**Aadit Paranjape**

Department of Computer Science and Engineering – Artificial Intelligence

Deep Learning Laboratory

Academic Year: 2026–27
