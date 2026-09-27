# Forward Propagation, Backpropagation and Hyperparameter Analysis using TensorFlow/Keras

## Introduction

This practical implements **forward propagation and backpropagation using TensorFlow/Keras** on the **MNIST handwritten digit dataset**. The neural network is trained to classify handwritten digits from 0 to 9.

The practical also analyzes how different **learning rates** and **numbers of epochs** affect the performance of the neural network. The final model is evaluated using test accuracy, loss, training/validation curves, and a confusion matrix.

---

## Project Overview

The main objectives of this practical are:

- Load and preprocess the MNIST dataset.
- Normalize image pixel values.
- Reshape the 28×28 images into 784-dimensional input vectors.
- Build a neural network using TensorFlow/Keras.
- Understand forward propagation and backpropagation.
- Experiment with different learning rates.
- Analyze the effect of different numbers of epochs.
- Train a final neural network model.
- Visualize training and validation performance.
- Evaluate the model using test accuracy and loss.
- Generate a confusion matrix for classification analysis.

---

## Machine Learning Pipeline

```text
                 MNIST Dataset
                       │
                       ▼
              Load Training Data
                       │
                       ▼
              Data Preprocessing
             ┌─────────┴─────────┐
             │                   │
          Reshape              Normalize
       28 × 28 → 784           / 255
             │                   │
             └─────────┬─────────┘
                       ▼
                Neural Network
                       │
                       ▼
              Forward Propagation
                       │
                       ▼
                    Loss
                       │
                       ▼
             Backpropagation
                       │
                       ▼
              Weight Optimization
                       │
                       ▼
             Hyperparameter Analysis
             ┌─────────┴─────────┐
             │                   │
        Learning Rate          Epochs
             │                   │
             └─────────┬─────────┘
                       ▼
                 Final Model
                       │
                       ▼
                Model Evaluation
                       │
             ┌─────────┼─────────┐
             │         │         │
          Accuracy    Loss   Confusion Matrix
```

---

## Dataset

The **MNIST dataset** is a standard dataset used for handwritten digit classification.

It contains:

- **60,000 training images**
- **10,000 testing images**
- Images of handwritten digits from **0 to 9**
- Each image has a resolution of **28 × 28 pixels**
- There are **10 output classes**

Each pixel contains an intensity value ranging from 0 to 255.

### Dataset Structure

```text
MNIST
│
├── Training Set
│   ├── Images: 60,000
│   └── Labels: 60,000
│
└── Testing Set
    ├── Images: 10,000
    └── Labels: 10,000
```

The dataset is loaded directly using TensorFlow/Keras:

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

---

## Data Preprocessing

Before training the neural network, the MNIST images are preprocessed.

### 1. Reshaping

The original image size is:

```text
28 × 28
```

Since the neural network uses fully connected `Dense` layers, each image is converted into a one-dimensional vector containing 784 pixels.

```text
28 × 28 = 784
```

The reshaping is performed using:

```python
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
```

### 2. Normalization

The original pixel values range from:

```text
0 to 255
```

These values are divided by 255 so that they fall within:

```text
0 to 1
```

```python
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
```

Normalization helps the neural network train more effectively.

---

## Neural Network Architecture

The neural network consists of:

```text
Input Layer
    ↓
Dense Layer – 128 neurons
    ↓
Dense Layer – 64 neurons
    ↓
Output Layer – 10 neurons
```

### Architecture Configuration

| Layer | Number of Neurons | Activation |
|------|-------------------:|------------|
| Input | 784 | — |
| Hidden Layer 1 | 128 | ReLU |
| Hidden Layer 2 | 64 | ReLU |
| Output Layer | 10 | Softmax |

The model is created using:

```python
def create_model(lr):
    m = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    m.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return m
```

---

## Why Flatten the Images?

The MNIST images are originally represented as:

```text
28 × 28
```

However, a fully connected neural network expects a one-dimensional input.

Therefore:

```text
28 × 28 image
      ↓
784 pixel values
      ↓
Input to Dense layer
```

The reshaping operation converts every image into a vector of 784 values.

---

## Activation Functions

### ReLU

The hidden layers use the **Rectified Linear Unit (ReLU)** activation function.

```text
ReLU(x) = max(0, x)
```

It allows the network to learn nonlinear relationships and is computationally efficient.

The hidden layers use:

```python
activation='relu'
```

### Softmax

The output layer uses the **Softmax** activation function.

```python
activation='softmax'
```

Softmax converts the output values into probabilities for the 10 digit classes.

For example:

```text
Digit 0 → 0.01
Digit 1 → 0.00
Digit 2 → 0.02
Digit 3 → 0.01
Digit 4 → 0.03
Digit 5 → 0.01
Digit 6 → 0.00
Digit 7 → 0.02
Digit 8 → 0.05
Digit 9 → 0.85
```

The class with the highest probability is selected as the predicted digit.

---

# Forward Propagation

Forward propagation is the process through which the input data passes through the neural network to produce an output.

The process can be represented as:

```text
Input
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Hidden Layer
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Output Layer
  ↓
Prediction
```

For a neuron, the weighted sum can be represented as:

```text
z = Wx + b
```

where:

- `W` = weights
- `x` = input
- `b` = bias

The activation function is then applied to `z`.

---

# Backpropagation

Backpropagation is used to update the neural network's weights based on the error between the predicted output and the actual output.

The general process is:

```text
Prediction
    ↓
Calculate Loss
    ↓
Calculate Gradients
    ↓
Propagate Error Backwards
    ↓
Update Weights
    ↓
Next Training Iteration
```

The Adam optimizer is used to update the weights.

```python
Adam(learning_rate=lr)
```

The objective is to minimize the loss function and improve classification accuracy.

---

# Model Compilation

The model uses the **Adam optimizer** and **sparse categorical cross-entropy** loss.

```python
m.compile(
    optimizer=Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

The **Adam optimizer** is used for gradient-based optimization.

### Loss Function

Since MNIST is a multi-class classification problem with integer class labels, the following loss function is used:

```text
Sparse Categorical Cross-Entropy
```

### Evaluation Metric

The primary metric used is:

```text
Accuracy
```

---

# Hyperparameter Analysis

One of the main objectives of this practical is to analyze the effect of different hyperparameters on model performance.

Two hyperparameters are investigated:

1. Learning Rate
2. Number of Epochs

---

# Experiment 1: Effect of Learning Rate

The following learning rates are tested:

```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
```

Each model is trained for 5 epochs.

```python
results = []

for lr in learning_rates:
    model = create_model(lr)

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        verbose=0
    )

    loss, acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results.append([lr, acc])
```

The results are stored in a Pandas DataFrame:

```python
df = pd.DataFrame(
    results,
    columns=['Learning Rate', 'Accuracy']
)

print(df)
```

### Learning Rate Results

| Learning Rate | Accuracy |
|--------------:|---------:|
| 0.1 | 71.88% |
| 0.01 | 97.10% |
| 0.001 | 97.54% |
| 0.0001 | 94.51% |

### Learning Rate Visualization

```python
plt.plot(
    df['Learning Rate'],
    df['Accuracy'],
    marker='o'
)

plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Learning Rate vs Accuracy')
plt.grid()
plt.show()
```

### Observation

The experiment shows that the learning rate has a significant effect on the training performance.

A very large learning rate such as `0.1` produces considerably lower accuracy, while learning rates such as `0.01` and `0.001` provide much better results.

The learning rate of `0.001` is used for the final model.

---

# Experiment 2: Effect of Number of Epochs

The following epoch values are tested:

```python
epochs_list = [5, 10, 20]
```

The learning rate is fixed at:

```text
0.001
```

The models are trained using:

```python
res = []

for ep in epochs_list:

    model = create_model(0.001)

    h = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=ep,
        batch_size=128,
        verbose=0
    )

    loss, acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    res.append([ep, acc])
```

The results are stored using Pandas:

```python
edf = pd.DataFrame(
    res,
    columns=['Epochs', 'Accuracy']
)

print(edf)
```

### Epoch Results

| Epochs | Accuracy |
|-------:|---------:|
| 5 | 97.12% |
| 10 | 97.41% |
| 20 | 97.50% |

### Epoch Visualization

```python
plt.plot(
    edf['Epochs'],
    edf['Accuracy'],
    marker='o'
)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epochs vs Accuracy')
plt.grid()
plt.show()
```

### Observation

Increasing the number of epochs improves the model's performance.

However, the improvement becomes smaller as the number of epochs increases, indicating that the model is approaching a stable performance level.

---

# Final Model

Based on the experiments, the final model is trained using:

```text
Learning Rate = 0.001
Epochs = 20
Batch Size = 128
Validation Split = 20%
```

The final model is created using:

```python
model = create_model(0.001)
```

It is trained for 20 epochs:

```python
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128
)
```

---

# Training and Validation Accuracy

The training history contains both training and validation accuracy.

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

This visualization helps analyze how the model's accuracy changes during training.

---

# Training and Validation Loss

The loss curves are also plotted.

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

The loss plot helps observe how the prediction error changes throughout training.

---

# Model Evaluation

After training, the model is evaluated on the test dataset.

```python
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

The final model achieves approximately:

```text
Test Accuracy ≈ 97.44%
Test Loss ≈ 0.1171
```

The exact result may vary slightly depending on the execution environment and training run.

---

# Confusion Matrix

A confusion matrix is generated to analyze the classification performance for each digit.

Predictions are obtained using:

```python
y_pred = np.argmax(
    model.predict(x_test),
    axis=1
)
```

The confusion matrix is calculated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

It can be displayed using:

```python
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title('Confusion Matrix')
plt.show()
```

The confusion matrix shows the number of correctly and incorrectly classified samples for each digit.

---

# Data Visualization

The practical generates several visualizations to analyze model performance.

### Visualizations Included

1. Learning Rate vs Accuracy
2. Epochs vs Accuracy
3. Training Accuracy vs Validation Accuracy
4. Training Loss vs Validation Loss
5. Confusion Matrix

These visualizations make it easier to understand the behavior of the neural network during training and evaluation.

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network API |
| NumPy | Numerical operations |
| Pandas | Data analysis and result tables |
| Matplotlib | Data visualization |
| Scikit-learn | Confusion matrix and evaluation |
| Google Colab | Development and execution environment |

---

# Key Concepts Demonstrated

This practical demonstrates the following deep learning concepts:

- MNIST digit classification
- Data preprocessing
- Image normalization
- Data reshaping
- Neural network architecture
- Dense layers
- ReLU activation
- Softmax activation
- Forward propagation
- Backpropagation
- Loss functions
- Adam optimizer
- Learning rate
- Epochs
- Batch size
- Validation split
- Model evaluation
- Accuracy
- Loss
- Confusion matrix
- Hyperparameter analysis
- Training and validation visualization

---

# Project Structure

```text
lab3/
│
├── DL_A3.ipynb
│
└── README.md
```

### DL_A3.ipynb

Contains the complete implementation of:

- MNIST dataset loading
- Data preprocessing
- Neural network creation
- Learning rate experiment
- Epoch experiment
- Final model training
- Training/validation visualization
- Test evaluation
- Confusion matrix

### README.md

Contains the project documentation, methodology, results, observations, and implementation details.

---

# How to Run

## Option 1: Google Colab

1. Open Google Colab.
2. Upload `DL_A3.ipynb`.
3. Run the notebook cells sequentially.
4. TensorFlow/Keras will load the MNIST dataset automatically.
5. Observe the generated results and graphs.

---

## Option 2: Jupyter Notebook

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Then open the notebook:

```bash
jupyter notebook DL_A3.ipynb
```

Run the cells sequentially.

---

# Results Summary

The experiments demonstrate the effect of learning rate and epochs on neural network performance.

### Learning Rate Experiment

| Learning Rate | Test Accuracy |
|--------------:|--------------:|
| 0.1 | 71.88% |
| 0.01 | 97.10% |
| 0.001 | 97.54% |
| 0.0001 | 94.51% |

### Epoch Experiment

| Number of Epochs | Test Accuracy |
|-----------------:|--------------:|
| 5 | 97.12% |
| 10 | 97.41% |
| 20 | 97.50% |

### Final Model

```text
Learning Rate : 0.001
Epochs        : 20
Batch Size    : 128
Test Accuracy : ~97.44%
Test Loss     : ~0.1171
```

The experiments demonstrate that appropriate hyperparameter selection can significantly affect neural network performance.

---

# Limitations

Although the model achieves high accuracy on MNIST, there are some limitations:

- MNIST is a relatively simple image classification dataset.
- The network uses fully connected layers rather than convolutional layers.
- The model may not perform as well on more complex image datasets.
- Only a limited number of learning rates and epoch values were tested.
- No extensive hyperparameter optimization was performed.

---

# Future Improvements

The project can be extended in several ways:

- Use a **Convolutional Neural Network (CNN)** for image classification.
- Experiment with different network architectures.
- Test additional learning rates.
- Experiment with different batch sizes.
- Use learning-rate scheduling.
- Apply dropout for regularization.
- Compare Adam with other optimizers such as SGD and RMSprop.
- Perform more extensive hyperparameter tuning.
- Evaluate the model on more complex datasets such as CIFAR-10.

---

# Key Takeaway

This practical demonstrates how a neural network can be trained for handwritten digit classification using TensorFlow/Keras.

The experiments show that **learning rate and number of epochs have a direct effect on model performance**. By analyzing these hyperparameters, an appropriate configuration can be selected for training the final model.

The practical provides hands-on understanding of:

```text
Data Preprocessing
       ↓
Neural Network
       ↓
Forward Propagation
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Weight Optimization
       ↓
Hyperparameter Analysis
       ↓
Model Evaluation
```

---

# Author

**Aadit Paranjape**

Department of Computer Science and Engineering – Artificial Intelligence

Deep Learning Laboratory

Academic Year: 2026–27
