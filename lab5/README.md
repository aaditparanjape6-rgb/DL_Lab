# RNN vs LSTM vs GRU for Human Activity Recognition

## Overview

This project compares three recurrent neural network architectures — **Simple RNN, LSTM, and GRU** — for **Human Activity Recognition (HAR)** using smartphone sensor data from the **UCI Human Activity Recognition Using Smartphones Dataset**.

The models process sequential sensor data and are evaluated using Accuracy, Precision, Recall, and F1-Score.

---

## Objectives

- Implement Simple RNN, LSTM, and GRU models for sequence classification.
- Use smartphone sensor data for human activity recognition.
- Compare the performance of the three recurrent architectures.
- Analyze validation accuracy and validation loss.
- Evaluate the models using Accuracy, Precision, Recall, and F1-Score.
- Visualize confusion matrices for each model.

---

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains sensor recordings collected from smartphones while participants performed six different physical activities.

### Activities

The six activity classes are:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

### Sensor Data

The project uses nine inertial sensor signals:

- Body acceleration X
- Body acceleration Y
- Body acceleration Z
- Body gyroscope X
- Body gyroscope Y
- Body gyroscope Z
- Total acceleration X
- Total acceleration Y
- Total acceleration Z

Each sample contains **128 time steps**.

Therefore, the input shape to the models is:

```text
128 time steps × 9 sensor features
```

---

## Dataset Download

The notebook automatically downloads and extracts the UCI HAR dataset if it is not already present.

The dataset is downloaded from:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
```

No manual dataset download is required before running the notebook.

---

## Methodology

The overall workflow is:

```text
UCI HAR Dataset
        ↓
Download / Extract Dataset
        ↓
Load Inertial Sensor Signals
        ↓
Combine 9 Sensor Features
        ↓
Create Sequential Input
        ↓
RNN / LSTM / GRU Models
        ↓
Training with Validation Split
        ↓
Model Evaluation
        ↓
Performance Comparison
        ↓
Confusion Matrices
```

---

## Data Preparation

The nine sensor signals are loaded separately and combined along the feature axis.

The resulting data representation is:

```text
Samples × 128 Time Steps × 9 Features
```

The activity labels originally range from 1 to 6 and are converted to:

```text
0 → Walking
1 → Walking Upstairs
2 → Walking Downstairs
3 → Sitting
4 → Standing
5 → Laying
```

---

## Models

Three different recurrent neural network architectures are implemented.

### 1. Simple RNN

```text
Input (128 × 9)
      ↓
SimpleRNN (64)
      ↓
Dropout (30%)
      ↓
Dense (32, ReLU)
      ↓
Dense (6, Softmax)
```

### 2. LSTM

```text
Input (128 × 9)
      ↓
LSTM (64)
      ↓
Dropout (30%)
      ↓
Dense (32, ReLU)
      ↓
Dense (6, Softmax)
```

### 3. GRU

```text
Input (128 × 9)
      ↓
GRU (64)
      ↓
Dropout (30%)
      ↓
Dense (32, ReLU)
      ↓
Dense (6, Softmax)
```

All three models use the same general classifier structure so that their recurrent layers can be compared under similar conditions.

---

## Training Configuration

The models use the following training configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 64 |
| Maximum Epochs | 10 |
| Validation Split | 15% |
| Early Stopping | Enabled |
| Early Stopping Patience | 2 |
| Recurrent Units | 64 |
| Dropout | 0.30 |
| Dense Units | 32 |
| Output Classes | 6 |

Early stopping is used to stop training when the validation loss stops improving and to restore the best-performing model weights.

---

## Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the overall proportion of correctly classified activity samples.

### Precision

Measures how many samples predicted as a particular activity actually belong to that activity.

### Recall

Measures how many samples belonging to an activity are correctly detected.

### F1-Score

Provides a combined measure of precision and recall.

The project calculates weighted Precision, Recall, and F1-Score.

---

## Visualizations

The notebook includes several visualizations for analyzing the dataset and model performance.

### Activity Distribution

The distribution of samples across the six activity classes is visualized to understand the dataset.

### Sensor Signal Visualization

Sensor readings are plotted across time steps to visualize the sequential nature of the data.

### Validation Accuracy

Validation accuracy is plotted for the RNN, LSTM, and GRU models to compare their learning behavior.

### Validation Loss

Validation loss curves are plotted to analyze model convergence and training behavior.

### Model Comparison

The final test accuracy of the three models is displayed for comparison.

### Confusion Matrices

A separate confusion matrix is generated for each model to analyze classification performance across all six activities.

---

## Model Comparison

The experiment compares:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Simple RNN | 69.97% | 71.39% | 69.97% | 69.86% |
| LSTM | 50.08% | 45.95% | 50.08% | 45.91% |
| GRU | 53.92% | 55.02% | 53.92% | 48.78% |

These values correspond to the experiment represented in the provided reference material. Model performance can vary depending on the execution environment, training behavior, and other experimental conditions.

---

## Why Compare RNN, LSTM, and GRU?

Recurrent architectures are designed to process sequential data, making them suitable for sensor-based activity recognition.

### Simple RNN

Simple RNNs provide a basic recurrent architecture for processing sequences but can have difficulty retaining information over longer sequences.

### LSTM

LSTM networks introduce memory cells and gating mechanisms to control the flow of information through the sequence.

### GRU

GRUs use a simpler gating structure than LSTMs while still providing mechanisms for retaining relevant information.

Comparing the three architectures helps demonstrate how different recurrent structures behave on the same sequential classification problem.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Project Structure

```text
lab5/
│
├── DL_A5.ipynb
└── README.md
```

The UCI HAR dataset is downloaded automatically by the notebook and does not need to be stored in the repository.

---

## How to Run

### 1. Open the notebook

Open:

```text
DL_A5.ipynb
```

using Jupyter Notebook, JupyterLab, Google Colab, or VS Code with the Jupyter extension.

### 2. Install Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 3. Run the Notebook

Execute the cells sequentially.

The notebook will:

1. Download the UCI HAR dataset if required.
2. Extract the dataset.
3. Load the sensor signals.
4. Prepare the sequential input data.
5. Build the RNN, LSTM, and GRU models.
6. Train each model.
7. Evaluate their performance.
8. Generate comparison plots.
9. Generate confusion matrices.

---

## Applications

Human Activity Recognition using sensor data can be used in areas such as:

- Fitness tracking
- Smartphone activity monitoring
- Wearable devices
- Healthcare monitoring
- Elderly activity monitoring
- Sports analysis
- Smart home systems
- Human-computer interaction

---

## Limitations

- The experiment uses the predefined UCI HAR dataset.
- The models use a relatively simple architecture.
- The number of training epochs is limited to 10.
- The models use only the available sensor signals from the dataset.
- Performance can vary between different training runs.

---

## Future Improvements

Possible improvements include:

- Increasing the training dataset.
- Performing longer hyperparameter tuning.
- Testing different numbers of recurrent units.
- Comparing bidirectional RNN, LSTM, and GRU architectures.
- Using CNN-RNN hybrid architectures.
- Applying attention mechanisms.
- Using additional sensor modalities.
- Performing cross-validation.
- Deploying the trained model for real-time activity recognition.

---

## Key Takeaway

This project demonstrates the application of recurrent neural networks to sequential smartphone sensor data and provides a direct comparison between **Simple RNN, LSTM, and GRU** architectures for human activity recognition.

The experiment also demonstrates how validation curves, classification metrics, and confusion matrices can be used to analyze and compare sequence classification models.

---

## Dataset Reference

**UCI Machine Learning Repository — Human Activity Recognition Using Smartphones Dataset**

Dataset:

```text
Human Activity Recognition Using Smartphones
```

The dataset was created using smartphone accelerometer and gyroscope measurements collected from human subjects performing different activities.

---

## Author

**Aadit Paranjape**

Deep Learning Lab  
Vishwakarma Institute of Technology, Pune