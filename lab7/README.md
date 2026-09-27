# Lab 7 – CNN Architecture Comparison using CIFAR-10

## Overview

This practical implements and compares four popular Convolutional Neural Network (CNN) architectures for image classification using the **CIFAR-10 dataset**.

The architectures compared in this practical are:

- **AlexNet**
- **VGG16**
- **ResNet50**
- **EfficientNetB0**

The models are trained and evaluated using a common dataset and training configuration. Their performance is compared using **test accuracy, test loss, validation accuracy, validation loss, training time, and model complexity**.

The practical also includes additional data visualization, augmentation visualization, parameter comparison, and incorrect prediction analysis.

## Objective

The main objectives of this practical are:

- Understand different CNN architectures used for image classification.
- Implement AlexNet using TensorFlow/Keras.
- Use pretrained VGG16, ResNet50, and EfficientNetB0 models.
- Apply transfer learning using ImageNet pretrained models.
- Train multiple architectures on the same image classification task.
- Compare the performance of different CNN architectures.
- Visualize training and validation performance.
- Compare model accuracy and training time.
- Analyze incorrect predictions.
- Identify the best-performing model based on test accuracy.

## Project Pipeline

```text
CIFAR-10 Dataset
        ↓
Load Dataset
        ↓
Select Training/Test Subset
        ↓
Visualize Sample Images
        ↓
Analyze Class Distribution
        ↓
Resize Images to 96 × 96
        ↓
Data Augmentation
        ↓
Create TensorFlow Data Pipeline
        ↓
Build CNN Architectures
        ↓
AlexNet
VGG16
ResNet50
EfficientNetB0
        ↓
Compile Models
        ↓
Train for 5 Epochs
        ↓
Evaluate on Test Data
        ↓
Compare Accuracy, Loss & Training Time
        ↓
Visualize Validation Performance
        ↓
Analyze Incorrect Predictions
        ↓
Compare Model Complexity
```

# Dataset

The project uses the **CIFAR-10 dataset**, which is available directly through TensorFlow/Keras.

The dataset contains:

- **60,000 color images**
- **10 classes**
- **32 × 32 RGB images**

The dataset is automatically downloaded when the following TensorFlow function is executed:

```python
tf.keras.datasets.cifar10.load_data()
```

## CIFAR-10 Classes

| Label | Class |
|------:|-------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

## Dataset Subset

Since four CNN architectures are trained and compared, a smaller subset of CIFAR-10 is used to reduce computational requirements.

The notebook uses:

```text
Training Images: 10,000
Testing Images: 2,000
```

This is controlled using:

```python
TRAIN_SIZE = 10000
TEST_SIZE = 2000
```

The subset is selected from the original CIFAR-10 training and test datasets.

# Data Preprocessing

## Label Encoding

The original CIFAR-10 labels are integer class values.

They are converted into one-hot encoded vectors using:

```python
to_categorical(y_train, 10)
```

and:

```python
to_categorical(y_test, 10)
```

This allows the models to use categorical cross-entropy as the loss function.

## Image Resizing

The original CIFAR-10 images are:

```text
32 × 32 × 3
```

They are resized to:

```text
96 × 96 × 3
```

using TensorFlow:

```python
tf.image.resize(
    x_train,
    (IMG_SIZE, IMG_SIZE)
)
```

The image size used in the practical is:

```python
IMG_SIZE = 96
```

The larger input size is useful when working with pretrained CNN architectures.

## Data Type Conversion

The image tensors are converted to floating-point values:

```python
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
```

# Data Visualization

Additional visualization has been included in the modified notebook to provide better understanding of the dataset before model training.

## Sample Image Visualization

Sample CIFAR-10 images are displayed along with their corresponding class labels.

This provides a visual understanding of the different categories present in the dataset.

## Class Distribution

A class distribution visualization is used to examine the number of samples belonging to each CIFAR-10 category.

This helps verify the distribution of the selected dataset subset.

## Data Augmentation Visualization

The notebook displays examples of images after augmentation.

This makes it easier to understand how transformations modify the original training images.

# Data Augmentation

Data augmentation is applied to increase the variety of training images.

The augmentation pipeline includes:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

## Augmentation Techniques

| Technique | Purpose |
|-----------|---------|
| Random Flip | Creates horizontally flipped image variations |
| Random Rotation | Slightly changes image orientation |
| Random Zoom | Creates variations at different scales |

Data augmentation can help the models generalize better to variations in the input images.

# TensorFlow Data Pipeline

The training data is converted into a TensorFlow dataset:

```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)
```

The training pipeline performs:

```text
Shuffle
   ↓
Batch
   ↓
Prefetch
```

The batch size is:

```python
BATCH_SIZE = 32
```

Prefetching is performed using:

```python
tf.data.AUTOTUNE
```

The test dataset is also batched and prefetched but is not shuffled.

# CNN Architectures

The practical compares four architectures:

```text
AlexNet
VGG16
ResNet50
EfficientNetB0
```

Each architecture represents a different CNN design approach.

# 1. AlexNet

AlexNet is implemented from scratch using TensorFlow/Keras layers.

The modified architecture uses convolutional layers followed by pooling, Batch Normalization, Global Average Pooling, Dense layers, and Dropout.

The architecture follows the general structure:

```text
Input
96 × 96 × 3
      ↓
Data Augmentation
      ↓
Conv2D
      ↓
Batch Normalization
      ↓
MaxPooling
      ↓
Conv2D
      ↓
Batch Normalization
      ↓
MaxPooling
      ↓
Conv2D
      ↓
Conv2D
      ↓
Conv2D
      ↓
MaxPooling
      ↓
Global Average Pooling
      ↓
Dense
      ↓
Dropout
      ↓
Dense — 10
      ↓
Softmax
```

The convolutional layers progressively extract more complex visual features from the CIFAR-10 images.

## Global Average Pooling

The modified implementation uses:

```python
layers.GlobalAveragePooling2D()
```

instead of directly flattening the complete feature maps.

Global Average Pooling reduces the number of parameters passed to the classification section.

## Batch Normalization

Batch Normalization is included in the modified convolutional architecture.

It helps stabilize the activations during training and can make optimization more consistent.

## Dropout

Dropout is used in the classification section to reduce overfitting.

# 2. VGG16

VGG16 is loaded using the Keras Applications API:

```python
VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

The model uses **ImageNet pretrained weights**.

The original ImageNet classification layer is removed using:

```python
include_top=False
```

## Transfer Learning

The VGG16 base model is frozen:

```python
base_model.trainable = False
```

A new classification head is added for the ten CIFAR-10 classes.

The general structure is:

```text
Input
   ↓
Data Augmentation
   ↓
VGG16 Pretrained Base
   ↓
Global Average Pooling
   ↓
Dense Layer
   ↓
Dropout
   ↓
Dense — 10
   ↓
Softmax
```

# 3. ResNet50

ResNet50 is loaded with ImageNet pretrained weights:

```python
ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

The original classification head is removed.

The pretrained feature extractor is frozen:

```python
base_model.trainable = False
```

The classification section contains:

```text
ResNet50 Base
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Dropout
      ↓
Dense — 10
      ↓
Softmax
```

ResNet50 is useful for demonstrating transfer learning using a deeper CNN architecture.

# 4. EfficientNetB0

EfficientNetB0 is loaded using:

```python
EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

The ImageNet pretrained feature extractor is frozen:

```python
base_model.trainable = False
```

The classification head follows:

```text
EfficientNetB0 Base
        ↓
Global Average Pooling
        ↓
Dense Layer
        ↓
Dropout
        ↓
Dense — 10
        ↓
Softmax
```

EfficientNet provides another modern architecture for comparison with the older AlexNet and VGG16 designs.

# Transfer Learning

VGG16, ResNet50, and EfficientNetB0 use **transfer learning**.

Instead of training the complete networks from scratch, pretrained ImageNet models are used as feature extractors.

The process is:

```text
ImageNet Pretrained Model
          ↓
Freeze Base Network
          ↓
Add New Classification Head
          ↓
Train on CIFAR-10
```

The pretrained layers are frozen using:

```python
base_model.trainable = False
```

This reduces the number of weights that need to be updated during training.

# Model Compilation

All four models use the same general compilation configuration.

## Optimizer

The Adam optimizer is used:

```python
tf.keras.optimizers.Adam(
    learning_rate=0.001
)
```

## Loss Function

The models use:

```text
Categorical Cross-Entropy
```

because the labels are one-hot encoded.

## Evaluation Metric

The primary evaluation metric is:

```text
Accuracy
```

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Training Images | 10,000 |
| Test Images | 2,000 |
| Image Size | 96 × 96 |
| Batch Size | 32 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | Categorical Cross-Entropy |
| Number of Classes | 10 |

The same general training configuration is used for the four architectures to make their comparison more consistent.

# Model Training

A reusable training function is used to train and evaluate each architecture.

The training process records:

- Training history
- Test loss
- Test accuracy
- Training time

Training time is measured using Python's `time` module.

The models are trained for:

```text
5 epochs
```

# Model Evaluation

After training, every model is evaluated using the test dataset.

The following values are recorded:

```text
Test Accuracy
Test Loss
Training Time
```

The results are stored in a Pandas DataFrame for easier comparison.

# Model Comparison

The notebook creates a comparison table containing:

| Model | Test Accuracy (%) | Test Loss | Training Time |
|-------|-------------------|-----------|---------------|
| AlexNet | Generated during execution | Generated during execution | Generated during execution |
| VGG16 | Generated during execution | Generated during execution | Generated during execution |
| ResNet50 | Generated during execution | Generated during execution | Generated during execution |
| EfficientNetB0 | Generated during execution | Generated during execution | Generated during execution |

The actual numerical results are generated when the notebook is executed.

# Validation Accuracy Comparison

The validation accuracy of all four models is plotted on the same graph.

This allows the learning behavior of the architectures to be compared across the five training epochs.

The plot can help identify:

- How quickly each model learns
- Changes in validation accuracy
- Differences between the architectures
- Potential overfitting behavior

# Validation Loss Comparison

Validation loss is also plotted for all four models.

The graph shows how the prediction error changes during training.

Comparing validation loss provides another way to analyze model behavior beyond accuracy.

# Test Accuracy Comparison

A bar chart is generated to compare the final test accuracy of:

- AlexNet
- VGG16
- ResNet50
- EfficientNetB0

This provides a simple visual comparison of their final classification performance.

# Training Time Comparison

Training time is recorded for each architecture.

This allows the practical to compare not only classification performance but also computational requirements.

A model with good accuracy may still require considerably more time or computational resources than another architecture.

# Parameter Comparison

The modified notebook also examines model complexity by comparing the number of parameters associated with the different architectures.

This provides additional information when analyzing the trade-off between model size and performance.

# Best Model Identification

The notebook automatically identifies the model with the highest test accuracy.

This is done using:

```python
best_model = results.loc[
    results["Test Accuracy (%)"].idxmax()
]
```

The selected model and its accuracy are then displayed.

The result is generated automatically from the actual execution rather than being manually specified.

# Sample Predictions

The notebook includes prediction visualization for sample test images.

Each displayed image can be associated with:

- Actual class
- Predicted class
- Prediction confidence

This provides a visual understanding of how the trained models classify individual images.

# Incorrect Prediction Analysis

Incorrect predictions are also analyzed.

The notebook identifies images where the predicted class differs from the actual class and visualizes them.

This helps investigate:

- Difficult image categories
- Similar-looking classes
- Misclassification patterns
- Examples where the model is uncertain

# Model Comparison Criteria

The four architectures are compared using multiple criteria rather than accuracy alone.

The main comparison factors are:

```text
Test Accuracy
       ↓
Test Loss
       ↓
Validation Accuracy
       ↓
Validation Loss
       ↓
Training Time
       ↓
Model Complexity
```

This provides a broader understanding of the differences between CNN architectures.

# Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **CIFAR-10**
- **AlexNet**
- **VGG16**
- **ResNet50**
- **EfficientNetB0**
- **Transfer Learning**
- **Deep Learning**
- **Computer Vision**

# Key Concepts Demonstrated

- Convolutional Neural Networks
- Image Classification
- CIFAR-10
- Image Preprocessing
- Image Resizing
- Data Augmentation
- Data Visualization
- TensorFlow Data Pipelines
- Batch Processing
- Prefetching
- CNN Feature Extraction
- Batch Normalization
- Global Average Pooling
- Dropout
- Transfer Learning
- ImageNet Pretrained Models
- Frozen Layers
- ReLU Activation
- Softmax Classification
- Categorical Cross-Entropy
- Adam Optimization
- Model Evaluation
- Validation Accuracy
- Validation Loss
- Test Accuracy
- Test Loss
- Training Time
- Model Complexity
- Model Comparison
- Error Analysis

# Project Structure

The Lab 7 folder contains the executed notebook and README file.

```text
DL_Lab/
├── lab1/
├── lab2/
├── lab3/
├── lab6/
└── lab7/
    ├── DL_A7.ipynb
    └── README.md
```

The CIFAR-10 dataset does not need to be manually stored inside the repository because TensorFlow downloads it automatically.

# How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DL_Lab
```

## 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

## 3. Open the Notebook

Open:

```text
lab7/DL_A7.ipynb
```

using VS Code, Jupyter Notebook, JupyterLab, or another compatible environment.

## 4. Run the Notebook

Run the cells sequentially.

The notebook will automatically download CIFAR-10 using TensorFlow if it is not already available.

The notebook then:

1. Loads CIFAR-10.
2. Selects the training and test subsets.
3. Visualizes sample images.
4. Analyzes class distribution.
5. Resizes the images.
6. Applies data augmentation.
7. Creates the TensorFlow data pipeline.
8. Builds AlexNet.
9. Builds VGG16.
10. Builds ResNet50.
11. Builds EfficientNetB0.
12. Compiles all four models.
13. Trains the models.
14. Evaluates the models.
15. Compares accuracy and loss.
16. Compares training time.
17. Analyzes model complexity.
18. Visualizes predictions.
19. Identifies the highest-accuracy model.

# Pretrained Model Weights

VGG16, ResNet50, and EfficientNetB0 use:

```python
weights="imagenet"
```

Therefore, TensorFlow may download the required pretrained ImageNet weights the first time the notebook is executed.

An internet connection may be required during the initial model setup.

# Results

The notebook generates the following outputs:

### Dataset Visualization

- Sample CIFAR-10 images
- Class distribution
- Augmented image examples

### Training Results

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

### Model Evaluation

- Test accuracy
- Test loss
- Training time

### Model Comparison

- Accuracy comparison
- Loss comparison
- Training-time comparison
- Model complexity comparison
- Best model identification

### Prediction Analysis

- Sample predictions
- Prediction confidence
- Incorrect predictions

The exact numerical results are generated from the executed notebook and depend on the training environment and model initialization.

# Limitations

The practical uses a subset of CIFAR-10 rather than the complete dataset:

```text
10,000 training images
2,000 test images
```

This was done to make the comparison of four architectures more manageable.

The models are also trained for only:

```text
5 epochs
```

Therefore, the results should be considered an experimental comparison rather than a fully optimized benchmark.

Additionally, VGG16, ResNet50, and EfficientNetB0 use frozen pretrained feature extractors, while AlexNet is implemented as a custom CNN. Therefore, the architectures are not being trained under exactly identical conditions.

# Future Improvements

The project can be extended by:

- Training on the complete CIFAR-10 dataset.
- Increasing the number of training epochs.
- Fine-tuning the pretrained VGG16 model.
- Fine-tuning the pretrained ResNet50 model.
- Fine-tuning the pretrained EfficientNetB0 model.
- Comparing different learning rates.
- Comparing different batch sizes.
- Adding Early Stopping.
- Adding learning-rate scheduling.
- Using more advanced data augmentation.
- Generating confusion matrices for each model.
- Comparing precision, recall, and F1-score.
- Performing detailed error analysis.
- Comparing parameter counts.
- Comparing inference time.
- Comparing memory requirements.
- Saving the best-performing model.
- Deploying the selected model as an image classification application.

# Key Takeaway

This practical provides a comparative study of four important CNN architectures on the same image classification problem.

The complete workflow can be summarized as:

```text
Same Dataset
      ↓
Same General Training Setup
      ↓
Different CNN Architectures
      ↓
Train
      ↓
Evaluate
      ↓
Visualize
      ↓
Compare
      ↓
Analyze
```

The practical demonstrates how different CNN architectures can be evaluated using multiple performance and computational criteria.

By comparing **AlexNet, VGG16, ResNet50, and EfficientNetB0**, the project provides practical experience with CNN architecture design, transfer learning, pretrained models, image classification, model evaluation, and performance analysis.

# Author

**Aadit Paranjape**

Deep Learning | Computer Vision | Neural Networks | Image Classification