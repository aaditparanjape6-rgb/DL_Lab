# Lab 6 – CNN for Image Classification using Cassava Leaf Dataset

## Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras for classifying cassava leaf images into different disease categories.

The project demonstrates an end-to-end image classification workflow including dataset loading, image preprocessing, data visualization, data augmentation, CNN architecture design, model training, validation, performance evaluation, disease prediction, and model saving.

The images are resized to **128 × 128 pixels** and processed as RGB images.

## Objective

The main objectives of this practical are:

- Implement a Convolutional Neural Network for image classification.
- Load and preprocess an image dataset using TensorFlow/Keras.
- Perform data visualization and analysis.
- Apply image augmentation techniques.
- Train a CNN model for cassava leaf disease classification.
- Analyze training and validation performance.
- Evaluate the model using a confusion matrix and classification report.
- Visualize incorrect predictions.
- Predict the class of a new leaf image.
- Save the trained CNN model.

## Machine Learning Pipeline

```text
Cassava Leaf Image Dataset
          ↓
Load Images from Folders
          ↓
80% Training / 20% Validation
          ↓
Resize Images to 128 × 128
          ↓
Data Visualization
          ↓
Data Augmentation
          ↓
Pixel Rescaling
          ↓
Convolutional Neural Network
          ↓
Model Training
          ↓
Accuracy and Loss Analysis
          ↓
Validation Evaluation
          ↓
Confusion Matrix
          ↓
Classification Report
          ↓
Sample Predictions
          ↓
Incorrect Prediction Analysis
          ↓
New Leaf Image Prediction
          ↓
Save Trained Model
```

## Dataset

The project uses the **Cassava Leaf Disease Dataset**.

The dataset is organized into separate folders according to the class labels:

```text
data/
├── cbb/
├── cbsd/
├── cgm/
├── cmd/
└── healthy/
```

TensorFlow's `image_dataset_from_directory()` automatically identifies the folder names and uses them as class labels.

The dataset is divided into:

- **80% training data**
- **20% validation data**

A fixed random seed of `42` is used to make the train-validation split reproducible.

### Dataset Used for the Practical

To reduce training time and local storage requirements, a smaller balanced subset of the dataset is used.

The working dataset contains:

- **300 images per class**
- **5 classes**
- **Approximately 1500 images total**

The dataset is kept locally and is **not uploaded to GitHub** because of its size.

## Image Preprocessing

### Image Resizing

All images are resized to:

```text
128 × 128 × 3
```

where:

- `128` = image height
- `128` = image width
- `3` = RGB color channels

Using a fixed image size ensures that all images have the same dimensions before being passed to the CNN.

### Pixel Rescaling

The model uses:

```python
layers.Rescaling(1.0 / 255)
```

to convert pixel values from `0–255` to `0–1`.

This normalizes the input values before they are processed by the neural network.

## Data Visualization

Several visualizations are included in the notebook to understand the dataset before and during training.

### Sample Images

A grid of sample images is displayed from the training dataset.

This provides a visual understanding of the different cassava leaf classes and the type of images being provided to the model.

### Class Distribution

A bar chart is generated showing the number of images available in each class.

This helps verify the distribution of images across the different disease categories.

### Pixel Intensity Distribution

A histogram of pixel intensities is generated for a sample image.

This provides a basic understanding of the distribution of pixel values in the input images.

## Data Augmentation

Data augmentation is applied to the training images to create slightly modified versions of the original images.

The project uses:

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.08)
layers.RandomZoom(0.10)
```

### Augmentation Techniques

| Technique | Purpose |
|---|---|
| Random Flip | Creates horizontally flipped image variations |
| Random Rotation | Changes the orientation of images slightly |
| Random Zoom | Creates images at slightly different scales |

Data augmentation increases the variety of training samples and can help the model generalize better to unseen images.

The notebook also visualizes augmented versions of sample images.

## Dataset Pipeline Optimization

The TensorFlow dataset pipeline is optimized using:

```python
tf.data.AUTOTUNE
```

along with:

```python
cache()
shuffle()
prefetch()
```

Caching keeps the dataset available after the first pass through the data, shuffling changes the order of training samples, and prefetching prepares upcoming batches while the model processes the current batch.

# CNN Architecture

The project uses a Sequential Convolutional Neural Network consisting of four convolutional blocks followed by classification layers.

```text
Input Image
128 × 128 × 3
       ↓
Data Augmentation
       ↓
Rescaling
       ↓
Conv2D — 32 Filters
       ↓
Batch Normalization
       ↓
MaxPooling2D
       ↓
Conv2D — 64 Filters
       ↓
Batch Normalization
       ↓
MaxPooling2D
       ↓
Conv2D — 128 Filters
       ↓
Batch Normalization
       ↓
MaxPooling2D
       ↓
Dropout
       ↓
Conv2D — 256 Filters
       ↓
Batch Normalization
       ↓
MaxPooling2D
       ↓
Dropout
       ↓
Global Average Pooling
       ↓
Dense — 128 Neurons
       ↓
Dropout
       ↓
Softmax Output
```

The number of convolution filters progressively increases:

```text
32 → 64 → 128 → 256
```

This allows deeper layers to learn increasingly complex visual features from the leaf images.

## Convolutional Layers

### Convolution Block 1

The first convolution layer uses:

- 32 filters
- 3 × 3 kernel
- ReLU activation

The first layer learns basic visual features such as edges, lines, and simple textures.

### Convolution Block 2

The second convolution layer uses:

- 64 filters
- 3 × 3 kernel
- ReLU activation

With more filters, the network can learn more detailed patterns from the images.

### Convolution Block 3

The third convolution layer uses:

- 128 filters
- 3 × 3 kernel
- ReLU activation

At this stage, the network can learn more complex shapes and textures.

### Convolution Block 4

The fourth convolution layer uses:

- 256 filters
- 3 × 3 kernel
- ReLU activation

The deeper layer extracts high-level visual features that are useful for distinguishing between disease classes.

## Max Pooling

Each convolution block is followed by:

```python
layers.MaxPooling2D()
```

Max pooling reduces the spatial dimensions of the feature maps while retaining important features.

This reduces computational requirements and helps the network focus on important visual features.

## Batch Normalization

Batch Normalization is included after the convolutional layers.

It helps stabilize the activations during training and can make neural network training more consistent.

## Dropout

Dropout is used at multiple points in the network.

The model uses dropout rates of:

- 0.25
- 0.30
- 0.40

Dropout randomly disables a portion of neurons during training, helping reduce overfitting.

## Global Average Pooling

Instead of directly using a Flatten layer, the modified CNN uses:

```python
layers.GlobalAveragePooling2D()
```

Global Average Pooling converts each feature map into a single average value.

This reduces the number of parameters passed to the Dense layer and provides a compact representation of the extracted features.

## Dense Layer

The classification section contains a Dense layer with:

- 128 neurons
- ReLU activation

This layer combines the features extracted by the convolutional layers before the final classification.

## Output Layer

The final layer is:

```python
layers.Dense(
    num_classes,
    activation="softmax"
)
```

The number of output neurons is determined dynamically from the number of classes.

Softmax converts the output into probability values for each class.

The class with the highest probability is selected as the model's prediction.

# Model Compilation

The CNN is compiled using the Adam optimizer:

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer

**Adam** is used to update the model's weights during training.

The learning rate used is:

```text
0.001
```

### Loss Function

**Sparse Categorical Cross-Entropy** is used because the disease labels are represented as integer class values.

### Evaluation Metric

**Accuracy** measures the proportion of images that are classified correctly.

# Model Training

The CNN is trained for:

**5 epochs**

Training is performed using the training dataset with the validation dataset supplied to `model.fit()`.

During training:

1. Images are passed through the CNN.
2. The model generates predictions.
3. The loss between predictions and actual labels is calculated.
4. Backpropagation updates the model weights.
5. The process is repeated for multiple epochs.
6. Validation performance is measured after each epoch.

# Training and Validation Accuracy

The notebook plots both training and validation accuracy using the training history.

The resulting curves show how the model's classification accuracy changes during training.

Comparing training and validation accuracy can help identify possible overfitting or underfitting.

# Training and Validation Loss

The notebook also plots training and validation loss.

Loss represents how far the model's predictions are from the correct labels.

Comparing training and validation loss provides additional information about model learning and generalization.

# Model Evaluation

After training, the model is evaluated on the validation dataset.

The notebook reports:

- Validation Loss
- Validation Accuracy
- Validation Accuracy in percentage

This provides a quantitative measure of the model's performance on images that were not used during training.

# Generating Predictions

The trained model generates predictions for images from the validation dataset.

The model returns probability values for each class.

The predicted class is obtained using:

```python
predicted_labels = np.argmax(
    predictions,
    axis=1
)
```

`argmax` selects the class with the highest predicted probability.

# Confusion Matrix

A confusion matrix is generated using the actual and predicted class labels.

The confusion matrix shows how images from each actual class are classified by the model.

It helps identify:

- Correct classifications
- Incorrect classifications
- Classes that are frequently confused with one another

# Classification Report

The project generates a classification report containing:

- Precision
- Recall
- F1-score
- Support

This allows the performance of individual disease classes to be analyzed.

# Sample Predictions

The notebook displays sample validation images along with:

- Actual class
- Predicted class
- Prediction confidence

The confidence value is obtained from the probability assigned to the predicted class.

# Incorrect Prediction Analysis

The notebook identifies incorrectly classified images from a validation batch.

The incorrectly classified images are displayed along with their:

- Actual class
- Predicted class

This provides a visual method for analyzing challenging images and potentially similar disease classes.

# Predicting a New Leaf Image

The trained CNN can also be used to classify a separate leaf image.

The notebook supports an image named:

```text
test_leaf.jpeg
```

The image is resized to 128 × 128, converted into an array, and given a batch dimension before being passed to the model.

The model outputs probabilities for each class.

The class with the highest probability is selected, and its probability is displayed as the prediction confidence.

# Model Saving

The trained CNN model is saved in Keras format using:

```python
model.save("cnn_image_classifier.keras")
```

The saved model can be loaded later without retraining the CNN.

The generated model file is kept locally and is not included in the GitHub repository.

# Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Convolutional Neural Networks**
- **Computer Vision**
- **Deep Learning**

# Key Concepts Demonstrated

- Convolutional Neural Networks
- Image Classification
- Image Preprocessing
- Image Resizing
- Pixel Normalization
- Data Augmentation
- Data Visualization
- Convolutional Layers
- Feature Extraction
- Max Pooling
- Batch Normalization
- Global Average Pooling
- Dense Layers
- ReLU Activation
- Softmax Activation
- Dropout
- Forward Propagation
- Backpropagation
- Adam Optimization
- Sparse Categorical Cross-Entropy
- Model Training
- Validation
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score
- Prediction Confidence
- Model Saving

# Project Structure

The GitHub repository contains the notebook and README, while the large image dataset remains local.

```text
DL_Lab/
├── lab1/
├── lab2/
├── lab3/
└── lab6/
    ├── DL_A6.ipynb
    └── README.md
```

### Local Dataset Structure

The dataset used for running the notebook is kept locally:

```text
lab6/
├── DL_A6.ipynb
├── README.md
└── data/
    ├── cbb/
    ├── cbsd/
    ├── cgm/
    ├── cmd/
    └── healthy/
```

The `data/` directory is excluded from GitHub because of its size.

# How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DL_Lab
```

## 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## 3. Prepare the Dataset

Place the cassava leaf image dataset inside:

```text
lab6/data/
```

Each class should have its own folder:

```text
data/
├── cbb/
├── cbsd/
├── cgm/
├── cmd/
└── healthy/
```

## 4. Open the Notebook

Open:

```text
lab6/DL_A6.ipynb
```

using Jupyter Notebook, JupyterLab, Google Colab, or VS Code.

## 5. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Load the dataset.
2. Display sample images.
3. Analyze class distribution.
4. Visualize pixel intensities.
5. Apply data augmentation.
6. Build the CNN.
7. Train the model for 5 epochs.
8. Plot accuracy and loss.
9. Evaluate the validation dataset.
10. Generate a confusion matrix.
11. Generate a classification report.
12. Display sample predictions.
13. Display incorrect predictions.
14. Predict a new leaf image.
15. Save the trained model.

# Results

The notebook generates the following outputs after execution:

- Sample image visualization
- Class distribution chart
- Pixel intensity histogram
- Augmented image visualization
- CNN model summary
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix
- Classification report
- Sample predictions
- Prediction confidence
- Incorrect prediction visualization
- New leaf image prediction

The actual numerical performance depends on the dataset, training run, hardware, and model initialization.

# Limitations

The model's performance depends on factors such as:

- Dataset quality
- Number of images per class
- Image diversity
- Lighting conditions
- Background variations
- Quality of disease labels

The current practical uses a validation split rather than a completely separate test dataset.

Therefore, validation accuracy should not automatically be interpreted as definitive real-world performance.

The model is intended as a deep learning classification demonstration and should not be treated as a replacement for expert agricultural or plant-pathology diagnosis.

# Future Improvements

The project can be extended by:

- Using a separate test dataset
- Increasing the size and diversity of the training dataset
- Handling class imbalance
- Adding more advanced augmentation techniques
- Experimenting with different CNN architectures
- Using transfer learning
- Comparing models such as MobileNet, ResNet, or EfficientNet
- Performing hyperparameter tuning
- Using learning-rate scheduling
- Adding early stopping
- Increasing the number of training epochs
- Performing more detailed error analysis
- Deploying the model as a web or mobile application
- Building a real-time leaf disease detection system

# Key Takeaway

This practical demonstrates how a **Convolutional Neural Network can be used to classify cassava leaf images based on learned visual features**.

The complete workflow covers:

**Load → Preprocess → Visualize → Augment → Extract Features → Train → Validate → Evaluate → Predict → Save**

The project provides practical experience with the fundamental components of image classification and deep learning, while also demonstrating how visualization and evaluation techniques can be used to understand CNN performance beyond simple accuracy.

## Author

**Aadit Paranjape**

Deep Learning | Computer Vision | Neural Networks | Image Classification