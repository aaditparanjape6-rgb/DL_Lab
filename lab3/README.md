# Lab 3 - Forward Propagation, Backpropagation and Hyperparameter Analysis

## Objective

To implement forward propagation and backpropagation using TensorFlow/Keras and analyze the effect of different learning rates and the number of epochs on model performance.

## Dataset

The **MNIST handwritten digit dataset** is used for training and evaluation.

The images are reshaped from `28 × 28` pixels into `784` input features and the pixel values are normalized.

## Neural Network Architecture

The neural network consists of:

- Input layer: 784 features
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation

The model uses the Adam optimizer and sparse categorical cross-entropy loss.

## Experiments

### 1. Effect of Learning Rate

The model is trained using the following learning rates:

- `0.1`
- `0.01`
- `0.001`
- `0.0001`

The test accuracy for each learning rate is recorded and visualized using a learning-rate-versus-accuracy plot.

### 2. Effect of Number of Epochs

The model is trained using:

- 5 epochs
- 10 epochs
- 20 epochs

The resulting test accuracies are compared using an epochs-versus-accuracy plot.

### 3. Final Model

The final model is trained using:

- Learning rate: `0.001`
- Epochs: `20`
- Batch size: `128`
- Validation split: `20%`

Training and validation accuracy and loss are visualized to analyze model performance.

## Evaluation

The final model is evaluated using:

- Test loss
- Test accuracy
- Confusion matrix

The confusion matrix shows the classification performance for each of the ten MNIST digit classes.

## Visualizations

The notebook includes:

- Learning Rate vs Accuracy
- Epochs vs Accuracy
- Training vs Validation Accuracy
- Training vs Validation Loss
- Confusion Matrix

## Technologies Used

- Python
- Google Colab
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Files

| File | Description |
|------|-------------|
| `DL_A3.ipynb` | Complete implementation of the Lab 3 practical |

## How to Run

1. Open `DL_A3.ipynb` in Google Colab.
2. Run the cells sequentially.
3. Allow the MNIST dataset to download when required.
4. Observe the results of the learning-rate and epoch experiments.
5. Analyze the final model's accuracy, loss curves, and confusion matrix.

## Conclusion

This practical demonstrates the implementation and training of a neural network using TensorFlow/Keras. The experiments show how learning rate and the number of training epochs affect model performance, while the final evaluation uses accuracy, loss curves, and a confusion matrix to analyze the trained model.