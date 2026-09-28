# BERT-Tiny Based News Text Classification

## Project Overview

This project implements a lightweight Transformer-based text classification system using **BERT-tiny** and the **AG News dataset**.

The model reads a news article and predicts which of four categories it belongs to:

- World
- Sports
- Business
- Sci/Tech

A smaller BERT model is used instead of standard BERT so that the project can run on systems with limited RAM, GPU, and disk space.

---

## Objectives

- Understand how Transformer-based text classification works.
- Apply tokenization to real-world news text.
- Fine-tune a pretrained BERT-based model for classification.
- Evaluate the model using standard classification metrics.
- Visualize classification performance using a confusion matrix.
- Use the trained model to classify new unseen news articles.

---

## Dataset

The project uses the **AG News dataset**.

The dataset contains news articles belonging to four categories:

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

The dataset is loaded automatically using the Hugging Face `datasets` library.

### Dataset Used

To keep the experiment lightweight, a smaller subset of the complete AG News dataset is used:

- Training samples: 8,000
- Validation samples: 800
- Test samples: 2,000
- Number of classes: 4

The validation set is created by taking 10% of the selected training data.

---

## Model

The project uses:

**BERT-tiny**

Model:

```text
prajjwal1/bert-tiny
```

BERT-tiny is a compact pretrained Transformer model that is significantly smaller than standard BERT, making it more practical for experimentation on systems with limited computational resources.

### Model Pipeline

```text
News Article
      ↓
BERT Tokenizer
      ↓
Token IDs + Attention Mask
      ↓
BERT-tiny
      ↓
Classification Layer
      ↓
4 Output Classes
      ↓
Predicted News Category
```

---

## Methodology

### 1. Dataset Loading

The AG News dataset is loaded using the Hugging Face `datasets` library.

```python
dataset = load_dataset("fancyzhx/ag_news")
```

A smaller subset is selected to reduce memory usage and training time.

---

### 2. Data Sampling

The selected dataset consists of:

```text
Training:   8000 samples
Testing:    2000 samples
```

The training data is further divided into:

```text
Training:    7200 samples
Validation:   800 samples
```

---

### 3. Tokenization

Raw news articles cannot be directly processed by BERT.

The BERT tokenizer converts the text into token IDs and attention masks.

The maximum sequence length is limited to:

```text
128 tokens
```

This keeps memory consumption manageable during training.

---

### 4. Dynamic Padding

Padding is performed dynamically for each batch instead of padding every article to the maximum sequence length.

This reduces unnecessary memory usage during training.

---

### 5. BERT-tiny Fine-Tuning

The pretrained BERT-tiny model is fine-tuned for four-class news classification.

The classification layer produces four outputs corresponding to:

```text
World
Sports
Business
Sci/Tech
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | BERT-tiny |
| Learning Rate | `2e-5` |
| Batch Size | `8` |
| Epochs | `5` |
| Weight Decay | `0.01` |
| Maximum Sequence Length | `128` |
| Optimizer | AdamW through Hugging Face Trainer |
| Evaluation | After each epoch |
| Best Model Metric | F1 Score |

The implementation uses the Hugging Face `Trainer` API for model training and evaluation.

---

## Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

### Accuracy

Accuracy represents the proportion of news articles classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Precision measures how many of the articles predicted as a particular category actually belonged to that category.

### Recall

Recall measures how many of the actual articles belonging to a category were correctly identified.

### F1 Score

F1 Score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Visualizations

The notebook includes visualizations to analyze the dataset and model performance.

### AG News Class Distribution

A bar chart is used to visualize the number of training samples belonging to each news category.

### Confusion Matrix

A confusion matrix is generated after testing the model.

It shows:

- Correct predictions for each category.
- Categories that are confused with one another.
- Classification behavior across World, Sports, Business, and Sci/Tech.

---

## Classification Report

The notebook generates a detailed classification report containing precision, recall, and F1-score for:

```text
World
Sports
Business
Sci/Tech
```

This provides class-wise evaluation in addition to the overall test metrics.

---

## New News Prediction

The project also includes a function for classifying completely new news text.

Example:

```text
Apple announced a new generation of artificial intelligence
chips designed to improve the performance of its latest
computing devices and machine learning applications.
```

The model returns:

```text
Predicted Category: Sci/Tech
Confidence: XX.XX%
```

The exact confidence value depends on the trained model.

---

## Technologies Used

- Python
- PyTorch
- TensorFlow-compatible Python environment
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

---

## Installation

Install the required libraries using:

```bash
pip install transformers datasets scikit-learn torch accelerate matplotlib seaborn sentencepiece tiktoken
```

The additional tokenizer dependencies are included to support the BERT-tiny tokenizer.

---

## Project Structure

```text
lab8/
│
├── DL_A8.ipynb
└── README.md
```

The AG News dataset and pretrained BERT-tiny model do not need to be stored in the repository. They are downloaded automatically when the notebook is executed.

---

## How to Run

### 1. Open the Notebook

Open:

```text
DL_A8.ipynb
```

using VS Code with the Jupyter extension.

### 2. Install Dependencies

Run:

```bash
pip install transformers datasets scikit-learn torch accelerate matplotlib seaborn sentencepiece tiktoken
```

### 3. Run the Notebook

Execute the notebook cells sequentially.

The notebook will:

1. Load the AG News dataset.
2. Select the required dataset subset.
3. Create a validation set.
4. Display the dataset information.
5. Visualize the class distribution.
6. Load the BERT-tiny tokenizer.
7. Tokenize the news articles.
8. Load the pretrained BERT-tiny model.
9. Configure the training process.
10. Fine-tune the model for 5 epochs.
11. Evaluate the model on the test dataset.
12. Generate classification metrics.
13. Generate a confusion matrix.
14. Classify a new news article.

---

## Results

The notebook automatically displays the final test performance:

```text
Accuracy
Precision
Recall
F1 Score
```

It also generates a detailed classification report and confusion matrix.

The exact numerical results depend on the execution environment and training run, so the notebook output should be used as the final experimental result.

---

## Why BERT-Tiny?

Standard BERT models can require significant memory and computational resources.

BERT-tiny was selected because:

- It is much smaller than standard BERT.
- It requires fewer computational resources.
- It is easier to run on a personal computer.
- It reduces the computational cost of experimentation.
- It still demonstrates pretrained Transformer-based text classification.

This makes it suitable for an academic deep learning and NLP experiment.

---

## Applications

Transformer-based news classification can be used in:

- News categorization
- News recommendation systems
- Content filtering
- Information retrieval
- Media monitoring
- Automated news organization
- Text analytics
- Document classification

---

## Limitations

- A smaller subset of the complete AG News dataset is used.
- The model is limited to four predefined news categories.
- The maximum sequence length is limited to 128 tokens.
- BERT-tiny is smaller than larger Transformer models and may have lower representational capacity.
- Training results can vary depending on the hardware and execution environment.

---

## Future Improvements

The project can be improved by:

- Training on the complete AG News dataset.
- Increasing the maximum sequence length.
- Using a larger pretrained Transformer model.
- Performing hyperparameter tuning.
- Increasing the number of training epochs.
- Adding learning-rate scheduling.
- Comparing BERT-tiny with DistilBERT and BERT-base.
- Comparing different Transformer architectures.
- Deploying the classifier as a web application.
- Adding a user interface for real-time news classification.

---

## References

- **AG News Dataset:**  
  https://huggingface.co/datasets/fancyzhx/ag_news

- **Hugging Face Transformers:**  
  https://huggingface.co/docs/transformers

- **BERT:**  
  https://arxiv.org/abs/1810.04805

- **BERT-tiny:**  
  https://huggingface.co/prajjwal1/bert-tiny

---

## Project Summary

This project demonstrates how a pretrained Transformer model can be fine-tuned for multi-class text classification.

Using **BERT-tiny** and the **AG News dataset**, the system learns patterns in news articles and classifies them into **World, Sports, Business, or Sci/Tech** categories.

The lightweight implementation makes the project suitable for running on systems with limited computational resources while demonstrating the main concepts involved in Transformer-based NLP classification.

## Author

Aadit Paranjape