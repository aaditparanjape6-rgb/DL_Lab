# LSTM-Based Traffic Volume Forecasting

## Project Overview

This project implements an **LSTM (Long Short-Term Memory) based time-series forecasting model** to predict hourly traffic volume.

The project uses the **Metro Interstate Traffic Volume Dataset**, which contains historical traffic observations along with date and time information. The LSTM model learns temporal patterns from previous traffic observations and predicts the traffic volume for the next hour.

The project demonstrates how deep learning can be applied to **traffic forecasting, smart-city transportation management, and intelligent transportation systems**.

---

## Objectives

The main objectives of this project are:

- Analyze historical traffic volume data.
- Perform data cleaning and preprocessing.
- Convert date-time information into a chronological format.
- Normalize traffic-volume data.
- Create sequential input data for an LSTM model.
- Use the previous 24 hours to predict the next hour.
- Train an LSTM neural network for traffic forecasting.
- Evaluate the model using MAE, RMSE, and MAPE.
- Visualize training and validation performance.
- Compare actual and predicted traffic volume.
- Predict the traffic volume for the next hour.

---

## Dataset

### Metro Interstate Traffic Volume Dataset

The project uses the **Metro Interstate Traffic Volume Dataset**, which contains hourly traffic volume measurements recorded on an interstate highway.

Important columns in the dataset include:

- `date_time` – Date and time of the observation
- `traffic_volume` – Number of vehicles recorded during the observation period
- `weather_main` – Main weather condition
- `weather_description` – Detailed weather condition
- `temp` – Temperature
- `rain_1h` – Rainfall during the previous hour
- `snow_1h` – Snowfall during the previous hour
- `clouds_all` – Cloud coverage
- `holiday` – Holiday information

For this implementation, **`traffic_volume`** is used as the primary time-series variable.

The dataset is included in this repository as:

```text
Metro_Interstate_Traffic_Volume.csv
```

---

# Methodology

The complete forecasting pipeline is:

```text
Historical Traffic Data
        ↓
Data Cleaning
        ↓
Date-Time Conversion
        ↓
Chronological Sorting
        ↓
Traffic Volume Extraction
        ↓
Min-Max Normalization
        ↓
80% Training / 20% Testing
        ↓
24-Hour Sequence Creation
        ↓
LSTM Model
        ↓
Model Training
        ↓
Traffic Prediction
        ↓
Inverse Scaling
        ↓
Model Evaluation
        ↓
Visualization
        ↓
Next-Hour Forecast
```

---

## 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
```

The dataset shape and initial records are displayed to understand the structure of the data.

---

## 2. Data Cleaning

The `date_time` column is converted into a proper datetime format.

```python
df["date_time"] = pd.to_datetime(df["date_time"])
```

The observations are then sorted chronologically.

Duplicate timestamps are removed and rows with missing traffic-volume values are dropped.

This ensures that the data is properly ordered before it is used for time-series forecasting.

---

## 3. Exploratory Data Visualization

Several visualizations are generated to understand the traffic data.

### Historical Traffic Volume

The complete traffic-volume history is plotted against time to observe long-term traffic patterns.

### Traffic Volume Distribution

A histogram is used to visualize the distribution of traffic-volume values.

### Average Traffic by Hour

Average traffic volume is calculated for each hour of the day to identify daily traffic patterns.

### Average Traffic by Day of Week

Average traffic volume is analyzed across different days of the week.

### 24-Hour Rolling Average

A rolling average is used to smooth short-term fluctuations and highlight broader traffic trends.

These visualizations provide an understanding of the temporal characteristics of the dataset before model training.

---

# Data Preprocessing

## Traffic Volume Extraction

For this implementation, only the traffic-volume column is used for forecasting.

```python
traffic = df[["traffic_volume"]].values
```

This makes the implementation a **univariate time-series forecasting problem**.

---

## Data Normalization

Traffic volume is normalized using `MinMaxScaler`.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_traffic = scaler.fit_transform(traffic)
```

Scaling the data to a range between 0 and 1 helps the neural network train more effectively.

---

# Train-Test Split

The dataset is divided chronologically:

```text
80% → Training Data
20% → Testing Data
```

Random shuffling is avoided because time-series data must preserve its chronological order.

The model learns from historical observations and is then evaluated on future observations.

---

# Sequence Creation

The model uses the previous **24 hours** of traffic observations to predict the traffic volume for the following hour.

```text
Previous 24 Hours
       ↓
      LSTM
       ↓
Next Hour Traffic Volume
```

For example:

```text
Hour 1  ─┐
Hour 2   │
Hour 3   │
...      ├──→ LSTM ──→ Hour 25
Hour 23  │
Hour 24 ─┘
```

A 24-hour sequence allows the model to learn short-term and daily traffic patterns.

The resulting LSTM input has the format:

```text
Samples × Time Steps × Features
```

where:

- Time Steps = 24
- Features = 1

---

# LSTM Architecture

The implemented model uses two LSTM layers followed by fully connected layers.

```text
Input: Previous 24 Hours
          ↓
LSTM Layer – 64 Units
          ↓
Dropout – 20%
          ↓
LSTM Layer – 32 Units
          ↓
Dropout – 20%
          ↓
Dense Layer – 32 Neurons, ReLU
          ↓
Dropout – 10%
          ↓
Output Layer – 1 Neuron
          ↓
Predicted Traffic Volume
```

### Model Structure

```python
LSTM(64, return_sequences=True)
Dropout(0.20)

LSTM(32)
Dropout(0.20)

Dense(32, activation="relu")
Dropout(0.10)

Dense(1)
```

The additional dropout layer helps reduce overfitting, while the dense layer transforms the learned temporal representation into the final traffic-volume prediction.

---

# Why LSTM?

LSTM networks are specifically designed for **sequential and time-dependent data**.

Traffic volume at one point in time is related to traffic conditions during previous hours. An LSTM can retain useful information from previous time steps through its internal memory and gating mechanisms.

This makes LSTM suitable for forecasting problems such as:

- Traffic volume prediction
- Weather forecasting
- Stock and financial time-series forecasting
- Energy consumption prediction
- Demand forecasting

---

# Model Compilation

The model is compiled using the **Adam optimizer** and **Mean Squared Error (MSE)** loss.

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)
```

MAE is also tracked during training as an additional performance metric.

---

# Model Training

The model is trained using:

```text
Epochs: 40
Batch Size: 32
Validation Split: 10%
Optimizer: Adam
Loss: Mean Squared Error
```

Early stopping is used during training.

```python
EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)
```

Early stopping terminates training when the validation loss stops improving and restores the best-performing model weights.

---

# Evaluation Metrics

The model is evaluated using three main metrics.

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted traffic volume.

```text
MAE = Average(|Actual - Predicted|)
```

A lower MAE indicates that predictions are closer to the actual traffic volume.

---

## Root Mean Squared Error (RMSE)

RMSE gives greater importance to larger prediction errors.

```text
RMSE = √(Average((Actual - Predicted)²))
```

A lower RMSE indicates fewer large prediction errors.

---

## Mean Absolute Percentage Error (MAPE)

MAPE expresses prediction error as a percentage.

```text
MAPE =
Average(|Actual - Predicted| / Actual) × 100
```

Zero-valued actual observations are excluded from the MAPE calculation because division by zero makes MAPE undefined.

---

# Prediction Process

After training, the model predicts traffic volume for the test dataset.

The predictions are initially in the normalized 0–1 range.

They are converted back to the original traffic-volume scale using:

```python
predicted_traffic = scaler.inverse_transform(predictions)
```

The actual test values are also converted back to their original scale for comparison.

---

# Visualizations

The project generates several visualizations.

## 1. Historical Traffic Volume

Shows traffic volume across the dataset over time.

## 2. Traffic Volume Distribution

Shows the distribution of traffic-volume values.

## 3. Average Traffic by Hour

Shows the average traffic volume for each hour of the day.

## 4. Average Traffic by Day of Week

Shows how average traffic varies across different days.

## 5. 24-Hour Rolling Average

Shows the smoothed traffic-volume trend using a 24-hour rolling window.

## 6. Training vs Validation Loss

Shows the model's training and validation loss across epochs.

This can be used to identify learning behavior and potential overfitting.

## 7. Training vs Validation MAE

Shows the change in mean absolute error during model training.

## 8. Actual vs Predicted Traffic

Compares the actual traffic volume with the traffic volume predicted by the LSTM model.

## 9. Prediction Error

Visualizes the difference between actual and predicted traffic values.

## 10. Next-Hour Forecast

The trained model uses the most recent 24 hours of traffic data to generate a prediction for the next hour.

---

# Results

The notebook calculates and displays:

```text
MAE
RMSE
MAPE
```

The exact values depend on the training run, TensorFlow/Keras version, hardware, and execution environment.

The project also generates an **Actual vs Predicted Traffic Volume** graph to visually evaluate the forecasting performance.

No fixed numerical result is included here because the model results are generated during execution.

---

# Real-World Applications

Traffic forecasting can support:

- Smart-city transportation systems
- Traffic congestion monitoring
- Route planning
- Intelligent transportation systems
- Traffic signal optimization
- Infrastructure planning
- Emergency response planning
- Public transportation management

Forecasting expected traffic conditions can help transportation systems make better-informed operational decisions.

---

# Future Improvements

The current implementation primarily uses historical traffic volume as the input variable.

The model can be extended into a **multivariate LSTM** by incorporating additional variables such as:

- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather conditions
- Holidays
- Day of the week
- Hour of the day

This would allow the model to learn relationships between traffic volume and external factors.

Other possible improvements include:

- Bidirectional LSTM
- GRU networks
- CNN-LSTM
- Attention mechanisms
- Transformer-based forecasting
- Hyperparameter optimization
- Longer forecasting horizons
- Multi-step traffic forecasting

---

# Future Scope

A more advanced version of this project could forecast traffic volume for multiple future hours instead of only the next hour.

The system could also be integrated with a transportation dashboard:

```text
Current Traffic
      ↓
Predicted Traffic
      ↓
Congestion Level
      ↓
Traffic Alert
      ↓
Suggested Route
```

Such an extension could make the system more suitable for a real-world intelligent transportation platform.

---

# Project Highlights

- Uses **Deep Learning** for time-series forecasting.
- Implements an **LSTM neural network**.
- Uses historical traffic observations as sequential input.
- Preserves chronological ordering during training and testing.
- Uses a **24-hour lookback window**.
- Uses Min-Max normalization.
- Uses early stopping during training.
- Evaluates predictions using MAE, RMSE, and MAPE.
- Includes multiple exploratory data visualizations.
- Provides training and validation performance plots.
- Provides actual vs predicted traffic visualization.
- Includes prediction error analysis.
- Generates a next-hour traffic forecast.
- Has potential applications in smart cities and intelligent transportation systems.

---

# Technologies Used

- **Python**
- **Pandas** – Data loading and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Scikit-learn** – Data scaling and evaluation metrics
- **TensorFlow / Keras** – LSTM model development
- **Jupyter Notebook** – Development environment

---

# Project Structure

```text
DL_Lab/
│
├── lab1/
├── lab2/
├── lab3/
│
├── lab4/
│   ├── DL_A4.ipynb
│   ├── Metro_Interstate_Traffic_Volume.csv
│   └── README.md
│
├── lab6/
├── lab7/
│
└── .gitignore
```

---

# Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

# How to Run

### 1. Clone the repository

```bash
git clone https://github.com/aaditparanjape6-rgb/DL_Lab.git
```

### 2. Navigate to the project

```bash
cd DL_Lab
```

### 3. Open the Lab 4 notebook

Open:

```text
lab4/DL_A4.ipynb
```

using Jupyter Notebook or VS Code.

### 4. Ensure the dataset is available

The notebook expects:

```text
Metro_Interstate_Traffic_Volume.csv
```

in the appropriate project directory.

### 5. Run the notebook

Run the cells sequentially.

The notebook will:

- Load the dataset
- Clean the data
- Convert date-time values
- Sort observations chronologically
- Visualize traffic patterns
- Normalize traffic volume
- Split the data chronologically
- Create 24-hour sequences
- Build the LSTM model
- Train the model
- Generate predictions
- Calculate MAE, RMSE, and MAPE
- Generate visualization plots
- Predict the next hour's traffic volume

---

# Project Information

**Project Type:** Deep Learning / Machine Learning

**Domain:** Smart Cities & Intelligent Transportation Systems

**Task:** Time-Series Forecasting

**Model:** Long Short-Term Memory (LSTM)

**Dataset:** Metro Interstate Traffic Volume

**Prediction Target:** Hourly Traffic Volume

**Lookback Window:** 24 Hours

---

# Conclusion

This project demonstrates how an **LSTM-based deep learning model** can be used for hourly traffic-volume forecasting.

By using the previous 24 hours of traffic observations as sequential input, the model learns temporal patterns and generates predictions for future traffic volume.

The project combines time-series preprocessing, exploratory data analysis, LSTM-based deep learning, model evaluation, and visualization to demonstrate a practical application of deep learning in intelligent transportation systems.

# Author

Aadit Paranjape