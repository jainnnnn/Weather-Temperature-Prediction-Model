# Weather Temperature Prediction using SimpleRNN

A beginner-friendly deep learning project that uses a **Simple Recurrent
Neural Network (SimpleRNN)** to predict daily temperature from
historical weather observations.

## Project overview

The original dataset contains hourly weather observations. This project
converts those observations into daily averages and uses the previous
**7 days** of weather data to predict the next day's temperature.

### Inputs

-   Temperature (°C)
-   Humidity
-   Wind speed (km/h)

### Target

-   Next day's temperature (°C)

## Project workflow

1.  Load and inspect the weather dataset.
2.  Check missing values and prepare the relevant columns.
3.  Convert hourly observations into daily averages.
4.  Normalize the input features using `MinMaxScaler`.
5.  Create sequences containing the previous 7 days.
6.  Split the data chronologically into training, validation, and test
    sets.
7.  Train a SimpleRNN model with a Dropout layer.
8.  Evaluate predictions using RMSE, MAE, and R².
9.  Generate a recursive 7-day temperature forecast.
10. Visualize the temperature trend, training loss, actual vs. predicted
    values, and forecast.

## Model architecture

``` text
Input: 7 days × 3 weather features
              |
        SimpleRNN (64 units)
              |
         Dropout (0.2)
              |
          Dense (1)
              |
   Predicted next-day temperature
```

The model uses: - **Optimizer:** Adam - **Loss:** Mean Squared Error
(MSE) - **Metric during training:** Mean Absolute Error (MAE) - **Early
stopping:** Monitors validation loss and restores the best weights.

## Project structure

``` text
weather_temperature_rnn/
│
├── dataset/
│   ├── your_weather_dataset.csv
│   ├── daily_weather.csv
│   ├── weather_sequences.npz
│   └── scaler.pkl
│
├── data_preprocessing.py
├── model.py
├── evaluation.py
├── forecast.py
├── Weather_Temperature_SimpleRNN.ipynb
├── weather_rnn.keras
├── requirements.txt
│
├── temperature_trend.png
├── loss_curve.png
├── actual_vs_predicted.png
├── 7_day_forecast.png
└── forecast_results.csv
```

*Your exact filenames and generated outputs may differ. The notebook may
save its outputs in an `outputs/` folder.*

## Requirements

-   Python 3.10--3.12 (use a TensorFlow-compatible version)
-   TensorFlow
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Joblib
-   Jupyter / VS Code Jupyter extension

## Setup and run

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create and activate a virtual environment (Windows PowerShell)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

``` cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

``` powershell
python -m pip install -r requirements.txt
```

If Jupyter cannot find the virtual environment as a kernel:

``` powershell
python -m pip install ipykernel
```

### 4. Add the dataset

Place the downloaded CSV file inside the `dataset/` folder.

Update the CSV filename/path in the relevant Python script and notebook
to match your actual dataset filename. For example:

``` python
CSV_PATH = "dataset/your_weather_dataset.csv"
```

### 5. Run the Python scripts

Run the scripts from the project root folder, in this order:

``` powershell
python data_preprocessing.py
python model.py
python evaluation.py
python forecast.py
```

### 6. Run the Jupyter notebook (alternative)

Open `Weather_Temperature_SimpleRNN.ipynb` in VS Code, select the
`.venv` Python kernel, update the dataset path, and choose **Run All**.

The notebook provides an end-to-end workflow. You can use the notebook
instead of running the scripts separately.

## Evaluation

The model is evaluated on the held-out test set using:

  -----------------------------------------------------------------------
  Metric                              Meaning
  ----------------------------------- -----------------------------------
  RMSE                                Root Mean Squared Error; penalizes
                                      larger errors more strongly

  MAE                                 Mean Absolute Error; average
                                      absolute prediction error

  R²                                  Indicates how much of the variation
                                      in the target is explained by the
                                      model
  -----------------------------------------------------------------------

An earlier run of the Python-script workflow produced the following
results:

  Metric       Result
  -------- ----------
  RMSE       1.919 °C
  MAE        1.428 °C
  R²           0.9485

These are example results from that run. Your results may differ
depending on the dataset, preprocessing, split, and random
initialization. The notebook uses a training-only scaler fit to reduce
data leakage, so its metrics may differ from the earlier script run.

## Forecasting note

The 7-day forecast is recursive: each predicted temperature is fed into
the next prediction step. Since future humidity and wind speed are not
known, the forecast carries forward the latest available values for
those features. Therefore, treat the multi-day forecast as an
approximate demonstration rather than a full weather forecast.

## Outputs

The project can produce: - Daily temperature trend plot - Training and
validation loss plot - Actual vs. predicted temperature plot - 7-day
forecast plot - CSV file containing forecast values - Saved trained
model (`.keras`) and scaler (`.pkl`)

## Learning objectives

-   Understand time-series data preparation.
-   Learn normalization and sliding-window sequence creation.
-   Build and train a basic recurrent neural network.
-   Evaluate regression predictions.
-   Generate recursive forecasts and visualize results.

## Disclaimer

This is an educational machine-learning project. It is not intended for
operational weather forecasting or safety-critical decisions.
