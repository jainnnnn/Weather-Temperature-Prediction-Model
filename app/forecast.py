import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Load model, scaler, and raw data
model = tf.keras.models.load_model("weather_rnn.keras")
scaler = joblib.load("dataset/scaler.pkl")
raw_df = pd.read_csv("dataset/weatherHistory.csv")

features = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)"
]

# Recreate daily aggregated dataframe (matching data_preprocessing.py)
raw_df["Date"] = pd.to_datetime(raw_df["Formatted Date"], utc=True).dt.date
daily_df = (
    raw_df.groupby("Date")[features]
    .mean()
    .reset_index()
)

# Scale all historical data
scaled_data = scaler.transform(daily_df[features].values)

# Take latest 7 days
current_sequence = scaled_data[-7:].copy()

predictions = []

# Predict next 7 days
for day in range(7):
    prediction = model.predict(
        current_sequence.reshape(1, 7, 3),
        verbose=0
    )[0][0]

    # Convert temperature to Celsius
    temp = prediction * scaler.data_range_[0] + scaler.data_min_[0]
    predictions.append(temp)

    # Carry forward latest humidity and wind speed
    new_day = current_sequence[-1].copy()
    new_day[0] = prediction
    current_sequence = np.vstack([current_sequence[1:], new_day])

# Create future dates
last_date = pd.to_datetime(daily_df["Date"].iloc[-1])
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Temperature (C)": predictions
})

print("\nNEXT 7 DAYS FORECAST")
print(forecast_df.to_string(index=False))

forecast_df.to_csv("forecast_results.csv", index=False)

# Plot historical and forecast temperatures
plt.figure(figsize=(12, 5))
recent = daily_df.tail(30)

plt.plot(
    pd.to_datetime(recent["Date"]),
    recent["Temperature (C)"],
    label="Historical Temperature"
)
plt.plot(
    future_dates,
    predictions,
    marker="o",
    linestyle="--",
    label="Predicted Temperature"
)

plt.title("7-Day Weather Temperature Forecast")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("7_day_forecast.png")
plt.show()

print("\nForecast completed successfully!")
