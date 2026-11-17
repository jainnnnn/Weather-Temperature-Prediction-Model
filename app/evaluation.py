
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load test data
data = np.load("dataset/weather_sequences.npz")

X_test = data["X_test"]
y_test = data["y_test"]

# Load model and scaler
model = tf.keras.models.load_model("weather_rnn.keras")
scaler = joblib.load("dataset/scaler.pkl")

# Predict
predictions_scaled = model.predict(X_test).flatten()

# Convert scaled temperature back to Celsius
temp_min = scaler.data_min_[0]
temp_max = scaler.data_max_[0]

predictions = predictions_scaled * (temp_max - temp_min) + temp_min
actual = y_test * (temp_max - temp_min) + temp_min

# Metrics
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)

print("MODEL EVALUATION")
print("-----------------------")
print("RMSE:", rmse, "°C")
print("MAE:", mae, "°C")
print("R2 Score:", r2)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(actual[:100], label="Actual Temperature")
plt.plot(predictions[:100], label="Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test Days")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
