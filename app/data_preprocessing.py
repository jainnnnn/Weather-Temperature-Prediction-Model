
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# 1. Load dataset
df = pd.read_csv("dataset/weatherHistory.csv")

# 2. Convert date
df["Date"] = pd.to_datetime(df["Formatted Date"].str[:10])

# 3. Select required columns
df = df[[
    "Date",
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)"
]]

# 4. Convert hourly data into daily averages
daily_df = df.groupby("Date").mean().reset_index()

# 5. Handle missing values
daily_df = daily_df.dropna().reset_index(drop=True)

print("Daily dataset shape:", daily_df.shape)
print(daily_df.head())

# 6. Select input features
features = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)"
]

data = daily_df[features].values

# 7. Normalize data
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)
import joblib
joblib.dump(scaler, "dataset/scaler.pkl")

# 8. Create sequences
sequence_length = 7

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(scaled_data[i-sequence_length:i])

    # Target: next day's temperature
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print("\nSequence Information:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 9. Split data chronologically
total_samples = len(X)

train_end = int(total_samples * 0.70)
val_end = int(total_samples * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Testing Shape:", X_test.shape)

# 10. Save processed arrays
np.savez(
    "dataset/weather_sequences.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    scaled_data=scaled_data
)

print("\nPreprocessing completed successfully!")
