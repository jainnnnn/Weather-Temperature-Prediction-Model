
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
data = np.load("dataset/weather_sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

# Build SimpleRNN model
model = Sequential([
    SimpleRNN(64, activation="tanh",
              input_shape=(7, 3)),
    Dropout(0.2),
    Dense(1)
])

# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save("weather_rnn.keras")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("SimpleRNN Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

print("Model training completed!")
