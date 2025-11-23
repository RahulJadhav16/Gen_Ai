import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Prepare data
X_raw = np.array([500, 1000, 1500, 2000], dtype=float)
y_raw = np.array([25, 50, 70, 90], dtype=float)

# Normalize (scale)
X = X_raw / 2000
y = y_raw / 100

# Build model
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train and record history
history = model.fit(X, y, epochs=3000, verbose=False)

pred = model.predict(np.array([1200]) / 2000) 

print(pred * 100) # convert back to original scale

# Plot loss curve
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
