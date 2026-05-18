import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.DataFrame({
    'traffic_volume': [100, 150, 200, 250, 300],
    'speed': [60, 55, 50, 45, 40],
    'incidents': [0, 1, 0, 1, 1]
})

# Features and target
X = data[['speed', 'incidents']]
y = data['traffic_volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)

print("Traffic Prediction Model")
print("Predictions:", predictions)
print("Mean Squared Error:", mse)
