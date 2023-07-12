# Matala 4 MN
# Erga 207829813
# Neta Cohen 325195774

import numpy as np

# Function to perform gradient descent
def gradient_descent(X, y, alpha, iterations):
    m = len(y)
    n = X.shape[1]
    w = np.zeros(n)
    b = 0

    for _ in range(iterations):
        y_pred = np.dot(X, w) + b
        error = y_pred - y
        gradient_w = (1/m) * np.dot(X.T, error)
        gradient_b = (1/m) * np.sum(error)
        w -= alpha * gradient_w
        b -= alpha * gradient_b

    return w, b

# Prepare the data
data = np.array([
    [4.9176, 1.0, 3.4720, 0.998, 1.0, 7, 4, 42, 3, 1, 0, 25.9],
    [5.0208, 1.0, 3.5310, 1.500, 2.0, 7, 4, 62, 1, 1, 0, 29.5],
    [4.5429, 1.0, 2.2750, 1.175, 1.0, 6, 3, 40, 2, 1, 0, 27.9],
    [4.5573, 1.0, 4.0500, 1.232, 1.0, 6, 3, 54, 4, 1, 0, 25.9],
    [5.0597, 1.0, 4.4550, 1.121, 1.0, 6, 3, 42, 3, 1, 0, 29.9],
    [3.8910, 1.0, 4.4550, 0.988, 1.0, 6, 3, 56, 2, 1, 0, 29.9],
    [5.8980, 1.0, 5.8500, 1.240, 1.0, 7, 3, 51, 2, 1, 1, 30.9],
    [5.6039, 1.0, 9.5200, 1.501, 0.0, 6, 3, 32, 1, 1, 0, 28.9],
    [16.4202, 2.5, 9.8000, 3.420, 2.0, 10, 5, 42, 2, 1, 1, 84.9],
    [14.4598, 2.5, 12.8000, 3.000, 2.0, 9, 5, 14, 4, 1, 1, 82.9],
    [5.8282, 1.0, 6.4350, 1.225, 2.0, 6, 3, 32, 1, 1, 0, 35.9],
    [5.3003, 1.0, 4.9883, 1.552, 1.0, 6, 3, 30, 1, 2, 0, 31.5],
    [6.2712, 1.0, 5.5200, 0.975, 1.0, 5, 2, 30, 1, 2, 0, 31.0],
    [5.9592, 1.0, 6.6660, 1.121, 2.0, 6, 3, 32, 2, 1, 0, 30.9],
    [5.0500, 1.0, 5.0000, 1.020, 0.0, 5, 2, 46, 4, 1, 1, 30.0],
    [5.6039, 1.0, 9.5200, 1.501, 0.0, 6, 3, 32, 1, 1, 0, 28.9],
    [8.2464, 1.5, 5.1500, 1.664, 2.0, 8, 4, 50, 4, 1, 0, 36.9],
    [6.6969, 1.5, 6.9020, 1.488, 1.5, 7, 3, 22, 1, 1, 1, 41.9],
    [7.7841, 1.5, 7.1020, 1.376, 1.0, 6, 3, 17, 2, 1, 0, 40.5],
    [9.0384, 1.0, 7.8000, 1.500, 1.5, 7, 3, 23, 3, 3, 0, 43.9],
    [5.9894, 1.0, 5.5200, 1.256, 2.0, 6, 3, 40, 4, 1, 1, 37.5],
    [7.5422, 1.5, 4.0000, 1.690, 1.0, 6, 3, 22, 1, 1, 0, 37.9],
    [8.7951, 1.5, 9.8900, 1.820, 2.0, 8, 4, 50, 1, 1, 1, 44.5],
    [6.0931, 1.5, 6.7265, 1.652, 1.0, 6, 3, 44, 4, 1, 0, 37.9],
    [8.3607, 1.5, 9.1500, 1.777, 2.0, 8, 4, 48, 1, 1, 1, 38.9],
    [8.1400, 1.0, 8.0000, 1.504, 2.0, 7, 3, 3, 1, 3, 0, 36.9],
    [9.1416, 1.5, 7.3262, 1.831, 1.5, 8, 4, 31, 4, 1, 0, 45.8],
    [12.0000, 1.5, 5.0000, 1.200, 2.0, 6, 3, 30, 3, 1, 1, 41.0]
])



# Split data into input (X) and output (y)
X = data[:, :-1]  # Take all columns except the last one
y = data[:, -1]  # Take the last column

# Normalize the input features (X)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Determine the number of data points for training (75% - 21 lines)
train_size = 21

# Split the data into training and testing sets
X_train = X[:train_size, :-1]  # Take all rows and all columns except the last one
y_train = y[:train_size]  # Take the last column for training
X_test = X[train_size:, :-1]  # Take all rows and all columns except the last one
y_test = y[train_size:]  # Take the last column for testing

# Train the linear regression model
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations for gradient descent
w, b = gradient_descent(X_train, y_train, alpha, iterations)

# Print the model details
print("Model Details from 75% of the given data:")
print("Weights (w):", w)
print("Bias (b):", b)

# Make predictions on the test set and compare with actual values
print("Predictions for last 25% of the data:")
for i in range(len(X_test)):
    example_input = X_test[i]
    predicted_output = np.dot(example_input, w) + b
    actual_output = y_test[i]
    print("Predicted :", predicted_output, "\tActual:", actual_output)

# Calculate mean squared error
mse = np.mean((y_test - (np.dot(X_test, w) + b)) ** 2)

# Print the mean squared error
print("Mean Squared Error:", mse)