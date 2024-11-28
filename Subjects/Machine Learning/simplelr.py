# Data points
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Independent variable
y = [1.5, 3.7, 2.8, 4.5, 6.1, 7.2, 7.8, 9.4, 10.1, 11.3]  # Dependent variable

# Calculate means of X and y
mean_x = sum(X) / len(X)
mean_y = sum(y) / len(y)

# Calculate slope (m) and intercept (b) using least squares method
numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(len(X)))
denominator = sum((X[i] - mean_x) ** 2 for i in range(len(X)))
m = numerator / denominator  # Slope
b = mean_y - m * mean_x  # Intercept

# Predict function
def predict(x):
    return m * x + b

# Calculate predictions and evaluation metrics
y_pred = [predict(x) for x in X]
mse = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y))) / len(y)
total_variance = sum((yi - mean_y) ** 2 for yi in y)
explained_variance = sum((y_pred[i] - mean_y) ** 2 for i in range(len(y)))
r_squared = explained_variance / total_variance

# Display results
print("Linear Regression Model:")
print(f"  Slope (m): {m:.2f}")
print(f"  Intercept (b): {b:.2f}")
print("\nModel Evaluation:")
print(f"  Mean Squared Error (MSE): {mse:.2f}")
print(f"  R-squared (R²): {r_squared:.2f}")

# Print predictions
print("\nPredictions:")
for xi, yi, yi_pred in zip(X, y, y_pred):
    print(f"  X={xi}, Actual Y={yi:.2f}, Predicted Y={yi_pred:.2f}")







# Sample data
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Independent variable
y = [1.5, 3.7, 2.8, 4.5, 6.1, 7.2, 7.8, 9.4, 10.1, 11.3]  # Dependent variable

# Function to calculate mean
def mean(values):
    return sum(values) / len(values)

# Function to calculate coefficients
def calculate_coefficients(X, y):
    x_mean = mean(X)
    y_mean = mean(y)
    numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(len(X)))
    denominator = sum((X[i] - x_mean) ** 2 for i in range(len(X)))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Function to make predictions
def predict(X, slope, intercept):
    return [slope * x + intercept for x in X]

# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

# Function to calculate R-squared
def r_squared(y_true, y_pred):
    y_mean = mean(y_true)
    ss_total = sum((y - y_mean) ** 2 for y in y_true)
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    return 1 - (ss_residual / ss_total)

# Calculate coefficients
slope, intercept = calculate_coefficients(X, y)

# Make predictions
y_pred = predict(X, slope, intercept)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)

# Display results
print("Model Coefficients:")
print(f"  Slope (m): {slope:.2f}")
print(f"  Intercept (b): {intercept:.2f}")
print("\nModel Evaluation:")
print(f"  Mean Squared Error (MSE): {mse:.2f}")
print(f"  R-squared (R²): {r2:.2f}")

print("\nPredictions:")
for xi, yi, yi_pred in zip(X, y, y_pred):
    print(f"  X={xi}, Actual Y={yi:.2f}, Predicted Y={yi_pred:.2f}")
