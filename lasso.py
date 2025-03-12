# Install necessary packages if they are not already installed (uncomment the line below)
# !pip install yfinance scikit-learn pandas numpy matplotlib

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Fetch historical stock data
ticker = 'SPY'  # You can change this to any ticker symbol you like
# Get today's date and then compute yesterday's date
today = datetime.datetime.now()
end_time = today.strftime("%Y-%m-%d")
print(end_time)

# Set the start time to 10 years ago using the current month and day
old_year = today.year - 10
start_time = datetime.datetime(old_year, today.month, today.day).strftime("%Y-%m-%d")

# Download data from start_time to end_time (which is yesterday)
df = yf.download(ticker, start=start_time, end=end_time)

# Download data
df = yf.download(ticker, start=start_time, end=end_time)

# Create moving averages
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Create the target variable (shifted closing price)
df['Future_Close'] = df['Close'].shift(-1)

# Drop rows with missing values
df.dropna(inplace=True)

# Features and target variable
features = ['Close', 'MA5', 'MA10', 'MA20', 'MA50', 'Volume']
X = df[features]
y = df['Future_Close']
print(X)

# Split data into training and test sets (80% training, 20% testing)
split_index = int(len(df) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Scale the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models with regularization parameter alpha
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)

# Fit the models on the training data
lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

# Make predictions
lasso_pred = lasso.predict(X_test_scaled)
ridge_pred = ridge.predict(X_test_scaled)

# Calculate Mean Squared Error
lasso_mse = mean_squared_error(y_test, lasso_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

print(f'Lasso Regression MSE: {lasso_mse:.4f}')
print(f'Ridge Regression MSE: {ridge_mse:.4f}')

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Price', color='black')
plt.plot(lasso_pred, label='Lasso Predicted Price', color='red', linestyle='--')
plt.plot(ridge_pred, label='Ridge Predicted Price', color='blue', linestyle='--')
plt.title(f'Actual vs Predicted Closing Prices for {ticker}')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Get the latest available data point
latest_data = df.iloc[-1]

# Prepare the features for prediction
latest_features = latest_data[features].values.reshape(1, -1)

# Scale the features
latest_features_scaled = scaler.transform(latest_features)

# Make predictions for the next trading day
lasso_pred_tomorrow = lasso.predict(latest_features_scaled)
ridge_pred_tomorrow = ridge.predict(latest_features_scaled)

latest_close_price = latest_data['Close']

# Predicting whether the stock price will go up or down

latest_close_price = latest_close_price.iloc[-1]

lasso_movement = "UP" if lasso_pred_tomorrow[0] > latest_close_price else "DOWN"
ridge_movement = "UP" if ridge_pred_tomorrow[0] > latest_close_price else "DOWN"

print(f"\nBased on Lasso Regression, the stock price is predicted to go {lasso_movement}.")
print(f"Based on Ridge Regression, the stock price is predicted to go {ridge_movement}.")

final_avg = ( lasso_pred_tomorrow + ridge_pred_tomorrow ) / 2

print(f"\nPredicted closing price for {ticker} on the next trading day:")
print(f"Lasso Regression Prediction: ${lasso_pred_tomorrow[0]:.2f}")
print(f"Ridge Regression Prediction: ${ridge_pred_tomorrow[0]:.2f}")
print(f"Avg Between the Two: ${final_avg[0]:.2f}")

