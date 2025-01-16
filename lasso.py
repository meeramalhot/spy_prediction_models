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
time = datetime.datetime.now()
end_time = time.strftime("%Y") + "-" + time.strftime("%m") + "-" + time.strftime("%d")
old_year = int(time.strftime("%Y")) - 10
start_time = str(old_year) + "-" + time.strftime("%m") + "-" + time.strftime("%d")

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