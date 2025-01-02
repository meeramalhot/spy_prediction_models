import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Download historical stock prices from Yahoo Finance
#look at spy
ticker = 'SPY'
data = yf.download(ticker, start='2021-01-01', end='2024-12-27')

# Use only the close and vol columns
data = data[['Close', 'Volume']]

#calculate each col w different ewm spans
data['EMA_5'] = data['Close'].ewm(span=5).mean()
data['EMA_10'] = data['Close'].ewm(span=10).mean()
data['EMA_20'] = data['Close'].ewm(span=20).mean()
data['EMA_50'] = data['Close'].ewm(span=50).mean()


# Calculate Relative Strength Index (RSI) 
#source: https://www.quora.com/What-are-some-alternative-ways-to-calculate-RSI-values-in-Python-and-Pandas

# Calculate rolling gains and losses 
delta = data['Close'].diff() 
gain = delta.where(delta > 0, 0) 
loss = -delta.where(delta < 0, 0) 
 
# Calculate average gain and loss 
average_gain = gain.rolling(window=14).mean() 
average_loss = loss.rolling(window=14).mean() 
 
# Calculate RSI 
rs = average_gain / average_loss.abs() 
rsi = 100 - (100 / (1 + rs)) 
data['RSI'] = rsi

# Calculate MACD
#source: https://www.alpharithms.com/calculate-macd-python-272222/

# Get the 26-day EMA of the closing price
k = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
# Get the 12-day EMA of the closing price
d = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
# Subtract the 26-day EMA from the 12-Day EMA to get the MACD
macd = k - d

#optional
# Get the 9-Day EMA of the MACD for the Trigger line
#macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
# Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
#macd_h = macd - macd_s

# Add all of our new values for the MACD to the dataframe
data['MACD'] = macd
#data['macd_h'] = macd_h
#data['macd_s'] = macd_s

print(data)

data['MACD_Signal'] = macd.macd_signal()
data['MACD_Diff'] = macd.macd_diff()

# Shift the closing prices to create the target variable
data['Target'] = data['Close'].shift(-1)

# Drop any rows with NaN values
data = data.dropna()

# Define the feature set and target variable
X = data.drop('Target', axis=1)
y = data['Target']

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=False
)

# Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and Train the Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=13, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to plot predictions
def plot_predictions(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual.values, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Plot predictions
plot_predictions(y_test, y_pred)

# Deviance Plot
test_score = np.zeros(model.n_estimators, dtype=np.float64)
for i, y_pred_stage in enumerate(model.staged_predict(X_test_scaled)):
    test_score[i] = mean_squared_error(y_test, y_pred_stage)

fig = plt.figure(figsize=(10, 6))
plt.title("Deviance")
plt.plot(np.arange(model.n_estimators) + 1, model.train_score_, "b-", label="Training Set Deviance")
plt.plot(np.arange(model.n_estimators) + 1, test_score, "r-", label="Test Set Deviance")
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
plt.show()

# Plot Feature Importances
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), features[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
