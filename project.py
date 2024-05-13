import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


company_ticker = input("Enter the company's ticker symbol (e.g., AAPL for Apple): ")


data = yf.download(company_ticker, start='2020-01-01', end='2022-01-01')

data['Price_Change'] = data['Close'].diff()
data['Target'] = (data['Price_Change'] > 0).astype(int)

data.dropna(inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]

X = data.drop(columns=['Target'])
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
