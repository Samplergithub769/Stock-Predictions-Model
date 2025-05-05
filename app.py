import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('D:/VS code projects/Stock Price Prediction/Stock Predictions Model.keras')

# Page settings
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Styling with CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
    }
    h1 {
        color: #003366;
        text-align: center;
    }
    .stTextInput input {
        background-color: #e6f2ff;
        border-radius: 5px;
        padding: 10px;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .dataframe {
        border-radius: 8px;
        overflow: auto;
        background-color: white;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📊 Stock Market Predictor</h1>", unsafe_allow_html=True)

# Input for stock ticker
stock = st.text_input("Enter Stock Ticker Symbol (e.g., GOOG, AAPL):", "GOOG")

# Download stock data
start = '2012-01-01'
end = '2022-12-31'
data = yf.download(stock, start=start, end=end)

# Display full dataset
st.subheader("📈 Stock Data")
st.dataframe(data, use_container_width=True)

# Prepare train/test data
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Normalize with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
final_data = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_data)

# Moving average plot function
def plot_with_mas(title, series_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    for series, label, color in series_list:
        ax.plot(series, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# Plot MA50
st.subheader("📉 Closing Price vs 50-Day MA")
plot_with_mas("Closing Price vs MA50", [
    (data['Close'].rolling(50).mean(), "MA50", "red"),
    (data['Close'], "Close", "green")
])

# Plot MA50 and MA100
st.subheader("📉 Closing Price vs 50 & 100-Day MA")
plot_with_mas("Closing Price vs MA50 vs MA100", [
    (data['Close'].rolling(50).mean(), "MA50", "red"),
    (data['Close'].rolling(100).mean(), "MA100", "blue"),
    (data['Close'], "Close", "green")
])

# Plot MA100 and MA200
st.subheader("📉 Closing Price vs 100 & 200-Day MA")
plot_with_mas("Closing Price vs MA100 vs MA200", [
    (data['Close'].rolling(100).mean(), "MA100", "red"),
    (data['Close'].rolling(200).mean(), "MA200", "blue"),
    (data['Close'], "Close", "green")
])

# Model prediction
x, y = [], []
for i in range(100, input_data.shape[0]):
    x.append(input_data[i-100:i])
    y.append(input_data[i, 0])
x, y = np.array(x), np.array(y)

# Predict
predicted_prices = model.predict(x)

# Rescale
scale_factor = 1 / scaler.scale_[0]
predicted_prices = predicted_prices * scale_factor
y = y * scale_factor

# Final prediction plot
st.subheader("📈 Predicted Price vs Original Price")
fig_final = plt.figure(figsize=(10, 6))
plt.plot(y, label="Original Price", color='green')
plt.plot(predicted_prices, label="Predicted Price", color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Original vs Predicted Stock Price")
plt.legend()
st.pyplot(fig_final)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
