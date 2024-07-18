# Stock Price Prediction Using LSTM
This project aims to predict stock prices using Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN) that is well-suited for time series prediction.

## Introduction
Stock price prediction is one of the most challenging tasks in the financial world. By leveraging an LSTM model, this project attempts to predict the closing prices of stocks based on historical data. The research uses daily stock price data and the Rupiah to USD exchange rate from January 2013 to December 2022. Open, Low, High, and Exchange Rate variables are features used to predict closing prices (Close).

## Features
- `Data Preparation` : The stages of data preparation. including data merging, data correlation, and data download.
- `Proses Pengujian` : The stages of testing the LSTM model. including data normalization, data splitting, timestep initialization, prediction result visualization, accuracy results, and data denormalization.
- `Prediksi Kedepan` : The stages of predicting the next day's closing price (h+1), including the timestep values used, the number of days to be predicted (up to 30 days), accuracy values, and prediction result visualization.

## Installation
### Prerequisites
Make sure you have Python 3.x and pip installed. Then, install the necessary libraries with the following command:
```sh
pip install -r requirements.txt
```

### Requirements
Here are the required libraries:
- matplotlib
- scikit-learn
- datetime
- tensorflow
- streamlit
- streamlit-option-menu

## Usage
### 1. Clone this repository
### 2. Prepare the dataset.
Download the stock price dataset and save it in the `Dataset` folder. Or you can download the data from [stock data](https://finance.yahoo.com/quote/BBCA.JK/history?period1=1356998400&period2=1672531200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) and [exchange rate data](https://finance.yahoo.com/quote/IDR%3DX/history/?period1=1325376000&period2=1327968000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) sites.
### 3. Run the App script on your terminal.
```sh
Streamlit run App.py
```

## Model
The model used in this project is LSTM, implemented using TensorFlow. This model is trained with historical stock price data to predict future closing prices. In this project, there are 4 best models produced by each timestep in the training process. You can access them in [here](model).

## Hasil
The prediction results will be visualized in the form of a graph, showing the comparison between the actual prices and the prices predicted by the model. The best result in this research is with an MSE value of 0.000323329.
