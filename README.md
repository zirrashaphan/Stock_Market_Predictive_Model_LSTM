# Stock_Market_Predictive_Model_LSTM
This project demonstrates how to scrape historical stock data, perform feature engineering, and use an LSTM (Long Short-Term Memory) neural network to predict future stock prices. The example uses Tesla (TSLA) stock data from Yahoo Finance, but it can be adapted for other stocks by changing the ticker symbol.

**Requirements**
To run this code, you need the following Python libraries:
* yfinance
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* tensorflow
You can install them using pip:
bash
  pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow
This code is designed to run in Google Colab, which provides a convenient environment with access to Google Drive for saving and loading data. You will need a Google account and access to Google Drive to execute the code as written.

# Usage
Follow these steps to run the code in Google Colab:
* (Run in Colab)[https://colab.research.google.com/github/zirrashaphan/Stock_Market_Predictive_Model_LSTM/blob/main/Stock_Market_Predictive_Model_LSTM.ipynb]
* **Mount Google Drive**: The code mounts your Google Drive to save and load data. Execute the cell with drive.mount('/content/drive') and authenticate when prompted.
* **Scrape Data**: The code scrapes historical stock data for the specified ticker (default: TSLA) from Yahoo Finance for the given date range (default: 2015-01-01 to 2024-10-31).
* **Save Data**: The scraped data is saved as a CSV file in your Google Drive under the Datasets folder (e.g., /content/drive/My Drive/Datasets/TSLA_stock_data.csv).
* **Load and Clean Data**: The data is loaded from the CSV, cleaned, and prepared for analysis.
* **Feature Engineering**: Technical indicators such as moving averages, volatility, daily returns, lagged features, and RSI are calculated.
* **Handle Missing Values**: Rows with missing values are dropped.
* **Visualize Data**: Several plots are generated to explore the data, including stock price trends, return distributions, and feature correlations.
* **Split Data**: The data is split into training and testing sets based on a specified date (default: 2022-01-01).
* **Scale Data**: Features and the target variable are scaled using MinMaxScaler.
* **Prepare LSTM Data**: The data is reshaped into sequences for LSTM input using a specified time step.
* **Build and Train LSTM Model**: An LSTM model is defined, compiled, and trained on the training data.
* **Make Predictions**: The model predicts stock prices on the test set.
* **Visualize Predictions**: A plot compares the actual and predicted stock prices.
Execute the cells in the provided order to complete the workflow.

# Data
The stock data is sourced from Yahoo Finance using the yfinance library. It includes daily values for Open, High, Low, Close, Adjusted Close, and Volume for the specified period.

**Preprocessing**
The data undergoes several preprocessing steps:

**Cleaning**:
* The CSV header is adjusted, setting the first column as 'Date'.
* The 'Date' column is converted to datetime and set as the index.
* The redundant 'Close' column is dropped, focusing on 'Adjusted Close'.
* All columns are converted to numeric types.

**Feature Engineering**:
* Moving Averages: 10-day, 50-day, and 200-day simple moving averages (SMA) of the Adjusted Close price.
* Volatility: 10-day rolling standard deviation of the Adjusted Close price.
* Daily Returns: Percentage change in the Adjusted Close price.
* Lagged Features: Adjusted Close prices from the previous 5 days (Lag_1 to Lag_5).
* RSI: 14-day Relative Strength Index to measure price momentum.
* Missing Values: Rows with any missing values in the feature columns are dropped to ensure data integrity for modeling.

# Model
The prediction model is an LSTM neural network with the following architecture:

**Input Layer**: LSTM layer with 50 units, return_sequences=True to pass sequences to the next layer.

**Hidden Layer**: LSTM layer with 50 units, return_sequences=False to output a single vector.

**Output Layer**: Dense layer with 1 unit for predicting the next day's Adjusted Close price.

The model is compiled with:

**Optimizer**: Adam

**Loss Function**: Mean Squared Error

It is trained for 50 epochs with a batch size of 32.

# Evaluation
The model's performance is evaluated by plotting the predicted stock prices against the actual prices on the test set. This visual comparison helps assess how well the model captures stock price trends.

**Outputs**
The code produces the following outputs:

**Saved Data**: 
* Raw scraped data saved as {ticker}_stock_data.csv in Google Drive.
* Formatted data saved as formatted_{ticker}_stock_data.csv (though not used further in this script).

**Plots**:
* Stock price trend over time.
* Distribution of daily returns (histogram).
* Feature correlation matrix (heatmap).
* Adjusted Close price with moving averages.
* Box plot of daily returns for outlier detection.
* Actual vs. predicted stock prices on the test set.
* 
# Customization
You can modify the following variables to customize the analysis:

**Stock Ticker**: Change ticker (e.g., from 'TSLA' to 'AAPL') to analyze a different stock.

**Date Range**: Adjust start_date and end_date to scrape data for a different period.

**Split Date**: Modify split_date to alter the training and testing periods.

**Time Step**: Change time_step (default: 5) to use a different number of previous days for prediction.

# Limitations

**Data Scope**: The model relies solely on historical price data and derived features, ignoring external factors such as news, economic events, or market sentiment.

**Purpose**: This is a demonstration project and may not be suitable for actual trading without further validation, testing, and refinement.

**Missing Values**: Dropping rows with missing values may discard useful information; alternative imputation methods could be considered for improvement.

**Model Simplicity**: The LSTM architecture is basic and could be enhanced with additional layers, hyperparameters tuning, or alternative models for better performance.
