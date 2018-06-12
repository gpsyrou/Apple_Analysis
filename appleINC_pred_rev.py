# Author: Georgios Spyrou 

# Prediction of Apple's revenue for fiscal year 2018

# Import data and libraries of interest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import lag_plot,autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/gpsyrou/Apple_Analysis/master/Apple_Revenues.csv'
data = pd.read_csv(url)
list(data.columns.values)

# Converting strings of dates to datetime objects
data['Date'] = pd.to_datetime(data['Date'])
data.groupby(data.Date.dt.year).agg(['count'])
data = data.drop(data.index[[0,49]])
print(data.shape)

# We can take a look at the general picture at the fluctuation of the revenues
# by plotting a line-plot that will show us the variation of quarterly revenues
rev_series = pd.Series.from_csv(url,header = 0)

rev_series = rev_series.drop(rev_series["2005-12-31"].index)
rev_series = rev_series.drop(rev_series["2018-03-31"].index)

plt.figure(figsize = (11,5))
rev_series.plot(style = '.-', linewidth = 1, color = 'r')
plt.title("Apple's Revenues for 2006 to 2017 per Quarter",fontsize = 14)
plt.ylim(min(rev_series),max(rev_series))
plt.ylabel("Revenue in Billions $",fontsize = 14)
plt.grid(True, color = 'black')
plt.show()

# Time series Lag scatter plot
plt.figure(figsize = (5,5))
plt.title("Lag plot for the Revenues", fontsize = 14)
plt.xlim(min(rev_series),max(rev_series))
plt.ylim(min(rev_series),max(rev_series))
lag_plot(rev_series)
plt.show()

# Autocorrelation plot
plt.figure(figsize = (10,4))
autocorrelation_plot(rev_series)
plt.title("Autocorrelation Plot for Revenues", fontsize = 14)

data = data.sort_values(by = 'Date')
grouped_data = data.groupby(data.Date.dt.year)
grouped_data.agg(['sum','mean','median'])

# Change of Revenue per Quarter of 2017 and comparison with the one of 2006
plt.figure(figsize=(7,7))
plt.subplot(2,1,1)
plt.title("2017 Revenue Change per Quarter", fontsize = 14)
plt.ylabel("Rev. in Billions $",fontsize = 14)
plt.plot(rev_series.iloc[0:4])
plt.grid(True)
plt.subplot(2,1,2)
plt.title("2006 Revenue Change per Quarter", fontsize = 14)
plt.ylabel("Rev. in Billions $",fontsize = 14)
plt.plot(rev_series.iloc[44:49])
plt.grid(True)
plt.tight_layout()
plt.show()

# We load the dataset again, but this time we hold Date as the Index as it will be useful
# for the time series forecast
dt_parse = lambda dates: pd.datetime.strptime(dates,'%d/%m/%Y')
data_ts = pd.read_csv(url,parse_dates = ['Date'], index_col = 'Date' , date_parser = dt_parse)
data_ts = data_ts.drop(data_ts.index[-1])
data_ts = data_ts.drop(data_ts.index[0])
print(data_ts.head())

# Function that calculates the rolling mean and standard deviation, as well as performing the Dickey-Fuller Test
def stationarity_checking(ts,win):
    
    # Calculating rolling mean and standard deviation:
    rolling_mn = pd.rolling_mean(ts, window = win )
    rolling_std = pd.rolling_std(ts, window  = win)
    
    plt.plot(ts, color = 'blue',label = 'Original TS')
    plt.plot(rolling_mn, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling St.Dev.')
    plt.legend(loc = 'best')
    plt.grid(True, color = 'lightgrey')
    plt.title('Rolling Mean & Standard Deviation of Revenues', fontsize = 10)
    
    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    fuller_test = adfuller(ts['Revenue(Quarterly)'], autolag = 'AIC')
    results_ts = pd.Series(fuller_test[0:4], index = ['Test Statistic','P-value','#Lags Used','Number of Observations Used'])
    for key,value in fuller_test[4].items():
        results_ts['Critical Value (%s)'%key] = value
    print(results_ts)
    

# We use 3 as we take into consideration the t-1,t-2,t-3 values ( average of last 3 values)
stationarity_checking(data_ts,3) 

# Log-transformation  in order to penalize the extreme values

data_log = np.log(data_ts)
moving_average_new = pd.rolling_mean(data_log ,3)

plt.plot(data_log, color = 'orange' , label = 'Original TS')
plt.plot(moving_average_new,color = 'red', label = 'Rolling Mean')
plt.title("Moving Average - Original Revenues Plot", fontsize = 13)
plt.ylabel("Quarterly Log-Revenue in Billions($)")
plt.legend(loc='best')
plt.grid(True, color = 'lightgrey')

# Smoothing
data_moving_avg_diff = data_log - moving_average_new
print(data_moving_avg_diff.head())

# First two are NaN as we test for average of last 3 values, and thus we can drop them and test for stationarity
data_moving_avg_diff.dropna(inplace = True)
stationarity_checking(data_moving_avg_diff,3)

# EWMA:
ewma_avg = pd.ewma(data_log, halflife = 3)
data_ewma_diff = data_log - ewma_avg
stationarity_checking(data_ewma_diff,3)

# Differencing
data_log_diff = data_log - data_log.shift()
plt.plot(data_log_diff)
data_log_diff.dropna(inplace = True)
stationarity_checking(data_log_diff,3)

# Decomposition

decomposition = seasonal_decompose(data_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

data_log_decomp = residual
data_log_decomp.dropna(inplace = True)
stationarity_checking(data_log_decomp, 3)

# Compute ACF,PACF
acf_lag = acf(data_log_decomp, nlags = 10)
pacf_lag = pacf(data_log_decomp, nlags = 10, method='ols')

# Plot ACF vs PACF: 
plt.subplot(121) 
plt.plot(acf_lag)
plt.axhline(y = 0,linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(data_log_decomp)),linestyle='--',color='red')
plt.axhline(y = 1.96/np.sqrt(len(data_log_decomp)),linestyle='--',color='red')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(pacf_lag)
plt.axhline(y = 0,linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(data_log_decomp)),linestyle='--',color='red')
plt.axhline(y = 1.96/np.sqrt(len(data_log_decomp)),linestyle='--',color='red')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# Function that computes the RSS for each of the models
def compute_RSS(mdl):
    comp = mdl.fittedvalues - data_log_decomp['Revenue(Quarterly)']
    comp.dropna(inplace = True)
    print("RSS is " + str(sum(comp**2)))
    

# AR model
ar_model = ARIMA(data_log, order = (1,1,0))
ar_results = ar_model.fit(disp = -1)
plt.plot(data_log_decomp)
plt.plot(ar_results.fittedvalues, color='red')
plt.title("Autoregressive Model")

compute_RSS(ar_results)

# MA model
ma_model = ARIMA(data_log, order = (0,1,1))
ma_results = ma_model.fit(disp = -1)
plt.plot(data_log_decomp)
plt.plot(ma_results.fittedvalues, color='red')
plt.title("Moving Average Model")

compute_RSS(ma_results)

# ARIMA model
arima_model = ARIMA(data_log, order = (1, 1, 1))  
arima_results = arima_model.fit(disp = -1)  
plt.plot(data_log_decomp)
plt.plot(arima_results.fittedvalues, color='red')
plt.title("ARIMA Model")

compute_RSS(arima_results)

# Final Prediction
periods = ar_results.forecast(steps = 5)[0]
val = [np.exp(i) for i in periods]

# Therefore we can calculate the prediction for each quarter of 2018
# Note that we are starting from [1] index as [0] corresponds to the one that we already have(Dec. of 2017)

rev_2018_Q1 = val[1] + data_ts['Revenue(Quarterly)'][1]
rev_2018_Q2 = val[2] + rev_2018_Q1
rev_2018_Q3 = val[3] + rev_2018_Q2
rev_2018_Q4 = val[4] + rev_2018_Q3

print('Apple\'s Revenue for Quarter 1 of 2018 is: ' +
      str(rev_2018_Q1)[0:2] +'.'+ str(rev_2018_Q1)[2:4]+' Billion dollars')
print('Apple\'s Revenue for Quarter 2 of 2018 is: ' +
      str(rev_2018_Q2)[0:2] +'.'+ str(rev_2018_Q2)[2:4]+' Billion dollars')
print('Apple\'s Revenue for Quarter 3 of 2018 is: ' +
      str(rev_2018_Q3)[0:2] +'.'+ str(rev_2018_Q3)[2:4]+' Billion dollars')
print('Apple\'s Revenue for Quarter 4 of 2018 is: ' +
      str(rev_2018_Q4)[0:2] +'.'+ str(rev_2018_Q4)[2:4]+' Billion dollars')


# Total Revenue of Apple for Fiscal year 2018
tot_rev_2018 = rev_2018_Q1 + rev_2018_Q2 + rev_2018_Q3 + rev_2018_Q4

print('Apple\'s Revenue prediction for Fiscal Year 2018 is: ' +
      str(tot_rev_2018)[0:3] +'.'+ str(tot_rev_2018)[3:5]+' Billion dollars')


# End of Project / Author: Georgios Spyrou