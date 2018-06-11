"""
Technical Task for Dataffirm
@author: Georgios Spyrou
"""

"""
I found the data at the following link: https://ycharts.com/companies/AAPL/revenues , where I exported them in a .csv file
For your convenience I uploaded the dataset on my github, and I am writing the data from there
"""


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

url = 'https://raw.githubusercontent.com/gpsyrou/Apple_Analysis/master/Apple_Revenues.csv'

data = pd.read_csv(url)

list(data.columns.values)

# Converting strings of dates to datetime objects
data['Date'] = pd.to_datetime(data['Date'])

data.groupby(data.Date.dt.year).agg(['count'])
# We observe that our dataset contains only 1 observation(single quarter) for the years 2005 and 2018
# Therefore we are going to drop these values as they dont provide sufficient information for these
# specific years

data = data.drop(data.index[[0,49]])
print(data.shape)

# We start we some Exploratory Data Analysis(EDA) 
# Plots:

# We can take a look at the general picture of the fluctuation of the revenues
# by plotting a line-plot that will show us the variation of quarterly revenues

rev_series = pd.Series.from_csv(url,header = 0)

rev_series = rev_series.drop(rev_series["2005-12-31"].index)
rev_series = rev_series.drop(rev_series["2018-03-31"].index)

groups = dict(list(data.groupby(data.Date.dt.year)))

plt.figure(figsize=(11,5))
rev_series.plot(style = '.-', linewidth = 1, color = 'r')
plt.title("Apple's Revenues for 2006 to 2017 per Quarter",fontsize = 14)
plt.ylim(min(rev_series),max(rev_series))
plt.ylabel("Revenue in Billions $",fontsize = 14)
plt.grid(True, color = 'black')
plt.show()

# We can clearly see that between 2005-2006 where our data start and 2017,the revenues rised
# from around 5 to 10 billions to over 40 billions per quarter.Specifically we can check the
# case of 2006 where the revenue per quarter ranged between 4.3 to 7.1 billion , while ten years later
# at 2016 for the same periods(quarters) it ranged from 50.5 to 78.3 billions

# Its clear that there is an overall increasing trend in the data along with some seasonal variations

# Time series Lag scatter plot
plt.figure(figsize=(5,5))
plt.title("Lag plot for the Revenues", fontsize = 14)
plt.xlim(min(rev_series),max(rev_series))
plt.ylim(min(rev_series),max(rev_series))
lag_plot(rev_series)
plt.show()

# We can observe a positive relationship for the corellation between observations and their lag1 values (t-1 values)
# Clearly, we can see a type of "cluster" for ranges 5 to 60 billion seem to have a stronger relationship, but
# its easy to observe that the relationship is weak for higher values. Therefore the general outcome is that we have
# a somehow weak relationship between t and t-1 values.

# Autocorrelation plot
plt.figure(figsize=(10,4))
autocorrelation_plot(rev_series)
plt.title("Autocorrelation Plot for Revenues", fontsize = 14)

# Finally from the autocorrelation plot we can observe exactly whay we expected.
# For small lag values (t-1,t-2) we can see that there is a high degree of autocorrelation (they dont happend at random)
# between adjacent and near-adjacent observations, but while we move to higher lag values (after t-10)
# we can see that there is no significant evidence that for example t and t-20 is not happening at random.
# We should note though, that uncorrelated doesnt necessarily mean random, as they might exhibit non-randomness in other ways that we didnt find here.

# Numerical EDA

data = data.sort_values(by = 'Date')
grouped_data = data.groupby(data.Date.dt.year)
grouped_data.agg(['sum','mean','median'])

# Clearly the revenue are constantly growing as we going from 2006 to 2017. We can see that the total revenue
# for year 2006 was around 20 billions , while at 2017 we Apple grown to 240 billion (x12 times more).The growth of revenue
# is constant between 2006 to 2014, but after that we can observe some fluctuations.


# Change of Revenue per Quarter of 2017 and comparison with the one of 2006
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.title("2006 Revenue Change per Quarter", fontsize = 14)
plt.ylabel("Rev. in Billions $",fontsize = 14)
plt.plot(rev_series.iloc[0:4])
plt.grid(True)
plt.subplot(2,1,2)
plt.title("2017 Revenue Change per Quarter", fontsize = 14)
plt.ylabel("Rev. in Billions $",fontsize = 14)
plt.plot(rev_series.iloc[44:49])
plt.grid(True)
plt.tight_layout()
plt.show()


#### Time series forecast

# We load the dataset again, but this time we hold Date as the Index as it will be useful
# for the time series forecast

dt_parse = lambda dates: pd.datetime.strptime(dates,'%d/%m/%Y')
data_ts = pd.read_csv(url,parse_dates = ['Date'], index_col = 'Date' , date_parser = dt_parse)
print(data_ts.head())

# Again, delete the observations for 2005 and 2018 as they are insufficient

data_ts = data_ts.drop(data_ts.index[-1])
data_ts = data_ts.drop(data_ts.index[0])
print(data_ts.head())


# Stationarity : We already observed before that our data appear (plot of the ts) to have a stationary form
# which is one of the required assumptions for Time Series

# Now we can check for the constant mean,constant variance requirements of stationarity


def stationarity_checking(ts,win):
    
    # Rolling mean,std:
    rolling_mn = pd.rolling_mean(ts, window = win )
    rolling_std = pd.rolling_std(ts, window  = win)
    
    plt.plot(ts, color='blue',label='Original TS')
    plt.plot(rolling_mn, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label = 'Rolling St.Dev.')
    plt.legend(loc='best')
    plt.grid(True, color = 'lightgrey')
    plt.title('Rolling Mean & Standard Deviation of Revenues')
    
    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    fuller_test = adfuller(ts['Revenue(Quarterly)'], autolag='AIC')
    results_ts = pd.Series(fuller_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in fuller_test[4].items():
        results_ts['Critical Value (%s)'%key] = value
    print(results_ts)

stationarity_checking(data_ts,3) # we use 3 as we take into consideration the t-1,t-2,t-3 values ( average of last 3 values)


# Thus we can observe that we have  high variation for the standard deviation. For the mean we can see that its increasing in general,
# although its clear that there are some up and downs .
# From the Dickey-Fuller  test we can see that the test statistic is more than the critical values.
# Therefore as we expected , the time series model is not stationary yet.
# Our issue behind non-stationarity is the fact that we have a constant increase in the Revenues,something that has an impact at the mean
# as well as, some seasonality issues-which we can observe at the plot.

# Reducing Trend (fluctuations of the mean):

# Log-transformation  in order to penalize the extreme values

data_log = np.log(data_ts)

moving_average_new = pd.rolling_mean(data_log ,3)

plt.plot(data_log, color = 'orange' , label = 'Original TS')
plt.plot(moving_average_new,color = 'red', label = 'Rolling Mean')
plt.title("Moving Average - Original Revenues Plot", fontsize = 13)
plt.ylabel("Quarterly Log-Revenue in Billions($)")
plt.legend(loc='best')
plt.grid(True, color = 'lightgrey')

# Method 1 - Then we subtract the rolling mean from the original values in the initial series

data_moving_avg_diff = data_log - moving_average_new
print(data_moving_avg_diff.head())

# first two are NaN as we test for average of last 3 values, and thus we can drop them and test for stationarity
data_moving_avg_diff.dropna(inplace = True)
stationarity_checking(data_moving_avg_diff,3)

# We can see that this looks like a much better series than before! The Test statistic is closer to the critical values

# Method 2 - Exponentially Weighted Moving Average Technique(ewma)
# where we assign weights to all the previous values with a decay factor

ewma_avg = pd.ewma(data_log, halflife = 3)
data_ewma_diff = data_log - ewma_avg
stationarity_checking(data_ewma_diff,3)

# Method 3 - Differencing

data_log_diff = data_log - data_log.shift()
plt.plot(data_log_diff)
data_log_diff.dropna(inplace = True)
stationarity_checking(data_log_diff,3)

# The results are getting better as we approaching a -somehow- stationary series.


# Method 4 - Decomposition

decomposition = seasonal_decompose(data_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

data_log_decomp = residual
data_log_decomp.dropna(inplace = True)
stationarity_checking(data_log_decomp, 3)

# Thats our best model ! We can clearly observe that our Test-Statistic is lower than the 1% critical value and therefore
# this is very close to stationary!


# Prediction 
# For the prediction purposes we will use the 3'rd method from above i.e. the Differencing

acf_lag = acf(data_log_diff, nlags = 3)
pacf_lag = pacf(data_log_diff, nlags = 3, method='ols')

#Plot ACF vs PACF: 

plt.subplot(121) 
plt.plot(acf_lag)
plt.axhline(y = 0,linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='red')
plt.axhline(y = 1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='red')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(pacf_lag)
plt.axhline(y = 0,linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='red')
plt.axhline(y = 1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='red')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# The red lines correspond to the confidence intervals, and they help us to find the proper values that we will
# use in our models. The lag value where both acf and pacf "hit" the upper CI is giving us the proper values for p and q for our models
# hence we will take p=1,q=1

# Function that computes the RSS for each of the following models
def compute_RSS(mdl):
    print("RSS is " + str(sum((mdl.fittedvalues - data_log_diff['Revenue(Quarterly)'])**2)))



# Module 1 - Prediction with AR Model (autoregressive model)

ar_model = ARIMA(data_log, order = (1,1,0))
ar_results = ar_model.fit(disp = -1)
plt.plot(data_log_diff)
plt.plot(ar_results.fittedvalues, color='red')
plt.title("Autoregressive Model")

compute_RSS(ar_results)

# Module 2 - Prediction with MA Model (Moving Average)
ma_model = ARIMA(data_log, order = (0,1,1))
ma_results = ma_model.fit(disp = -1)
plt.plot(data_log_diff)
plt.plot(ma_results.fittedvalues, color='red')
plt.title("Movign Average Model")

compute_RSS(ma_results)


# Module 3 -  Combination ARIMA
arima_model = ARIMA(data_log, order = (1, 1, 1))  
arima_results = arima_model.fit(disp = -1)  
plt.plot(data_log_diff)
plt.plot(arima_results.fittedvalues, color='red')
plt.title("ARIMA Model")

compute_RSS(arima_results)

# We can see that the ARIMA model achieves the lowest RSS and thus it's the optimal model

# Final Prediction
periods = arima_results.forecast(steps = 5)[0]

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


# Total Revenue of Apple for Fiscal year

tot_rev_2018 = rev_2018_Q1 + rev_2018_Q2 + rev_2018_Q3 + rev_2018_Q4

print('Apple\'s Revenue prediction for Fiscal Year 2018 is: ' +
      str(tot_rev_2018)[0:3] +'.'+ str(tot_rev_2018)[3:5]+' Billion dollars')






