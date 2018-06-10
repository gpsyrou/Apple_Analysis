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
plt.subplot(2,1,2)
plt.title("2017 Revenue Change per Quarter", fontsize = 14)
plt.ylabel("Rev. in Billions $",fontsize = 14)
plt.plot(rev_series.iloc[44:49])
plt.tight_layout()
plt.show()







