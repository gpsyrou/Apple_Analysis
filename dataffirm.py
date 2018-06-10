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


# We start we some Exploratory Data Analysis(EDA) 

# We can take a look at the general picture of the fluctuation of the revenues
# by plotting a line-plot that will show us the variation of quarterly revenues

line_plot = pd.Series.from_csv(url,header = 0)
plt.figure(figsize=(11,5))
line_plot.plot(style = '.-', linewidth = 1, color = 'r')
plt.title("Apple's Revenues for 2006 to 2017 per Quarter",fontsize = 14)
plt.ylim(min(data['Revenue(Quarterly)']),max(data['Revenue(Quarterly)']))
plt.ylabel("Revenue in Billions $",fontsize = 14)
plt.grid(True)
plt.show()

# We can clearly see that between 2005-2006 where our data start and 2017,the revenues rised
# from around 5 to 10 billions to over 40 billions per quarter.Specifically we can check the
# case of 2006 where the revenue per quarter ranged between 4.3 to 7.1 billion , while ten years later
# at 2016 for the same periods(quarters) it ranged from 50.5 to 78.3 billions










