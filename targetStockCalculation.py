import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

df=pd.read_csv('stock_details/AMZN.csv',index_col=0,parse_dates=True)


#This adds the 50 day moving average in the dataframe 
#Actually, it is the mean of the i-50 to i values of the “Adj Close” values for the ith index
df['Moving_av']= df['Adj Close'].rolling(window=50,min_periods=0).mean()

#Now we try to get more features by getting 
#Rate of increase in volume, Rate of increase in Adjusted Close for the stock
i=1
rate_increase_in_vol=[0]
rate_increase_in_adj_close=[0]
while i<len(df):
    rate_increase_in_vol.append(df.iloc[i]['Volume']-df.iloc[i-1]['Volume'])
    rate_increase_in_adj_close.append(df.iloc[i]['Adj Close']-df.iloc[i-1]['Adj Close'])
    i+=1
    
df['Increase_in_vol']=rate_increase_in_vol
df['Increase_in_adj_close']=rate_increase_in_adj_close

#df['Moving_av'].plot()
#df['Increase_in_vol'].plot()
#df['Increase_in_adj_close'].plot()
plt.show()

#In the lines above:
    #It takes the current volume minus the previous volume
    #It takes the current adjusted close minues the previous adjusted close
#The array is is saving the calculation in is then being saved in a new column in the dataframe

print(df)

#Exporting all the Amazon stock data to a csv 
df.to_csv("dataset_target_2.csv",index=False)