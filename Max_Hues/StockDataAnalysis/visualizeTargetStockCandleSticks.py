import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

def visualizeTargetStockCandleSticks():
    #Read the Target stock
    df_target_stock=pd.read_csv('Max_Hues/S&P500_Stock_Data/AAPL.csv',index_col=0,parse_dates=True)

    #This will scnifcantly chrink the data
    #I could also do 10Min instead of 10D
    #Open High Low Close which is ohlc
    df_ohlc= df_target_stock['Adj Close'].resample('10D').ohlc()
    #Using the sum of 10 days
    df_volume=df_target_stock['Volume'].resample('10D').sum()

    df_ohlc.reset_index(inplace=True)
    #In order to graph the candels we need to convert it to mdates
    df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)

    axis1=plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    axis2=plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1 , sharex=axis1)
    #Necessary for displaying the proper dates
    axis1.xaxis_date()

    #Its a framework that is used to graph candels
    #g for green
    #As it is a candelstick framework the other candels are red
    candlestick_ohlc(axis1,df_ohlc.values, width=2, colorup='g')
    #This shows the blue graph in the button of the candle stick graph which represents the volume 
    axis2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)

    plt.show()

visualizeTargetStockCandleSticks()
