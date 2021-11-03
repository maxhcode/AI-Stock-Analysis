import bs4 as bs
import pickle
import requests
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

#This function goes through the s and p 500 ticker symbol list an moves the data from 2010 until 2021 into csv files into the folder stock_details
def fetch_data():
    with open("tickers.pickle",'rb') as f:
        tickers=pickle.load(f)
        if not os.path.exists('stock_details'):
            os.makedirs('stock_details')
            count=200
        start= dt.datetime(2010,1,1)
        end=dt.datetime(2021,11,2)#year,month,day
        count=0
        for ticker in tickers:
            if count==200:
                break
            count+=1
            print(ticker)

            try:
                df=web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_details/{}.csv'.format(ticker))
            except:
                print("Error")
                continue
#fetch_data()
