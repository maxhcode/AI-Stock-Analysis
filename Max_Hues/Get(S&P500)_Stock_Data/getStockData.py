import pickle
import bs4 as bs
import pickle
import datetime as dt
import os
import pandas_datareader.data as web

#This function goes through the S and P 500 ticker symbol list an moves the data into csv files into the folder S&P500_Stock_Data
def get_stock_data():
    #Open Ticker file
    with open("Get(S&P500)_Stock_Tickers/S&P500_Tickers.pickle",'rb') as SANDP500_Tickers:
        tickers_data=pickle.load(SANDP500_Tickers)
        #If folder does not exists yet create one
        if not os.path.exists('S&P500_Stock_Data'):
            os.makedirs('S&P500_Stock_Data')
        #Set data start and end date
        start= dt.datetime(2010,1,1)
        end=dt.datetime(2021,11,2)#year,month,day

        stock_ticker_count=0
        #Looping through ticker data
        for ticker in tickers_data:
            try:
                #Finished going through all the tickers
                if stock_ticker_count==500:
                    break
                stock_ticker_count+=1
                print(ticker)
            except:
                continue
            
            #Getting stock information for each ticker from start to end date
            #Each stock is then being exported into a csv file
            try:
                df=web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('S&P500_Stock_Data/{}.csv'.format(ticker))
            except:
                print("Error")
                continue
get_stock_data()
