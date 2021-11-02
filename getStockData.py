import bs4 as bs
import pickle
import requests

def save_tickers():
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup=bs.BeautifulSoup(resp.text)
    table=soup.find('table',{'class':'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        ticker=row.findAll('td')[0].text[:-1]
        tickers.append(ticker)
    
    #This creates a pickle file which is saving all of the S and P 500 companies tickers
    #Pickle in Python is primarily used in serializing and deserializing a Python object structure.
    with open("tickers.pickle",'wb') as f:
        pickle.dump(tickers, f)
    return tickers

print(save_tickers())

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
            count=500
        start= dt.datetime(2010,1,1)
        end=dt.datetime(2021,11,2)#year,month,day
        count=0
        for ticker in tickers:
            if count==500:
                break
            count+=1
            print(ticker)

            try:
                df=web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_details/{}.csv'.format(ticker))
            except:
                print("Error")
                continue

fetch_data()
