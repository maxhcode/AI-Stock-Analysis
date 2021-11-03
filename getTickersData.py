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

#print(save_tickers())