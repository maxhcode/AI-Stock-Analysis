import bs4 as bs
import pickle
import requests
import pickle
import requests

def save_S_AND_P_500_tickers():
    #Requesting Website Code
    wikipedia_responds=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #Convert that data into text so beautifulSoup can read it
    soup=bs.BeautifulSoup(wikipedia_responds.text)
    #Then go to the HTML Code and go to table data in the HTML code which class is called wikitable sortable
    wikitable=soup.find('table',{'class':'wikitable sortable'})
    #Add all tickers in the Table to the array tickers
    tickers=[]
    #In table find all 'td' html elements
    count = 0 
    for row in wikitable.findAll('tr')[1:]:
        count +=1
        ticker=row.findAll('td')[0].text[:-1]
        tickers.append(ticker)

    #Meassure how many tickers were moved into pickle file
    print(count)    
    #This creates a pickle file which is saving all of the S and P 500 companies tickers from the Array
    #Pickle in Python is primarily used in serializing and deserializing a Python object structure.
    with open("Get(S&P500)_Stock_Tickers/S&P500_Tickers.pickle",'wb') as f:
        pickle.dump(tickers, f)
    return tickers

print(save_S_AND_P_500_tickers())

