import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

df=pd.read_csv('dataset_target_2.csv')
#renamed to df1
df1=pd.read_csv('stock_details/AMZN.csv')
print(df1.columns)

#renamed to df1
Dates=[]
i=0
while i<len(df1):
    Dates.append(df1.iloc[i]['Date'])
    i+=1

df2=pd.read_csv('dataset_target_2.csv')
df2['Date']=Dates
print(df2.columns)
df2.to_csv("dataset_target_2.csv",index=False)

def merge():
	df1=pd.read_csv('dataset_target_2.csv',index_col='Date')

	df3=pd.read_csv('stock_details/AMZN.csv')
	df2=pd.read_csv('Dataset_temp.csv',index_col='Date')

	Dates=[]
	i=0
	while i<len(df3):
		Dates.append(df3.iloc[i]['Date'])
		i+=1
		
	
	df_new=df1.join(df2,how='outer')
	df_new.fillna(0.0)

	df_new['Date']=Dates

	df_new.to_csv('Dataset_main.csv',index=False)

merge()

df=pd.read_csv('Dataset_main.csv')
print(df.columns)

df10=pd.read_csv('Dataset_temp.csv')
print(df10.columns)

#Reihenfolge noch machen mit nummer an am ende 
df15=pd.read_csv('stock_details/AMZN.csv')
df20 = pd.DataFrame(df15)
df20.to_csv('dataset_target.csv', index=False)
