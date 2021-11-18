import os
import pandas as pd
import pickle

#This is creating a file which is moving all of the Adj Close Data into one column for each stock
#rename the Adj Close of each stock as the corresponding stock ticker
#Include it in the feature set
#In this case AMZN ticker Amazon is the target stock here
def compileAllStocksTogetherInOneCSV():
	with open("Get(S&P500)_Stock_Tickers/S&P500_Tickers.pickle",'rb') as SANDP500_Tickers:
			stock_tickers=pickle.load(SANDP500_Tickers)

	#Create new dataframe main dataframe all stocks together and target stock
	main_all_stocks_together_df=pd.DataFrame()

	for count,ticker in enumerate(stock_tickers):
		#If amazon is in ticker continue and not added to the csv or dataframe
		#This is the choosen stock
		if 'AMZN' in ticker:
			continue
		#If ticker does not exitst then continue 
		if not os.path.exists('S&P500_Stock_Data/{}.csv'.format(ticker)):
			continue
		# This reads out all of the stocks and there Date, High, Low, Open, Close, Volume, Adj Close this is read out by the ticker
		df=pd.read_csv('S&P500_Stock_Data/{}.csv'.format(ticker))
		#Then the index is set to Date
		df.set_index('Date',inplace=True)
		#Rename all stock Adjusted closes to the ticker symbol
		df.rename(columns={'Adj Close': ticker}, inplace=True)
		#Drop all 'Open','High','Low',"Close",'Volume' of each Stock
		df.drop(['Open','High','Low',"Close",'Volume'],axis=1,inplace=True)
		#Now add all stock tickers with each all Adjusted close values into a the new dataframe main_all_stocks_together_df
		#But only added if the Dataframe is empty 
		if main_all_stocks_together_df.empty:
			main_all_stocks_together_df=df
		else:
			#If the dataframe is not empty then the current and already existing datframe would be joined with "outher"
			#The outer keyword returns all records when there is a match in left (table1) or right (table2) table records
			main_all_stocks_together_df=main_all_stocks_together_df.join(df,how='outer')

	print(main_all_stocks_together_df.head())
	main_all_stocks_together_df.to_csv('StockDataCSVSheetsUSED/AllStocksTogetherAdjustedClose.csv')

compileAllStocksTogetherInOneCSV()
