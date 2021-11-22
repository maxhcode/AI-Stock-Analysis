import pandas as pd

def combineAllStocksWithTargetStock():
	df_TargetStockWithAllValues=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/TargetStockWithAllValues.csv',index_col='Date')
	df_target_stock=pd.read_csv('Max_Hues/S&P500_Stock_Data/GOOG.csv')
	df_AllStocksTogetherAdjustedClose=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/AllStocksTogetherAdjustedClose.csv',index_col='Date')
	
	#Creates an Array for Dates for Target Stock
	Dates=[]
	i=0
	#Appendes Dates to Array from Target Stock 
	while i<len(df_target_stock):
		Dates.append(df_target_stock.iloc[i]['Date'])
		i+=1
	
	#The current and already existing datframe would be joined with "outher"
	#The outer keyword returns all records when there is a match in left (table1) or right (table2) table records
	df_new=df_TargetStockWithAllValues.join(df_AllStocksTogetherAdjustedClose,how='outer')
	#fillna to fill out the missing values in the given series object with NaN
	df_new.fillna(0.0)

	df_new['Date']=Dates

	df_new.to_csv('Max_Hues/StockDataCSVSheetsUSED/EverythingTogetherAll.csv',index=False)

combineAllStocksWithTargetStock()

