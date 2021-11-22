import pandas as pd

def copyTargetStock():
    df=pd.read_csv('Max_Hues/S&P500_Stock_Data/AAPL.csv')
    df1 = pd.DataFrame(df)
    df1.to_csv('Max_Hues/StockDataCSVSheetsUSED/TargetStockWithNoNewValues.csv', index=False)

copyTargetStock()