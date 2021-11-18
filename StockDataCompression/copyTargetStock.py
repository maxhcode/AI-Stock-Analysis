import pandas as pd

def copyTargetStock():
    df=pd.read_csv('S&P500_Stock_Data/AMZN.csv')
    df1 = pd.DataFrame(df)
    df1.to_csv('StockDataCSVSheetsUSED/TargetStockWithNoNewValues.csv', index=False)

copyTargetStock()