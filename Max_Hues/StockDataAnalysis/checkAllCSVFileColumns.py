#Consider to use test here to check automatically that everything is correct here
import pandas as pd

def checkAllCSVFileColumns():
    df_TargetStockWithNoNewValues=pd.read_csv('StockDataCSVSheetsUSED/TargetStockWithNoNewValues.csv')
    col_TargetStockWithNoNewValues=df_TargetStockWithNoNewValues.columns
    print(col_TargetStockWithNoNewValues)

    df_TargetStockWithAllValues=pd.read_csv('StockDataCSVSheetsUSED/TargetStockWithAllValues.csv')
    col_TargetStockWithAllValues=df_TargetStockWithAllValues.columns
    print(col_TargetStockWithAllValues)

    df_EverythingTogetherAll=pd.read_csv('StockDataCSVSheetsUSED/EverythingTogetherAll.csv')
    col_EverythingTogetherAll=df_EverythingTogetherAll.columns
    print(col_EverythingTogetherAll)

checkAllCSVFileColumns()