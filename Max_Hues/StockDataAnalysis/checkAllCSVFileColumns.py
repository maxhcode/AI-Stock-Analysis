#Consider to use test here to check automatically that everything is correct here
import pandas as pd

def checkAllCSVFileColumns():
    df_TargetStockWithNoNewValues=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/TargetStockWithNoNewValues.csv')
    col_TargetStockWithNoNewValues=df_TargetStockWithNoNewValues.columns
    print(col_TargetStockWithNoNewValues)

    df_TargetStockWithAllValues=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/TargetStockWithAllValues.csv')
    col_TargetStockWithAllValues=df_TargetStockWithAllValues.columns
    print(col_TargetStockWithAllValues)

    df_EverythingTogetherAll=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/EverythingTogetherAll.csv')
    col_EverythingTogetherAll=df_EverythingTogetherAll.columns
    print(col_EverythingTogetherAll)

checkAllCSVFileColumns()