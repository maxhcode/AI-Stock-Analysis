import pandas as pd

def complieTargetStock():
    #Reading Target Stock in a dataframe 
    df=pd.read_csv('Max_Hues/S&P500_Stock_Data/AAPL.csv',index_col=0,parse_dates=True)

    #Adds 'Moving_av' as a column to the df dataframe
    #Which is the 50 day moving average
    #It takes the Adjusted Close and takes the mean of the Adjusted Close 
    #Actually, it is the mean of the i-50 to i values of the “Adj Close” values for the index
    df['Moving_av']= df['Adj Close'].rolling(window=50,min_periods=0).mean()
    
    #Rate of increase in volume, Rate of increase in Adjusted Close for the stock
    i=1
    #Create Arrays for Rate of increase in volume, Rate of increase in Adjusted Close
    rate_increase_in_volume=[0]
    rate_increase_in_adjusted_close=[0]
    while i<len(df):
        #It takes the current volume minus the previous volume
        #It takes the current adjusted close minues the previous adjusted close
        rate_increase_in_volume.append(df.iloc[i]['Volume']-df.iloc[i-1]['Volume'])
        rate_increase_in_adjusted_close.append(df.iloc[i]['Adj Close']-df.iloc[i-1]['Adj Close'])
        i+=1

    #Now add the arrays with the Rate of increase in volume, Rate of increase in Adjusted Close
    #Add To the Columns in the Dataframs
    df['Increase_in_vol']=rate_increase_in_volume
    df['Increase_in_adj_close']=rate_increase_in_adjusted_close

    #JUST FOR PLOTING
    #df['Moving_av'].plot()
    #df['Increase_in_vol'].plot()
    #df['Increase_in_adj_close'].plot()
    #plt.show()

    print(df)
    #Exporting all the Target stock data to a csv 
    df.to_csv("Max_Hues/StockDataCSVSheetsUSED/TargetStockWithAllValues.csv",index=False)
    
    #########ADDING DATES############
    
    df1=pd.read_csv('Max_Hues/S&P500_Stock_Data/AAPL.csv')
    print(df1.columns)

    Dates=[]
    i=0
    while i<len(df1):
        Dates.append(df1.iloc[i]['Date'])
        i+=1

    df2=pd.read_csv('Max_Hues/StockDataCSVSheetsUSED/TargetStockWithAllValues.csv')
    df2['Date']=Dates
    print(df2.columns)
    df2.to_csv("Max_Hues/StockDataCSVSheetsUSED/TargetStockWithAllValues.csv",index=False)

complieTargetStock()
