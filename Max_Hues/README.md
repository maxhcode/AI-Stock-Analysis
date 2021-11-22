# AI-Stock-Analysis

## Stock Data two types
The **first one** is in the folder **Data** which contains downloaded Stocks and ETFS until 2017.

The **second one** is in the folder **S&P500_Stock_Data** which contains 500 stocks from 2010 until today.  

## Requirements
First, the requirements.txt should be installed. After that, there is a file called **CudaActivatedCheck.py**, which tells you if Cuda is recognized or not. 
Note: You do not have to have Cuda installed to run this Project 

## 1.Get(S&P500)_Stock_Tickers (Getting the Stock Tickers)
**getTickersData.py** is a file in which gets all the S&P500 tickers from Wikipedia. This first needs to be executed to web scrap the stock data and save it in S&P500_Tickers.pickle.

## 2.Get(S&P500)_Stock_Data (Getting the Stock Data)
This folder contains one python file, which is called **getStockData.py**, which is web scrapping yahoo finance. 
In the file, you can set the start and end date of all the stocks that you want to scrap.
It takes its tickers from the file called **S&P500_Tickers.pickle**, which is being created in **getTickersData.py**. 

## 3.StockDataCompression (Compress Stock Data ready for usage)

1.**copyTargetStock.py** is copying the target stock file .csv and naming it TargetStockWithNoNewValues.csv file.

2.**compileTargetStock.py** is calculating the 50-day moving average, the rate increase in volume, the rate increase in adjusted close, and taking all Target stock values with its High, Low, Open, Close, Volume, Adj Close and Dates and all moving them into TargetStockWithAllValues.csv file.

3.**compileAllStocksTogether.py** is combining all of the downloaded stocks, in this case, the S&P500 with their adjusted close value into one AllStocksTogetherAdjustedClose.csv file.

4.**combineAllStocksWithTargetStock.py** is combining all of the downloaded stocks, in this case, the S&P500 with their adjusted close value and the chosen Target stock with its High, Low, Open, Close, Volume, Adj Close, Moving_av, Increase_in_vol, Increase_in_adj_close into one EverythingTogetherAll.csv file.

## 4.StockDataAnalysis (Stock Data Analysis after Stock Data Compression)

1.**checkAllCSVFileColumns.py** it is a check that after the stock data compression that everything went well, so it will print all of the columns values.

2.**visualizeTargetStockCandleSticks.py** it is showing you the target stock with Candles for each stock that you are Analysing

3.**analyseAllStocksPatterns.py** is reading the EverythingTogetherAll.csv file and creating a heatmap where you can see the correlation stocks have with each other and a second graph which is creating a bar graph. Both graphs are being created with all the 500 stocks adjusted close values and the target stock values, and the calculated values from before.

## 5.Leftover with Two Folders (SavedAIStockModels and StockDataCSVSheetsUSED and StockDataPredictionResults)
Those folders are quite self-explanatory, and the StockDataCSVSheetsUSED just save the CSV file, and the SavedAIStockModels one saves the Stock Models, and StockDataPredictionResults just has some pictures of some Results. 

## 6.After following the readme file in order to execute chosen Machine Learning Model
**linear_regression_model.py** which uses the not scraped stock data, which is the data in txt files. For the model, it uses linear regression as a model and then makes a prediction. It's quite an old file and quite messy.
**RNN.py** is a recurrent neural network model which is currently using 2 epochs for a quick check, but I would also recommend using 2000 epochs to get real results. I am currently using a batch size of 32, which worked pretty well compared to other sizes that I have tried.
