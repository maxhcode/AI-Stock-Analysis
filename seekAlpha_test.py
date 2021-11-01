from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas_ta
import os

parentDir = os.getcwd()

class test_data():

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self):
        self.df = pd.read_csv(self.file_path)
        
        self.df['Open'] = self.df.Open.astype(float)
        self.df['High'] = self.df.High.astype(float)
        self.df['Low'] = self.df.Low.astype(float)
        self.df['Volume'] = self.df.Volume.astype(float)
        #self.df['OpenInt'] = self.df.OpenInt.astype(int) #Not working with some stocks
        self.df['Close'] = self.df.Close.astype(float)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    

        self.df.set_index(pd.DatetimeIndex(self.df['Date']), inplace=True)
        self.df = self.df.drop(['Open', 'High', 'Low', 'Volume', 'OpenInt', 'Date'], axis=1)
        
        #self.df.plot(y='Close', color='orange', linewidth=3, alpha=0.5)

        self.df.ta.ema(Close='Close', length=10, append=True)
        self.df = self.df.iloc[10:] #drop the first 10 rows
        #print(self.df.shape)
        #self.df.plot(y=['Close','EMA_10'], color=['black','orange'], style='.', linewidth=3)
        # Split data into testing and training sets
        
        
    #def csv_TSLA(self):
    #    self.df = pd.read_csv(parentDir+'/Data/tsla.us.txt')
    #    self.df.set_index(pd.DatetimeIndex(self.df['Date']), inplace=True)
    #    self.df = self.df[['Adj Close']]
    #    self.df['Adj Close'] = self.df['Adj Close'].astype(float)
    #    self.df.ta.ema(close='Adj Close', length=10, append=True)
    #    #print(self.df.head(10))
    #    self.df = self.df.iloc[10:]
    #    #print(self.df.head(10))
#
    def linear_regression(self):
        # Split data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(self.df[['Close']], self.df[['EMA_10']], test_size=.2)
        #X_train, X_test, y_train, y_test = train_test_split(self.df[['Adj Close']], self.df[['EMA_10']], test_size=.2)
        #print(X_test.describe())
        #print(X_train.describe())
        ## Test set
        #print(X_test.describe())
        ## Training set
        #print(X_train.describe())
        
        model = LinearRegression()
        # Train the model
        #80% percentage of the data 
        #194 samples 
        model.fit(X_train, y_train)

        #With 80 percent of the data train and get a pattern etc 
        #With 20% of the data test it and see if it you can make assumations
        #give the 20% of x to predict y 
        #Now I can check the 20% of Y with the 20% I have previusoly split and compare them
        y_pred = model.predict(X_test)
        #print(y_pred)
        
        # Printout relevant metrics

        #Here I now compare the real value with the predcition and calculate the absoult error and coefficients
        print("Model Coefficients:", model.coef_)
        print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("Coefficient of Determination:", r2_score(y_test, y_pred))

        #print(X_train) #194 samples 80%
        #print(X_test) #other half 20%
        #print(y_test) #EMA_10 smaller 20%
        #print(y_train) #EMA_10 bigger 80%
        #print(y_pred)
        #plt.scatter(y_pred)
        plt.scatter(X_test, y_test,  color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        
        plt.show()

    def ridge():
        pass


t = test_data(file_path=parentDir+"/Data/Stocks/aapl.us.txt")
t.get_data()
t.linear_regression()
 