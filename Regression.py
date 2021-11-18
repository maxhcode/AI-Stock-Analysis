import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

class RegressionModelStockPrediction():

    def dataPreperationForRegression(self):
        #Reading All stocks and Target stock data
        df_EverythingTogetherAll=pd.read_csv('StockDataCSVSheetsUSED/EverythingTogetherAll.csv')

        #Get Rid of all NULL Values for zeros see print difference 
        #Inplace edits on the current dataframe
        df_EverythingTogetherAll.fillna(0, inplace=True)

        #Only select those values from the Target stock
        y_df_all_target_stock_values=df_EverythingTogetherAll[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
        col_y_all_target_stock_values=y_df_all_target_stock_values.columns #Select all columns ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

        #Values selected from above drop Adjusted Close and Volume 
        y_df_mod_all_target_stock_values=y_df_all_target_stock_values.drop(['Adj Close','Volume'],axis=1)
        #Left over in y_df_mod_all_target_stock_values with 'High', 'Low', 'Open', 'Close'

        #Now it only has the values in an array without any columns
        #Those Values are 'High', 'Low', 'Open', 'Close' 
        self.y_mod_all_values_target_stock = y_df_mod_all_target_stock_values.values 

        ###############Y Axis is done NOW ###########################


        #Conatining all the columns from above y
        #'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'
        drop_target_stock_columns=col_y_all_target_stock_values

        #Now list with all coolumn names'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'
        drop_target_stock_columns=drop_target_stock_columns.tolist()

        #Here the String 'Date' is added to the list
        drop_target_stock_columns.append('Date')

        #Now drop all columns in the list from the Target stock and only use all of the remaining stock values adj close values
        #Which leaves the Moving_av and Increase_in_vol and Increase_in_adj_close and all other stock values adj close
        X_df_all_stocks_some_target_stock=df_EverythingTogetherAll.drop(drop_target_stock_columns,axis=1)

        self.X_all_stocks_some_target_stock=X_df_all_stocks_some_target_stock.values

        #############X Axis Done Now################

        ##########IMPORTANT############
        #Just for understanding dataframes
        df1 = pd.DataFrame(self.X_all_stocks_some_target_stock) #All Moving_av  Increase_in_vol  Increase_in_adj_close and all other 195 stocks adjusted close together
        df2 = pd.DataFrame(self.y_mod_all_values_target_stock) #All Target Stock Prices 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'

    def prediction(self):
        #Dataset split 70% and 30% split for Training and Test data
        X_train, X_test, y_train, y_test = train_test_split(self.X_all_stocks_some_target_stock, self.y_mod_all_values_target_stock, test_size=0.3) 

        print(train_test_split(self.X_all_stocks_some_target_stock, self.y_mod_all_values_target_stock, test_size=0.3))
    
        #Notes: Frank try smaller nodes for dense layers
        def model():
            #Depending on what size of the S&P 500 you have choosen in the getTickersData.py that should be the number of the amount of tickers
            mod=Sequential() 
            mod.add(Dense(32, kernel_initializer='normal',input_dim = 500, activation='relu'))
            mod.add(Dense(64, kernel_initializer='normal',activation='relu'))
            mod.add(Dense(128, kernel_initializer='normal',activation='relu'))
            mod.add(Dense(256, kernel_initializer='normal',activation='relu'))
            mod.add(Dense(4, kernel_initializer='normal',activation='linear'))
            
            mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
            mod.summary()
            return mod

        model()

        regressor = KerasRegressor(build_fn=model, batch_size=16,epochs=10) #16

        callback=tf.keras.callbacks.ModelCheckpoint(filepath='SavedAIStockModels/Regressor_model.h5',monitor='mean_absolute_error',verbose=0,save_best_only=True,save_weights_only=False,mode='auto')

        es = EarlyStopping(monitor='val_loss', mode='min')
        results=regressor.fit(X_train,y_train, callbacks=[es, callback])

        #All Target Stock Prices 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'
        y_target_stock_prediction=regressor.predict(X_test)
        print(y_target_stock_prediction)

        self.y_mod_target_stock_prediction=[]
        self.y_mod_test_target_stock_prediction=[]
        for i in range(0,4):
            j=0
            #Each iterations the Array will be empty again
            #For each Graph to keep it seperate 
            y_mod_target_stock_prediction_array=[]
            y_mod_test_target_stock_prediction_array=[]
            #y_test is the tested set in this 30%
            while(j<len(y_test)):
                #i for each iteration  of for loop
                #j for the entrie length of the test set
                y_mod_target_stock_prediction_array.append(y_target_stock_prediction[j][i])
                y_mod_test_target_stock_prediction_array.append(y_test[j][i])
                j+=1
                
            self.y_mod_target_stock_prediction.append(np.array(y_mod_target_stock_prediction_array))
            self.y_mod_test_target_stock_prediction.append(np.array(y_mod_test_target_stock_prediction_array))

    def graphPrediction(self):
        fig, ax = plt.subplots()
        ax.scatter(self.y_mod_test_target_stock_prediction[0], self.y_mod_target_stock_prediction[0])
        ax.plot([self.y_mod_test_target_stock_prediction[0].min(),self.y_mod_test_target_stock_prediction[0].max()], [self.y_mod_test_target_stock_prediction[0].min(), self.y_mod_test_target_stock_prediction[0].max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(self.y_mod_test_target_stock_prediction[1], self.y_mod_target_stock_prediction[1])
        ax.plot([self.y_mod_test_target_stock_prediction[1].min(),self.y_mod_test_target_stock_prediction[1].max()], [self.y_mod_test_target_stock_prediction[1].min(), self.y_mod_test_target_stock_prediction[1].max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(self.y_mod_test_target_stock_prediction[2], self.y_mod_target_stock_prediction[2])
        ax.plot([self.y_mod_test_target_stock_prediction[2].min(),self.y_mod_test_target_stock_prediction[2].max()], [self.y_mod_test_target_stock_prediction[2].min(), self.y_mod_test_target_stock_prediction[2].max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(self.y_mod_test_target_stock_prediction[3], self.y_mod_target_stock_prediction[3])
        ax.plot([self.y_mod_test_target_stock_prediction[3].min(),self.y_mod_test_target_stock_prediction[3].max()], [self.y_mod_test_target_stock_prediction[3].min(), self.y_mod_test_target_stock_prediction[3].max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

RegressionPredict = RegressionModelStockPrediction()
RegressionPredict.dataPreperationForRegression()
RegressionPredict.prediction()
RegressionPredict.graphPrediction()