import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from keras.layers import CuDNNLSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

class RNNModelStockPrediction():
    def dataSet1PreperationForRNN(self):
        ####Reading Target Stock with all values and new values####
        df_TargetStockWithAllValues=pd.read_csv('StockDataCSVSheetsUSED/TargetStockWithAllValues.csv')

        ####DROP DATE from df_TargetStockWithAllValues####
        self.df_TargetStockWithAllValuesWithoutDates=df_TargetStockWithAllValues.drop(['Date'],axis=1)

        ####SELECT ONLY THE FIRST 2200 for df_train####
        df_target_stock_train_set=self.df_TargetStockWithAllValuesWithoutDates[:2200]

        
        self.sc = MinMaxScaler(feature_range = (0, 1))

        ####Only select High, Low, Open, Close####
        df_target_stock_train_set_high_low_open_close=df_target_stock_train_set[['High','Low','Open','Close']]

        ####Takes all stock values####
        train_stock_set=df_target_stock_train_set.values
        ####Has all Stock values from Target stock####
        target_stock_set=df_target_stock_train_set_high_low_open_close.values

        ###This will format it into normalized data will convert data to 0 and 1###
        training_set_scaled = self.sc.fit_transform(train_stock_set)
        target_set_scaled = self.sc.fit_transform(target_stock_set)

        #We need to create the train and test set for the LSTM model.
        #Append the fit ransform / normalized data into an array / numpy array 
        self.X_train = []
        self.y_train = []
        for i in range(50,len(train_stock_set)):
            self.X_train.append(training_set_scaled[i-50:i,:])
            self.y_train.append(target_set_scaled[i,:])
            
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        #Shape of our Data to see the shape of the data
        print(self.X_train.shape)
        print(self.y_train.shape)

    def dataSet2PreperationForRNN(self):

        #This is selecting the other paret from 2200 to 2636
        df_target_stock_test_set=self.df_TargetStockWithAllValuesWithoutDates[2200:]

        #From 2200 to 2636 select only High, Low, Open, Close
        df_target_stock_test_set_high_low_open_close=df_target_stock_test_set[['High','Low','Open','Close']]

        #Select only the values of High, Low, Open, Close values
        self.target_set_test=df_target_stock_test_set_high_low_open_close.values

        #Select only the values of High, Low, Open, Close values
        test_set=df_target_stock_test_set.values

        #Now it normalized the data again from 0,1 and is making a prediction on the 50 days
        #On the trianed data
        test_set_scaled = self.sc.fit_transform(test_set)
        target_set_scaled = self.sc.fit_transform(self.target_set_test)

        #Now doing the same as before again and adding the 2200 from 2636 to the set
        self.X_test = []
        y_test = []
        for i in range(50,len(test_set)):
            self.X_test.append(test_set_scaled[i-50:i,:])
            y_test.append(target_set_scaled[i,:])
            
        self.X_test, y_test = np.array(self.X_test), np.array(y_test) 

    def prediction(self):        
        #Change from LSTM to CuDNNLSTM
        def model():
            mod=Sequential()
            mod.add(CuDNNLSTM(units = 64, return_sequences = True, input_shape = (self.X_train.shape[1], 9)))
            mod.add(Dropout(0.2))
            mod.add(BatchNormalization())
            mod.add(CuDNNLSTM(units = 64, return_sequences = True))
            mod.add(Dropout(0.1))
            mod.add(BatchNormalization())
        
            mod.add((CuDNNLSTM(units = 64)))
            mod.add(Dropout(0.1))
            mod.add(BatchNormalization())
            mod.add((Dense(units = 16, activation='tanh')))
            mod.add(BatchNormalization())
            mod.add((Dense(units = 4, activation='tanh')))
            mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_squared_error'])
            mod.summary()
            
            return mod

        #Show current RNN Model
        RNN_model=model()

        #Execute RNN Model and run it 

        callback=tf.keras.callbacks.ModelCheckpoint(filepath='SavedAIStockModels/RNN_model.h5',
                                                monitor='mean_squared_error',
                                                verbose=0,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='auto',
                                                save_freq='epoch')
        RNN_model.fit(self.X_train, self.y_train, epochs = 2, batch_size = 32,callbacks=[callback])
        #Was set to 2000 epochs
        #batch size 32
        self.dataSet2PreperationForRNN()
        ######## Predciting the values with trained mode ########
        self.predicted_stock_price = RNN_model.predict(self.X_test)

        self.predicted_stock_price = self.sc.inverse_transform(self.predicted_stock_price)

        print(self.predicted_stock_price)

    def graphPrediction(self):
        #High PLOT
        plt.figure(figsize=(20,10))
        plt.plot(self.target_set_test[0], color = 'green', label = 'Real Targeted Stock stock')
        plt.plot(self.predicted_stock_price[0], color = 'red', label = 'Predicted Targeted Stock Price')
        plt.title('Targeted Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Targeted Stock Price')
        plt.legend()
        plt.show()

        #Low Plot
        plt.figure(figsize=(20,10))
        plt.plot(self.target_set_test[1], color = 'green', label = 'Real Targeted Stock stock')
        plt.plot(self.predicted_stock_price[1], color = 'red', label = 'Predicted Targeted Stock Price')
        plt.title('Targeted Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Targeted Stock Price')
        plt.legend()
        plt.show()

        #Open Plot 
        plt.figure(figsize=(20,10))
        plt.plot(self.target_set_test[2], color = 'green', label = 'Real Targeted Stock stock')
        plt.plot(self.predicted_stock_price[2], color = 'red', label = 'Predicted Targeted Stock Price')
        plt.title('Targeted Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Targeted Stock Price')
        plt.legend()
        plt.show()

        #Close Plot
        plt.figure(figsize=(20,10))
        plt.plot(self.target_set_test[3], color = 'green', label = 'Real Targeted Stock stock')
        plt.plot(self.predicted_stock_price[3], color = 'red', label = 'Predicted Targeted Stock Price')
        plt.title('Targeted Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Targeted Stock Price')
        plt.legend()
        plt.show()

        #All over Plot
        plt.figure(figsize=(20,10))
        plt.plot(self.target_set_test, color = 'green', label = 'Real Targeted Stock stock')
        plt.plot(self.predicted_stock_price, color = 'red', label = 'Predicted Targeted Stock Price')
        plt.title('Targeted Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Targeted Stock Price')
        plt.legend()
        plt.show()

RNN = RNNModelStockPrediction()
RNN.dataSet1PreperationForRNN()
#RNN.dataSet2PreperationForRNN() # is being executed while predicting 
RNN.prediction()
RNN.graphPrediction()