import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

#Reading Data
df3=pd.read_csv('Dataset_main.csv')
col3=df3.columns
print(col3)
print(df3)

#Get Rid of all NULL Values for zeros see print difference 
#Inplace edits on the current dataframe
df3.fillna(0, inplace=True)

print(df3)

#Only select those values 
#Which only are the values of the amazon stock
y_df=df3[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
col_y=y_df.columns #Select all columns ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
print(y_df)

#Values selected from above drop Adjusted Close and Volume 
y_df_mod=y_df.drop(['Adj Close','Volume'],axis=1)
print(y_df_mod.columns)

#Left over in y_df_mod with 'High', 'Low', 'Open', 'Close'

#Now it only has the values in an array wihtout any columns
#Those Values are 'High', 'Low', 'Open', 'Close' 
y = y_df_mod.values 
print(y) 

###############Y Axis is done NOW ###########################


#Conatining all the columns from above y
#'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'
Drop_cols=col_y
print(Drop_cols) 

#Moved all the columns from y into a list with just the string names of the columns
#'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'
Drop_cols=Drop_cols.tolist()
print(Drop_cols)

#Here the String 'Date' is added to the list
Drop_cols.append('Date')
print(Drop_cols)


X_df=df3.drop(Drop_cols,axis=1)
print(X_df)
print(X_df.columns)

X=X_df.values
print(X)

#Just for understanding dataframes
df2 = pd.DataFrame(X) #All Moving_av  Increase_in_vol  Increase_in_adj_close and all other 195 stocks adjusted close together
df4 = pd.DataFrame(y) #All Amazon Stock Prices


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test2 = pd.DataFrame(X_test)
X_train2 = pd.DataFrame(X_train)
y_test2 = pd.DataFrame(y_test)
y_train2 = pd.DataFrame(y_train)

#train_x = X_train2.plot()
#test_x = X_test2.plot()
#train_y = y_train2.plot()
#test_y = y_test2.plot()

#train_y.set_title('y_train2')
#test_y.set_title('y_test2')
#train_x.set_title('X_train2')
#test_x.set_title('X_test2')

#plt.show()
print(train_test_split(X, y, test_size=0.3))



from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import accuracy_score

def model():
    mod=Sequential()  #used to be 200
    mod.add(Dense(32, kernel_initializer='normal',input_dim = 195, activation='relu'))
    mod.add(Dense(64, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(128, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(256, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(4, kernel_initializer='normal',activation='linear'))
    
    mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    mod.summary()
    #Model and the nodes contain that are shown above 
    return mod

model()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=model, batch_size=16,epochs=100000) #16

#model is the model itself the neuoitons
#batch_size is the memory size it goes through the entire so with 5 it would be 
    #5 and then going through the next 5
#epochs is how monay total circles it will make 

import tensorflow as tf
callback=tf.keras.callbacks.ModelCheckpoint(filepath='Regressor_model.h5',monitor='mean_absolute_error',verbose=0,save_best_only=True,save_weights_only=False,mode='auto')
#This is using the callback defined varibale above
#ModelCheckpoint is used to save the state of the machine learni9ng model in a file
#ModelCheckpoint is used together with the fit function

#results regressor.fit 
#Goes throught the epocho
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min')
results=regressor.fit(X_train,y_train, callbacks=[es, callback])

y_pred= regressor.predict(X_test)
print(y_pred)

#print(X)
#print(y)

import numpy as np
y_pred_mod=[]
y_test_mod=[]
for i in range(0,4):
    j=0
    y_pred_temp=[]
    y_test_temp=[]
    #y_test is the tested set in this 30%
    while(j<len(y_test)):
        y_pred_temp.append(y_pred[j][i])
        y_test_temp.append(y_test[j][i])
        j+=1
        
    
    y_pred_mod.append(np.array(y_pred_temp))
    y_test_mod.append(np.array(y_test_temp))


fig, ax = plt.subplots()
ax.scatter(y_test_mod[0], y_pred_mod[0])
ax.plot([y_test_mod[0].min(),y_test_mod[0].max()], [y_test_mod[0].min(), y_test_mod[0].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test_mod[1], y_pred_mod[1])
ax.plot([y_test_mod[1].min(),y_test_mod[1].max()], [y_test_mod[1].min(), y_test_mod[1].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test_mod[2], y_pred_mod[2])
ax.plot([y_test_mod[2].min(),y_test_mod[2].max()], [y_test_mod[2].min(), y_test_mod[2].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test_mod[3], y_pred_mod[3])
ax.plot([y_test_mod[3].min(),y_test_mod[3].max()], [y_test_mod[3].min(), y_test_mod[3].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
### Regression Complete