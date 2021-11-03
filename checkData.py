#Consider to use test here to check automatically that everything is correct here
import pandas as pd

df1=pd.read_csv('dataset_target.csv')
col1=df1.columns
print(col1)

df2=pd.read_csv('dataset_target_2.csv')
col2=df2.columns
print(col2)

df3=pd.read_csv('Dataset_main.csv')
col3=df3.columns
print(col3)

import seaborn as sb
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

C_mat = df3.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()