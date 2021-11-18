import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

def analyseAllStocksPatterns():
    #Reading all Stock data and Taregt stock file
    df_everything_together_all=pd.read_csv('StockDataCSVSheetsUSED/EverythingTogetherAll.csv')

    #PLOT
    #With corr() Any na values are automatically excluded. 
    C_mat = df_everything_together_all.corr()
    fig = plt.figure(figsize = (15,15))
    fig.figsize= (15,15)
    #Showing the coorlation patterns wih other stock compared to target stock
    #If you do plt.show() twice it is shown seperate otherwise all graphs shown at the same time
    sb.heatmap(C_mat, vmax = .8, square = True)
    plt.show()
    
    df_everything_together_all.hist(figsize = (35,35))
    plt.show()

analyseAllStocksPatterns()