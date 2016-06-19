from collections import OrderedDict

import pandas as pd
import numpy as np
import pandas.io.data as web
from math import *
import time

from pandas import DataFrame, Series
import matplotlib.pyplot as plt

def visualize_data(df, field_names):
    df[field_names].plot()
    plt.show()

def explore_data(file_name):
    df = pd.read_csv(file_name)
    df = df.iloc[::-1] #reverse data to have the right chronological order

    n_days = len(df.axes[0])
    n_features = len(df.axes[1]) - 1 #exclude last feature
    
    print df.describe()
    print "Number of days (samples) ", n_days
    print "Number of features ", n_features
    
    return df

def enhance_data(df):
    rolling_days = 30
    df['Bol_upper'] = pd.rolling_mean(df['Adj Close'], window=rolling_days) + 2 * pd.rolling_std(df['Adj Close'], rolling_days, min_periods=rolling_days)
    df['Bol_lower'] = pd.rolling_mean(df['Adj Close'], window=rolling_days) - 2 * pd.rolling_std(df['Adj Close'], rolling_days, min_periods=rolling_days)
    rm_adjustedClose = pd.rolling_mean(df['Adj Close'].shift(), window=rolling_days)

    df['Rolling'] = rm_adjustedClose  #rm_adjustedClose.fillna(df['Adj Close'])

        
    df = df[rolling_days:]
    return df

"""
    Build a q-learning model to maximize profit from a begin capital
    Think of the agent as a novice user with a starting capital
    He learns/act by trading everyday. The goal is to maximize 

    Action: hold, buy (how much?), sell (how much?)
    Reward: daily return

    State: quantized combination of
        adjusted close/ rolling mean
        Bollinger band (use both lower and upper)
    
    The effectiveness of the model can be judged by how much he/she has earnt
    after the model is learnt
"""
from trader_agent import TraderAgent

    
def Q_learning(df):
    df = enhance_data(df)

    trader = TraderAgent()
    for i in range(0, 2):        
        trader.learn(df, 100000) #learning

    #when we want to invest for a shorter period of time we need to use shorter rolling windows
    #result for a short time period
    #trader.trade(df[len(df) - 500: len(df) - 10], 100000)
    trader.trade(df, 100000)

#choose your own data set
    
#df = explore_data("data/Google.csv")
#df = explore_data("data/Apple.csv")
#df = explore_data("data/Intel.csv")
#df = explore_data("data/Microsoft.csv")
#df = explore_data("data/MicrosoftDotComBurst.csv")
#df = explore_data("data/ibmfrom2005.csv")
#df = explore_data("data/philips.csv")
#df = explore_data("data/shell.csv")
#df = explore_data("data/exxon.csv")
#df = explore_data("data/bankA.csv")

#df = explore_data("data/GE.csv") 
#df = explore_data("data/Walmart.csv")
#df = explore_data("data/McDonald.csv")
#df = explore_data("data/Siemens.csv")
df = explore_data("data/HP.csv")
#df = explore_data("data/yahoo.csv")
#df = explore_data("data/facebook.csv")
#df = explore_data("data/tesla.csv")


Q_learning(df)

