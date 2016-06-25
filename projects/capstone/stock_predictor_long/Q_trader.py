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
    
    return df

def enhance_data(df):
    rolling_days = int(len(df) / 20)
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


number_days_max = 540 #max data set around 2 years (working days)
#the more different types of stocks we train the agent, the more "range" of data it has (e.g. more state
# can be reached and train: e.g. adjust price / rolling range is defined between 0.5 and 1.5, we need both
# high performers and poor performers to reach the wider range
stocks_train = ['Walmart', 'HP', 'yahoo', 'Philips', 'shell', 'Google', 'facebook', 'tesla', 'Apple']
stocks_trade =stocks_train

trader = TraderAgent()
money_to_learn = 100000
money_to_trade = money_to_learn 

#learning (with more data set)
for stock_name in stocks_train:
    print '********Learning with data set ', stock_name
    df = explore_data('data/'+ stock_name +'.csv')
    df = df[max(len(df) - number_days_max, 0): len(df)]
    df = enhance_data(df)
    #put True to second parameter for visualization
    trader.learn(df, money_to_trade)
    print "*********************************"


#trading
for stock_name in stocks_trade:
    print '********trading with data set ', stock_name
    df = explore_data('data/'+ stock_name +'.csv')
    df = df[max(len(df) - number_days_max, 0): len(df)]
    df = enhance_data(df)
    #put True to second parameter for visualization
    trader.trade(df, money_to_trade, True, stock_name)
    print "*********************************"



