from collections import OrderedDict
import random
import math
import pandas as pd
import numpy as np
import time
from math import *
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

class TraderAgent:
    
    #may need to give better default value for "hold" to prevent constant buying/ selling (longer term investment)?
    
    DEFAULT_Q_VALUE_HOLD = 0
    DEFAULT_Q_VALUE_BUY = 10 # in the beginning we need to put more "motivation" for buying, because stock value tends to increase over time
    #and because we have money to invest
    DEFAULT_Q_VALUE_SELL = 0
    
    LEARNING_DECAY_RATE = 0.005 
    DISCOUNT_FACTOR = 0.1 #0 long term, 1 short term
    EXPLOITATION_INCREASE_RATE = 0.005
    MAX_EXPLORE_PERCENTAGE = 3
    
    #for "blue-chip" stocks generally we want to buy early a lot, and sell a little?
    #in the beginning needs to put more motivation for buying?
    
    SELL_FACTOR_EACH_TRADE = 20 #1/Factor of total stocks (sell a bit less because we have money)
    BUY_FACTOR_EACH_TRADE = 20 #1/factor of total money #for growing stock, the earlier to buy the better
    SUB_STATE_QUANTIZE_RESOLUTION = 5
    
    ACTIONS = ['sell', 'buy', 'hold']
    
    def __init__(self):
        self.available_money = 0
        self.original_money = 0
        self.holding_stock = 0

        self.nr_explores = 0
        self.nr_exploits = 0
        
        self.state = OrderedDict()
        self.q_table = OrderedDict()
        #self.q_stats = OrderedDict() 
        
        self.learning_rate = 1.0 #begin as an "eager" learner, will decay over time
        self.exploitation_factor = 0.0 #beginner has nothing to exploit, only explore

        for i in range(0, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION+1):
            for j in range(0, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION+1):
                for k in range(0, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION+1):
            
                    self.q_table[((i, j, k), 'buy')] = TraderAgent.DEFAULT_Q_VALUE_BUY
                    
                    self.q_table[((i, j, k), 'sell')] = TraderAgent.DEFAULT_Q_VALUE_SELL
                    self.q_table[((i, j, k), 'hold')] = TraderAgent.DEFAULT_Q_VALUE_HOLD

    def quantize(self, value, expected_min, expected_max, resolution):
        result = int((value - expected_min) * resolution / (expected_max - expected_min))
        result = min(max(0, result), resolution) #clamp

        return result

    def combine_states(self, adj_closed_rolling_ratio, adj_closed_bol_low, adj_closed_bol_up, earning_ratio):
        #since in priciple we do not know what the "future" looks like, we do not know the whole min/max
        #we just have to guess on a reasonable min/max
        #quantize state space should be dependent on the number of data sample available
        closed_rolling_quantized = self.quantize(adj_closed_rolling_ratio, 0.5, 1.5, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION)
        bol_low_quantized = self.quantize(adj_closed_bol_low, 1, 1.5, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION)
        bol_up_quantized = self.quantize(adj_closed_bol_up, 0.6, 1, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION)
        #TODO earning ratio should be quantized based on expected market return?
        #e.g. for a long historcal time (e.g. apple, microsoft, the max can be 300)
        earning_ratio_quantized = self.quantize(earning_ratio, 0, 2, TraderAgent.SUB_STATE_QUANTIZE_RESOLUTION)

        #test removing earning ratio from state, earning ratio is sunk cost
        #return (closed_rolling_quantized, bol_low_quantized, bol_up_quantized, earning_ratio_quantized)
        return (closed_rolling_quantized, bol_low_quantized, bol_up_quantized)
    
    def get_state(self, data, holding_stock, available_money):
        adj_closed_rolling_ratio= data['Adj Close'] / data['Rolling']

        adj_closed_bol_low = data['Adj Close'] / data['Bol_lower']
        adj_closed_bol_up = data['Adj Close'] / data['Bol_upper']

        holding_stock_value = holding_stock * data['Adj Close'] #not realtime, just take the price at the end of the day

        earning_ratio = (holding_stock_value + available_money) / self.original_money

        return self.combine_states(adj_closed_rolling_ratio, adj_closed_bol_low, adj_closed_bol_up, earning_ratio)

    def get_next_action(self, state):
        #balance exploration (randomness) and exploitation (from Q-learning table)
        next_action = 'hold' #default, doing nothing
        exploitation_rate = TraderAgent.EXPLOITATION_INCREASE_RATE #increase overtime
        self.exploitation_factor = min(0.99, self.exploitation_factor + exploitation_rate)
        
        if random.randint(0, int(100 * self.exploitation_factor)) < TraderAgent.MAX_EXPLORE_PERCENTAGE: #explore
            next_action = random.choice(TraderAgent.ACTIONS)
            self.nr_explores = self.nr_explores + 1
            
        else: #exploit
            self.nr_exploits = self.nr_exploits + 1
            max_Q_value = 0
            #find the action that has highest q value
            #TODO: should we validate amount of available stocks + money available?

            for action in TraderAgent.ACTIONS:
                q_key = (state, action)
                #if self.q_table.has_key(q_key) == False: #first time in this state, action
                #    self.q_table[q_key] = TraderAgent.DEFAULT_Q_VALUE

                if self.q_table[q_key] > max_Q_value:
                    max_Q_value = self.q_table[q_key]
                    next_action = action
                    #print "Find a Q value ", max_Q_value , " action " , action

        if next_action == 'sell' and self.holding_stock < 5: #prevent some basic stupidity
            next_action = 'hold'

        #next_action = 'buy' #test
        #next_action = random.choice(TraderAgent.ACTIONS)
        #TODO: should we use relevant rule from technical analysis when a certain rule is not
        #available? 
        return next_action

    def update_Q_table(self, current_state, action, reward, state_after_action):

        min_learning_rate = 0.01
        self.learning_rate = max(min_learning_rate, self.learning_rate - TraderAgent.LEARNING_DECAY_RATE)

        q_key = (current_state, action)

        max_next_state = 0
        for next_action in TraderAgent.ACTIONS:
            q_key_next_state= (state_after_action, next_action)

            max_next_state = max(max_next_state, self.q_table[q_key_next_state])

        self.q_table[q_key] = (1 - self.learning_rate) * self.q_table[q_key] + self.learning_rate * (reward + TraderAgent.DISCOUNT_FACTOR * max_next_state)        
                                                    
    def calculate_reward_update_asset(self, action, data):
        #ah calculation is wrong. number of stocks can be splitted
        #reward diff can not be used between Open and Adj Close (does not take split into account)
        #for calculating reward and asset update, we use "close". shall be corrected later
        #(for number of stocks) when we find split of stocks
        
        current_stock = self.holding_stock
        current_money = self.available_money

        #current value is taken at the beginning of the day
        current_total_value = current_stock * data['Open'] + current_money

        if action == 'hold':
            new_value = current_stock * data['Close'] + current_money
            reward = new_value - current_total_value
                        
        elif action == 'buy':
            if current_money < data['Open']:
                reward = 0 #penalty for stupidity
            else:     
                money_allocated = current_money / TraderAgent.BUY_FACTOR_EACH_TRADE
                
                if money_allocated < data['Open']: #this should not happen anymore, already checked by validating action
                    money_allocated = current_money

                if money_allocated < data['Open']:
                    reward = -10 #penalty for stupidity (no money and still try to buy)
                else:
                    nr_stock_buy = int(money_allocated / data['Open'])
                    money_spent = nr_stock_buy * data['Open']
                    #update asset and reward
                    self.available_money = self.available_money - money_spent
                    self.holding_stock = self.holding_stock + nr_stock_buy
                    new_value = self.holding_stock * data['Close'] + self.available_money
                    reward = new_value - current_total_value
        elif action == 'sell':
            nr_stock_to_sell = 0
            if current_stock == 0:
                reward = 0 #penalty for stupidity (no stocks and still try to sell)
            elif current_stock < TraderAgent.SELL_FACTOR_EACH_TRADE:
                nr_stock_to_sell = current_stock
            else:
                nr_stock_to_sell = current_stock / TraderAgent.SELL_FACTOR_EACH_TRADE

            if nr_stock_to_sell > 0: 
                money_gained = nr_stock_to_sell * data['Open']
                self.holding_stock = self.holding_stock - nr_stock_to_sell
                self.available_money = self.available_money + money_gained

                new_value = self.holding_stock * data['Close'] + self.available_money
                reward = new_value - current_total_value

        return reward

    #give a data frame to learn
    def learn(self, df, capital):
        self.nr_explores = 0
        self.nr_exploits = 0

        self.available_money = capital
        self.original_money = self.available_money
        self.holding_stock = 0

        last_price = 0.0
        first_price = -1.0
        rolling_begin = -1.0
        rolling_end = 0.0
        yesterday_adjustment_factor = -1.0
        
        #create a new series with total assets overtime
        assets = Series(range(len(df)))
        cash = Series(range(len(df)))
        stock_value = Series(range(len(df)))
        for (index, row) in df.iterrows():
            #somehow can not index df with df.ix[row, 'col']
            if first_price < 0:
                first_price = row['Adj Close']
                rolling_begin = row['Rolling']
            #correct holding stock with split
            adjustment_factor = row['Adj Close'] / row['Close']
            split_factor = adjustment_factor / yesterday_adjustment_factor
            if yesterday_adjustment_factor > 0 and split_factor < 0.98 or split_factor > 1.02:
                split_factor = adjustment_factor / yesterday_adjustment_factor
                #print "Stock split ", split_factor , ' date ', row['Date']
                self.holding_stock = self.holding_stock * split_factor
                
            yesterday_adjustment_factor = adjustment_factor
            current_state = self.get_state(row, self.holding_stock, self.available_money)
            action = self.get_next_action(current_state)

            reward = self.calculate_reward_update_asset(action, row)
            
            next_state = self.get_state(row, self.holding_stock, self.available_money)

            self.update_Q_table(current_state, action, reward, next_state)
            
            last_price = row['Adj Close']
            rolling_end = row['Rolling']
            
            assets.loc[index] = self.holding_stock * row['Adj Close'] + self.available_money
            cash.loc[index] = self.available_money
            stock_value.loc[index] = self.holding_stock * row['Adj Close']

        df.loc[:,'ROI'] = assets / (capital / rolling_begin) #normalize begin point
        df.loc[:,'Cash'] = cash / (capital / rolling_begin) #normalize begin point
            
        df.loc[:,'Stock owned'] = stock_value / (capital / rolling_begin) #normalize begin point
        
        new_value = self.holding_stock * last_price + self.available_money

        print '*****************'
        print 'Original money ', self.original_money, " Holding money ", self.available_money, " Holding stock ", self.holding_stock , ' new value ', new_value
        #calculate market gain based on rolling?
        print 'Earning ratio ', (new_value ) / (self.original_money), ' Market gain ', (rolling_end ) / (rolling_begin)
        print '*****************'

        print "Number of Q value ", len(self.q_table.keys()) , ' nr explores ' , self.nr_explores , ' nr exploits ', self.nr_exploits

        return df
    
    #become a real trader
    def trade(self, df, capital):

        result = self.learn(df, capital) #maybe still do learning with another factor

        #for key in self.q_table.keys():
        #    print key , " => " , self.q_table[key]
            
        result[['Rolling', 'ROI', 'Stock owned', 'Cash']].plot()
        plt.show()

    
