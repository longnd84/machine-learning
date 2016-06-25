from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

import pandas.io.data as web
from math import *
import time

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

feature_cols = ['Open','High','Low','Close','Volume','Rolling','Adj Close Prev']
#feature_cols = ['Open','High','Low','Close','Volume']
number_days_max = 270 #max data set around 1 year (working days)
number_days_pred = 5
def explore_data(file_name):
    df = pd.read_csv(file_name)
    df = df.iloc[::-1] #reverse data to have the right chronological order

    n_days = len(df.axes[0])
    n_features = len(df.axes[1]) - 1 #exclude last feature
    
    #print df.describe()
    print "Number of days (samples) ", n_days
    print "Number of features ", n_features

    return df[max(len(df) - number_days_max, 0): len(df)] #test with a small data set first
    #return df
              
def preprocess_data(df):
    #no data translation needed because all input data are number
    #enhance the data with Rolling average and the "Adj Close" of previous day (shift(-1))
    #Shift all the feature columns "one day to the left" so we use the data of the "previous" day, to train the
    #adjusted close of the "current" day, which has more practical value.
    #one would think shift(-1) is shift to previous day, but since we already reverse the data set (for chronological order), we do shift(1)
    
    rolling_window_days = len(df) / 10 #make rolling window dependent on length of data set size
    df['Adj Close Prev'] = df['Adj Close'].shift(1) 

    df['Open'] = df['Open'].shift(1)
    df['High'] = df['High'].shift(1)
    df['Low'] = df['Low'].shift(1)
    df['Close'] = df['Close'].shift(1)
    df['Volume'] = df['Volume'].shift(1)
    
    rm_adjustedClose = pd.rolling_mean(df['Adj Close Prev'], window=rolling_window_days)
    
    df['Rolling'] = rm_adjustedClose  #rm_adjustedClose.fillna(df['Adj Close'])

    #having bollingen band seems to make regression worse (but may make prediction more safe): predict inside the band
    #df['Bol_upper'] = pd.rolling_mean(df['Adj Close'], window=rolling_window_days) + 2 * pd.rolling_std(df['Adj Close'], rolling_window_days, min_periods=rolling_window_days)
    #df['Bol_lower'] = pd.rolling_mean(df['Adj Close'], window=rolling_window_days) - 2 * pd.rolling_std(df['Adj Close'], rolling_window_days, min_periods=rolling_window_days)
    
    df = df [rolling_window_days:len(df) -1] #exclude first days without rolling mean from data
    #exclude last days as well

    X_all = df[feature_cols]
    y_all = df['Adj Close']

    return X_all, y_all

"""
    Split the data into train and test data set. Remember: order is important: we want to /test with "new" data, not "old" data. 
"""
def split_data(X_all, y_all):
    num_all = X_all.shape[0]
    num_train = num_all - number_days_pred # only need to test 5 days, stock data is time-sensitive, we don't have to predict long in advance
    num_test = num_all - num_train
    

    X_train = X_all[:num_train] #train earlier in time
    X_test = X_all[num_train:num_all] #test later in time
    
    y_train = y_all[:num_train] 
    y_test = y_all[num_train:num_all]
    

    print "Traing set: {} samples ".format(X_train.shape[0])
    print "Testing set: {} samples ".format(X_test.shape[0])
    
    return X_train, X_test, y_train, y_test


"""
    experiment with another training algorithm (tree and ada regressor) to compare the performance to the choosen
    regressor (GradientBoosting)
"""
def train_learning_model_decision_tree_ada_boost(df):
    #code taken from sklearn
    X_all, y_all = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all)

    tree_regressor = DecisionTreeRegressor(max_depth = 6)
    ada_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = 500, learning_rate = 0.01, random_state = 1)

    tree_regressor.fit(X_train, y_train)
    ada_regressor.fit(X_train, y_train)

    y_pred_tree = tree_regressor.predict(X_test)
    y_pred_ada = ada_regressor.predict(X_test)
    
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    mse_ada = mean_squared_error(y_test, y_pred_ada)

    mse_tree_train = mean_squared_error(y_train, tree_regressor.predict(X_train))
    mse_ada_train = mean_squared_error(y_train, ada_regressor.predict(X_train))
    
    print ("MSE tree: %.4f " %mse_tree)
    print ("MSE ada: %.4f " %mse_ada)

    print ("MSE tree train: %.4f " %mse_tree_train)
    print ("MSE ada train: %.4f " %mse_ada_train)

def plot_feature_importance(regressor, params, X_test, y_test):
    test_score = np.zeros((params['n_estimators'],), dtype = np.float64)

    for i, y_pred in enumerate(regressor.staged_predict(X_test)):
        test_score[i] = regressor.loss_(y_test, y_pred)

    plt.figure(figsize = (12, 6))
    plt.subplot(1, 2, 1)
    plt.title('MAE Prediction vs. Actual (USD) ')

    plt.plot(np.arange(params['n_estimators']) + 1, regressor.train_score_, 'b-', label = 'Training set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label = 'Test set deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Mean absolute error')

    #plot feature importance
    feature_importance = regressor.feature_importances_
    #normalize
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')

    feature_names = np.array(feature_cols)

    plt.yticks(pos, feature_names[sorted_idx])

    plt.xlabel('Relative importance')
    plt.title('Variable Importance')
    
    plt.show()

def optimize_learning_model(regressor, X_train, y_train, X_test, y_test, display_graph):
    start =  time.time()
    #plot_feature_importance(regressor, params, X_test, y_test)


    """params = {'n_estimators' : (50, 100, 200, 500, 1000),
              'learning_rate': (0.01, 0.02, 0.1),
              'max_depth' : (1, 2, 3, 4),
               'loss': ('ls', 'lad', 'huber', 'quantile')}"""

    params = {'n_estimators' : (400, 500),
              'learning_rate': (0.01, 0.02),
              'max_depth' : (5, 6),
               'loss': ('ls', 'lad'),
              'min_samples_split':(1,2)}
    #best param: estimator: 500, loss: ls, learning rate 0.02, max_depth = 6
    
    scorer = make_scorer(mean_absolute_error, greater_is_better = False)
    grid_search = GridSearchCV(regressor, params, scoring = scorer)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print best_params
    
    boost_optimized = ensemble.GradientBoostingRegressor(n_estimators = best_params['n_estimators'], 
                                   learning_rate = best_params['learning_rate'], 
                                   max_depth = best_params['max_depth'], 
                                   loss = best_params['loss'])

    boost_optimized.fit(X_train, y_train)
    end = time.time()
    print "Time needed for tuning "  ,(end-start)
    print "Optimized boosting results **************"    
    calculate_results(boost_optimized, X_train, X_test, y_train, y_test)
    print "End optimized boosting results *****"
    if display_graph:
        plot_feature_importance(boost_optimized, best_params, X_test, y_test)

"""
    Calculate the regressor performance
    Print the stats for a regressor using the trainign, testing data set
    and the defined performance metrics (MAE)
"""
def calculate_results(regressor, X_train, X_test, y_train, y_test):
    print "Stats for ",regressor.__class__.__name__
    
    y_predict = regressor.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_predict)
    mae_price_ratio = mae * 100/ np.mean(y_test)
    mae_train = mean_absolute_error (y_train, regressor.predict(X_train))
    mae_train_price_ratio = mae_train * 100 / np.mean(y_test)
    
    print ("MAE test set: %.4f" %mae)
    print ("MAE test set as percentage of price: %.4f" %mae_price_ratio)
    print ("MAE train set: %.4f" %mae_train)
    print ("MAE train set as as percentage of price: %.4f" %mae_train_price_ratio)
    print "-----------------------------"
    
def train_learning_model_svm(df):
    X_all, y_all = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all)

    regressor = SVR()
    regressor.fit(X_train, y_train)
    calculate_results(regressor, X_train, X_test, y_train, y_test)
    
"""
    Build a GradientBoostingRegressor model, train the model with the data set,
    test the model with the test say and print out the performance metrics
"""
def train_learning_model_gradient_boost(df, use_grid_search, display_graph):
    #code taken and adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#example-ensemble-plot-gradient-boosting-regression-py
    
    X_all, y_all = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all)
    
    #just use default parameter first
    regressor = ensemble.GradientBoostingRegressor()
    regressor.fit(X_train, y_train)


    calculate_results(regressor, X_train, X_test, y_train, y_test)
    if use_grid_search == True:
        optimize_learning_model(regressor, X_train, y_train, X_test, y_test, display_graph)
        
    return regressor
    
def visualize_data(df, field_names):
    df[field_names].plot()
    plt.show()

def explore_web_data(symbol, start_date, end_date):
    data = web.DataReader(symbol, data_source='google', start = start_date, end = end_date)

    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = pd.rolling_std(data['Log_Ret'], window = 252) * np.sqrt(252)

    data[['Close', 'Volatility']].plot(subplots = True, color = 'blue', figsize=(8, 6))
    
    plt.show()
    print data.tail()

stocks = ['Google','facebook', 'tesla']

for stock_name in stocks:
    print '********Experimenting with data set ', stock_name
    df = explore_data('data/'+stock_name+'.csv')
    #True for 3rd parameter for visualization, True for 2nd for Gridsearch
    train_learning_model_gradient_boost(df, True, False)
    train_learning_model_svm(df)
    print "*********************************"


