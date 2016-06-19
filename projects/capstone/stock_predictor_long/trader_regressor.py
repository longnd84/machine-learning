from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

import pandas.io.data as web
from math import *
import time

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

feature_cols = ['Open','High','Low','Close','Volume','Rolling']
feature_cols_bols = ['Open','High','Low','Close','Volume', 'Rolling', 'Bol_lower', 'Bol_upper']

rolling_window_days = 30

def explore_data(file_name):
    df = pd.read_csv(file_name)
    df = df.iloc[::-1] #reverse data to have the right chronological order

    n_days = len(df.axes[0])
    n_features = len(df.axes[1]) - 1 #exclude last feature
    
    print df.describe()
    print "Number of days (samples) ", n_days
    print "Number of features ", n_features
    
    return df

def preprocess_data(df):
    #no data translation needed because all input data are number

    #should we exclude "close" field from input prediction as well?
    #that can give an unfair advantage
    
    #rolling mean seems to take current value into account
    #shift() so that we exclude today from rolling mean to prevent some
    #unfair advantage (especialling when rolling window is small)
    #TODO: shift by 1 or -1
    rm_adjustedClose = pd.rolling_mean(df['Adj Close'].shift(1), window=rolling_window_days)

    df['Rolling'] = rm_adjustedClose  #rm_adjustedClose.fillna(df['Adj Close'])

    #having bollingen band seems to make regression worse (but may make prediction more safe): predict inside the band
    #df['Bol_upper'] = pd.rolling_mean(df['Adj Close'], window=rolling_window_days) + 2 * pd.rolling_std(df['Adj Close'], rolling_window_days, min_periods=rolling_window_days)
    #df['Bol_lower'] = pd.rolling_mean(df['Adj Close'], window=rolling_window_days) - 2 * pd.rolling_std(df['Adj Close'], rolling_window_days, min_periods=rolling_window_days)
    
    df = df [rolling_window_days:] #exclude first days without rolling mean from data
    X_all = df[feature_cols]
    y_all = df['Adj Close']

    print X_all.head()
    print "Features summary "
    print X_all.describe()
    print "---------------------------------------------------"

    print "Target summary"
    print y_all.describe()
    print "---------------------------------------------------"

    return X_all, y_all

def split_data(X_all, y_all):
    num_all = X_all.shape[0]
    num_train = int(0.7 * num_all)
    num_test = num_all - num_train
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size = num_train, test_size = num_test, random_state = 1)

    print "Traing set: {} samples ".format(X_train.shape[0])
    print "Testing set: {} samples ".format(X_test.shape[0])
    
    return X_train, X_test, y_train, y_test

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
    plt.title('Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, regressor.train_score_, 'b-', label = 'Training set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label = 'Test set deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    #plot feature importance
    feature_importance = regressor.feature_importances_
    #normalize
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    
    print "Feature importance " , feature_importance 
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')

    feature_names = np.array(feature_cols)
    print "Sorted idx " , sorted_idx , ' Type ', type(sorted_idx)
    plt.yticks(pos, feature_names[sorted_idx])

    plt.xlabel('Relative importance')
    plt.title('Variable Importnace')
    
    plt.show()

def optimize_learning_model(regressor, X_train, y_train, X_test, y_test):
    start =  time.time()
    #plot_feature_importance(regressor, params, X_test, y_test)


    """params = {'n_estimators' : (50, 100, 200, 500, 1000),
              'learning_rate': (0.01, 0.02, 0.1),
              'max_depth' : (1, 2, 3, 4),
               'loss': ('ls', 'lad', 'huber', 'quantile')}"""

    params = {'n_estimators' : (400, 500),
              'learning_rate': (0.01, 0.02),
              'max_depth' : (5, 6),
               'loss': ('ls', 'lad')}
    #best param: estimator: 500, loss: ls, learning rate 0.02, max_depth = 6
    
    scorer = make_scorer(mean_squared_error, greater_is_better = False)
    grid_search = GridSearchCV(regressor, params, scoring = scorer)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print best_params
    
    boost_optimized = ensemble.GradientBoostingRegressor(n_estimators = best_params['n_estimators'], 
                                   learning_rate = best_params['learning_rate'], 
                                   max_depth = best_params['max_depth'], 
                                   loss = best_params['loss'])

    boost_optimized.fit(X_train, y_train)
    
    mse = mean_squared_error(y_test, boost_optimized.predict(X_test))
    mse_train = mean_squared_error (y_train, boost_optimized.predict(X_train))

    print "Optimized boosting ******"
    print ("MSE testing (optimized) : %.4f" %mse)
    print ("MSE train (optimized): %.4f" %mse_train)

    end = time.time()
    print "Time needed for tuning "  ,(end-start)

    plot_feature_importance(boost_optimized, best_params, X_test, y_test)


def train_learning_model(df):
    #code taken from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#example-ensemble-plot-gradient-boosting-regression-py
    
    X_all, y_all = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all)
    
    params = {'n_estimators':500, 'max_depth': 6, 'min_samples_split':1, 'learning_rate':0.02, 'loss':'ls'}

    regressor = ensemble.GradientBoostingRegressor(**params)
    regressor.fit(X_train, y_train)

    
    y_predict = regressor.predict(X_test)
    
    mse = mean_squared_error(y_test, y_predict)
    mse_train = mean_squared_error (y_train, regressor.predict(X_train))

    print ("MSE: %.4f" %mse)
    print ("MSE train: %.4f" %mse_train)

    #comment out when needed, takes longer
    optimize_learning_model(regressor, X_train, y_train, X_test, y_test)
    
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

#Choose the data set
    
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
#df = explore_data("data/HP.csv")
df = explore_data("data/yahoo.csv")
#df = explore_data("data/facebook.csv")
#df = explore_data("data/tesla.csv")


#visualize(df)
train_learning_model(df)
#train_learning_model_decision_tree_ada_boost(df)
