#ignoring warnings
import warnings
warnings.simplefilter('ignore')

#importing neccesary modules
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import datetime
import yfinance as yf

import sklearn
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split

import xgboost
from xgboost import XGBRegressor, DMatrix


class forex:
    
    def __init__(self):
        self._scaler = 0
        
    def get_forex_data(self, currency1, currency2, start_date, end_date):
        # On ajoute un jour pour que le dernier jour soit inclus
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)

        cour = yf.download(currency1+currency2+'=X', start=start_date, end=end_date)
        cour.columns = map(str.lower, cour.columns)
        cour = cour.drop('volume', axis=1)
        cour = cour.drop('adj close', axis=1)
        return cour
    
    def plot_forex(self, currency1, currency2, start_date, end_date):
        self.get_forex_data(currency1, currency2, start_date, end_date).close.plot()

    def get_features(self, currency1, currency2, start_date, end_date):

        cour = self.get_forex_data(currency1, currency2, start_date, end_date)

        def generate_features(df):
            """ Generate features for a stock/index/currency/commodity based on historical price and performance
            Args:
                df (dataframe with columns "open", "close", "high", "low", "volume")
            Returns:
                dataframe, data set with new features
            """
            df_new = pd.DataFrame()

            # 6 original features
            # df_new['open'] = df['open']
            df_new['open_1'] = df['open'].shift(1)
            df_new['close_1'] = df['close'].shift(1)
            df_new['high_1'] = df['high'].shift(1)
            df_new['low_1'] = df['low'].shift(1)

            # 50 original features
            # average price
            df_new['avg_price_5'] = df['close'].rolling(window=5).mean().shift(1)
            df_new['avg_price_30'] = df['close'].rolling(window=21).mean().shift(1)
            df_new['avg_price_90'] = df['close'].rolling(window=63).mean().shift(1)
            df_new['avg_price_365'] = df['close'].rolling(window=252).mean().shift(1)

            # average price ratio
            df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
            df_new['ratio_avg_price_905_'] = df_new['avg_price_5'] / df_new['avg_price_90']
            df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
            df_new['ratio_avg_price_30_90'] = df_new['avg_price_30'] / df_new['avg_price_90']
            df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
            df_new['ratio_avg_price_90_365'] = df_new['avg_price_90'] / df_new['avg_price_365']                                            


            # standard deviation of prices
            df_new['std_price_5'] = df['close'].rolling(window=5).std().shift(1)
            df_new['std_price_30'] = df['close'].rolling(window=21).std().shift(1)
            df_new['std_price_90'] = df['close'].rolling(window=63).std().shift(1)                                               
            df_new['std_price_365'] = df['close'].rolling(window=252).std().shift(1)

            # standard deviation ratio of prices 
            df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
            df_new['ratio_std_price_5_90'] = df_new['std_price_5'] / df_new['std_price_90']
            df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
            df_new['ratio_std_price_30_90'] = df_new['std_price_30'] / df_new['std_price_90'] 
            df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']                                               
            df_new['ratio_std_price_90_365'] = df_new['std_price_90'] / df_new['std_price_365']                                                


            # return
            df_new['return_1'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)).shift(1)
            df_new['return_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)).shift(1)
            df_new['return_30'] = ((df['close'] - df['close'].shift(21)) / df['close'].shift(21)).shift(1)
            df_new['return_90'] = ((df['close'] - df['close'].shift(63)) / df['close'].shift(63)).shift(1)                                                
            df_new['return_365'] = ((df['close'] - df['close'].shift(252)) / df['close'].shift(252)).shift(1)

            #average of return
            df_new['moving_avg_5'] = df_new['return_1'].rolling(window=5).mean()
            df_new['moving_avg_30'] = df_new['return_1'].rolling(window=21).mean()
            df_new['moving_avg_90'] = df_new['return_1'].rolling(window=63).mean() # avant appelé moving_avg_30 mais bizarre car doublons
            df_new['moving_avg_365'] = df_new['return_1'].rolling(window=252).mean()

            # the target
            df_new['close'] = df['close']
            df_new = df_new.dropna(axis=0)
            return df_new


        return generate_features(cour)
    
    def scale_data(self, df):

        if self._scaler == 0:
            scaler = StandardScaler()
            scaler.fit(df)
            self._scaler = scaler
        else:
            scaler = self._scaler
            
        #Rescale both sets using the trained scaler
        return scaler.transform(df)
    
    
    def train_model(self, df):
        df = df[:-1]
        X_train = df.drop('close', axis='columns')
        y_train = df.close
        X_train_scaled = self.scale_data(X_train)
        
        lin = LinearRegression()
        lin.fit(X_train_scaled, y_train)

        bgr = BaggingRegressor(base_estimator=lin, n_estimators=100, oob_score=True, n_jobs=-1)
        bgr.fit(X_train_scaled, y_train)
        
        self._bgr = bgr
        return bgr

    def predict(self, df):
        model = self._bgr
        X_test = df.drop('close', axis='columns')
        X_test_scaled = self.scale_data(X_test);

        return model.predict(X_test_scaled)

