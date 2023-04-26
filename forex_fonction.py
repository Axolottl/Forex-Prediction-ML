#ignoring warnings
import warnings
warnings.simplefilter('ignore')

#importing neccesary modules
import pickle
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import joblib
from sklearn.model_selection import GridSearchCV, train_test_split

import xgboost
from xgboost import XGBRegressor, DMatrix


class forex:
    
    def __init__(self):
        self._scaler = 0
        self._nb_jour_prev = 1
        self._split_date = None
    
    def set_pred_day(self, nb_jour):
        self._nb_jour_prev =  nb_jour
        
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

    def get_features(self, currency1, currency2, start_date, end_date, variation=True):
        self._variation = variation

        cour = self.get_forex_data(currency1, currency2, start_date, end_date)

        def generate_features(df):
            """ Generate features for a stock/index/currency/commodity based on historical price and performance
            Args:
                df (dataframe with columns "open", "close", "high", "low")
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
            nb_jour_prev = self._nb_jour_prev
            df_new['close'] = df['close'].shift(1 - nb_jour_prev) 
            variation_value = df['close'].shift(1 - nb_jour_prev) - df_new.close.shift(nb_jour_prev)
            if variation:
                df_new['close'] = variation_value
            df_new['class'] = variation_value > 0 
            df_new = df_new.dropna(axis=0)
            return df_new

        df_features = generate_features(cour)
        self._df_features = df_features
        return df_features
    
    def scale_data(self, df):

        if self._scaler == 0:
            scaler = StandardScaler()
            scaler.fit(df)
            self._scaler = scaler
        else:
            scaler = self._scaler
            
        #Rescale both sets using the trained scaler
        return scaler.transform(df)
    
    def split(self, date):
        self._split_date = datetime.datetime.strptime(date, '%Y-%m-%d')

    def train_model(self, _df_features=None, classification=False):
        df_f = self._df_features if _df_features is None else _df_features
        self._classification = classification

        split_date = self._split_date if self._split_date is not None else datetime.datetime.strptime('2099-12-31', '%Y-%m-%d')

        X_train = df_f[df_f.index < split_date].drop(['close','class'], axis='columns')
        X_test = df_f[df_f.index >= split_date].drop(['close','class'], axis='columns')
        if classification:
            y_train = df_f[df_f.index < split_date]['class']
            y_test = df_f[df_f.index >= split_date]['class']
        else:
            y_train = df_f[df_f.index < split_date].close
            y_test = df_f[df_f.index >= split_date].close

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        X_train_scaled = self.scale_data(X_train)

        if classification:
          from sklearn.tree import DecisionTreeClassifier
          clf = DecisionTreeClassifier(random_state=0)
          clf.fit(X_train_scaled, y_train)
          self._model = clf

        else:        
            lin = LinearRegression()
            lin.fit(X_train_scaled, y_train)

            bgr = BaggingRegressor(base_estimator=lin, n_estimators=200, oob_score=True, n_jobs=-1)
            bgr.fit(X_train_scaled, y_train)
        
            self._model = bgr
        # return bgr

    def predict(self, df_test=None, coef=1):
        X_test = self._X_test if df_test is None else df_test.drop(['close','class'], axis='columns')

        X_test_scaled = self.scale_data(X_test)
        model = self._model
        pred = model.predict(X_test_scaled)*coef

        self._pred = pred
        self._X_test = X_test
        self._X_scaled_test = X_test_scaled
        return pred
    
        
    def score(self, y_test=None, pred=None):
        y_test = self._y_test if y_test is None else y_test
        pred = self._pred if pred is None else pred

        if not self._classification:
            print('RMSE: {0:.5f}'.format(mean_squared_error(y_test, pred)**0.5))
            print('MAE: {0:.5f}'.format(mean_absolute_error(y_test, pred)))
            print('R^2: {0:.3f}'.format(r2_score(y_test, pred)))

        if self._variation:
            y_test_bool = y_test > 0
            pred_bool = pred > 0
            print('Accuracy: {0:.3f}'.format(accuracy_score(y_test_bool, pred_bool)))
            self.plot_confusion_matrix(y_test_bool, pred_bool)


    def simulate_forecasting(self, currency1, currency2, start_date, end_date):
        
        all_data = self.get_features(currency1, currency2, '2010-01-01', end_date)
        dates = self.get_forex_data(currency1, currency2, start_date, end_date).index
        results = pd.DataFrame(index=dates, columns=['prediction','truth'])
        
        first_train_data = all_data[all_data.index < dates[0]]
        self.train_model(first_train_data) # first big train

        for date in dates:
            # train_data = all_data[all_data.index < date].tail(1)
            # print('entrainement du jour ',train_data.index[0],' pour prédire le jour ', date)
            # self.train_model(train_data)
            predict = self.predict(all_data[all_data.index == date])
            results['prediction'][date] = predict[0]
            results['truth'][date] = all_data[all_data.index == date]['close'].values[0]
        self.show_evolution(results['truth'], results['prediction'])
        return results
        
    def show_evolution(self, y_test=None, pred=None, title=None, rolling=3, sorted=True):
        pred   = self._pred   if pred   is None else pred
        y_test = self._y_test if y_test is None else y_test

        evolution_df = self.get_evolution_df(y_test, pred, sorted)

        title = 'Price evolution : Prediction vs Truth' if title is None else title

        plt.figure(figsize = (18,9))
        plt.plot(evolution_df["evolution_test_y"].values, linewidth=3 if sorted else 0.5, label='Truth')
        if not sorted:
            plt.plot(range(len(evolution_df["evolution_test_y"])), evolution_df["evolution_test_y"].values, 'o', markersize=3, color='tab:blue')
        plt.scatter(range(len(evolution_df["evolution_prediction"])), evolution_df["evolution_prediction"].values, s=15, color='tab:orange', label='Prediction')

        for i in range(len(evolution_df)):
            plt.plot([i, i], [evolution_df["evolution_test_y"].values[i], evolution_df["evolution_prediction"].values[i]], color='tab:orange', linewidth=1)

        rolling_mean = evolution_df["evolution_prediction"].rolling(window=rolling).mean().values
        if sorted:
            plt.plot(rolling_mean, color='tab:green', label='Rolling Mean', linewidth=1)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.title(title)
        plt.legend()
        plt.show()
        
    def get_evolution_df(self, y_test, pred,sorted):
        variation = self._variation
        nb_jour_prev = self._nb_jour_prev

        evolution_df = pd.DataFrame()
        if variation:
            evolution_df["evolution_test_y"] = y_test
            evolution_df["evolution_prediction"] = pred
        else:
            evolution_df["evolution_test_y"] = (y_test - y_test.shift(nb_jour_prev)).values
            evolution_df["evolution_prediction"] = (pred - y_test.shift(nb_jour_prev)).values
        
        # evolution_df = evolution_df[evolution_df['evolution_prediction'] > -0.2]
        evolution_df = evolution_df.dropna()
        if sorted:
            evolution_df = evolution_df.sort_values('evolution_test_y', ascending=False)

        return evolution_df
        
    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        labels = ['True Negative','False Positive','False Negative','True Positive']
        categories = ['Zero', 'One']
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()