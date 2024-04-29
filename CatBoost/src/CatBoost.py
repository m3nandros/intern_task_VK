import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import catboost

os.chdir('../../data_loader/src')
from data_loader import DataLoader

data_loader = DataLoader()
df = data_loader.load_data()

os.chdir('../../data_preprocessing/src')
from data_preprocessing import DataPreprocessor

data_prep = DataPreprocessor()
y, data, X_train, X_test, y_train, y_test, X_train_rus, X_test_rus, y_train_rus, y_test_rus = data_prep.preprocess_data(df)

class CatBoost_models:
    def __init__(self):
        pass

    def CatBoost_no_balance(self, X_train, y_train, X_test):
        CB_no_balance = catboost.CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE')
        CB_no_balance.fit(X_train, y_train)
        y_pred_no_balance = CB_no_balance.predict(X_test)
        return y_pred_no_balance

    def CatBoost_rus(self, X_train_rus, y_train_rus, X_test_rus):
        CB_rus = catboost.CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE')
        CB_rus.fit(X_train_rus, y_train_rus)
        y_pred_rus = CB_rus.predict(X_test_rus)
        return y_pred_rus