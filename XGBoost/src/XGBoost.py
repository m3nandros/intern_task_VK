import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

os.chdir('../../data_loader/src')
from data_loader import DataLoader

data_loader = DataLoader()
df = data_loader.load_data()

os.chdir('../../data_preprocessing/src')
from data_preprocessing import DataPreprocessor

data_prep = DataPreprocessor()
y, data, X_train, X_test, y_train, y_test, X_train_rus, X_test_rus, y_train_rus, y_test_rus = data_prep.preprocess_data(df)

class XGBoost_models():
    def __init__(self):
        pass

    def XGBoost_no_balance(self, X_train, y_train, X_test):
        X_train.sort_values(by='query_id', inplace=True)
        ranker_no_balance = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg",
                                          eval_metric='ndcg@5', lambdarank_pair_method="topk")
        ranker_no_balance.fit(X_train, y_train, qid=X_train['query_id'])
        predict_no_balance = ranker_no_balance.predict(X_test)
        visual_no_balance = plot_importance(ranker_no_balance,max_num_features=10, grid=False)
        return predict_no_balance, visual_no_balance

    def XGBoost_rus(self, X_train_rus, y_train_rus, X_test_rus):
        X_train_rus.sort_values(by='query_id', inplace=True)
        ranker_rus = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg",
                                   eval_metric='ndcg@5', lambdarank_pair_method="topk")
        ranker_rus.fit(X_train_rus, y_train_rus, qid=X_train_rus['query_id'])
        predict_rus = ranker_rus.predict(X_test_rus)
        visual_rus = plot_importance(ranker_rus,max_num_features=10, grid=False)
        return predict_rus, visual_rus
