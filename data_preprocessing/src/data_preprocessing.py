import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

os.chdir('../../data_loader/src')
from data_loader import DataLoader

data_loader = DataLoader()
df = data_loader.load_data()

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_data(self, data):
        # Удаление ненужных кластеров
        data = self.delete_clusters(data)

        # Стандартизация и масштабирование
        data = self.standardization(data)

        # Удаление выбросов
        data = self.removing_emissions(data)

        # Выделение целевой переменной
        X, y = self.target_variable(data)

        # Балансировка классов
        X_resamp_rus, y_resamp_rus = self.balancing_classes(X, y)
        X_resamp_rus = pd.merge(X_resamp_rus, data['query_id'], left_index=True, right_index=True)

        X = pd.merge(X, data['query_id'], left_index=True, right_index=True)

        # Разделение на обучающую и тестовую выборки
        """
        В нашем случае будет три набора таких выборок:
        - без балансировки
        - с Oversampling (не работает корректно, см. data_preprocessing.ipynb)
        - с Undersampling
        """
        # Без балансировки
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # С Undersampling
        X_train_rus, X_test_rus, y_train_rus, y_test_rus = self.split_data(X_resamp_rus, y_resamp_rus)

        return y, data, X_train, X_test, y_train, y_test, X_train_rus, X_test_rus, y_train_rus, y_test_rus

    def delete_clusters(self, df):
        clusters_to_remove = ['feature_' + str(i) for i in range(110, 126)] + ['feature_' + str(i) for i in range(15, 21)]
        df_cleaned = df.drop(columns=clusters_to_remove)
        return df_cleaned

    def standardization(self, df_cleaned):
        features_to_standard = [col for col in df_cleaned.columns if col not in ['rank', 'query_id']]
        st_scaler = StandardScaler().fit(df_cleaned[features_to_standard])
        df_cleaned_scaled = st_scaler.transform(df_cleaned[features_to_standard])
        df_cleaned_scaled = pd.DataFrame(StandardScaler().fit_transform(df_cleaned[features_to_standard]), columns=df_cleaned[features_to_standard].columns)
        if 'rank' in df_cleaned_scaled.columns:
            df_cleaned_scaled.drop('rank', axis=1, inplace=True)
        if 'query_id' in df_cleaned_scaled.columns:
            df_cleaned_scaled.drop('query_id', axis=1, inplace=True)

        # Добавим столбцы rank и query_id в начало
        df_cleaned_scaled.insert(0, 'rank', df_cleaned['rank'])
        df_cleaned_scaled.insert(1, 'query_id', df_cleaned['query_id'])
        return df_cleaned_scaled

    def removing_emissions(self, df_cleaned_scaled):
        z_scores = stats.zscore(df_cleaned_scaled)
        threshold = 3
        outliers = df_cleaned_scaled[(np.abs(z_scores) > 3).any(axis=1)]  # Это наши выбросы
        outlier_indices = outliers.index
        df_cleaned_scaled_z = df_cleaned_scaled.drop(outlier_indices)
        return df_cleaned_scaled_z

    def target_variable(self, data):
        X = data.drop(['rank', 'query_id'], axis=1)
        y = data['rank']
        return X, y

    def balancing_classes(self, X, y):
        ros = RandomOverSampler(random_state=0)
        X_resamp_ros, y_resamp_ros = ros.fit_resample(X, y)

        rus = RandomUnderSampler(random_state=0)
        X_resamp_rus, y_resamp_rus = rus.fit_resample(X, y)
        return X_resamp_rus, y_resamp_rus

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test