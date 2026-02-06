import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.special import boxcox1p

DATA_DIR = '../docs/dataset'
RESULTS_DIR = '../docs/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("Initial shape:", df_train.shape)

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Id')
if 'SalePrice' in numeric_cols:
    numeric_cols.remove('SalePrice')

Q1 = df_train[numeric_cols].quantile(0.25)
Q3 = df_train[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((df_train[numeric_cols] < (Q1 - 3 * IQR)) | (df_train[numeric_cols] > (Q3 + 3 * IQR))).any(axis=1)
df_train = df_train[outlier_mask]
print("After outlier removal:", df_train.shape)

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

for col in numeric_cols:
    if df_train[col].skew() > 1:
        df_train[col] = boxcox1p(df_train[col], 0.15)
        df_test[col] = boxcox1p(df_test[col], 0.15)

missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print("Missing values:\n", missing_train)

for col in df_train.columns:
    if df_train[col].dtype == 'object':
        df_train[col] = df_train[col].fillna('None')
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna('None')
    else:
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(median_val)

df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']

df_train['TotalBath'] = df_train['FullBath'] + 0.5 * df_train['HalfBath'] + df_train['BsmtFullBath'] + 0.5 * df_train['BsmtHalfBath']
df_test['TotalBath'] = df_test['FullBath'] + 0.5 * df_test['HalfBath'] + df_test['BsmtFullBath'] + 0.5 * df_test['BsmtHalfBath']

df_train['HouseAge'] = df_train['YrSold'] - df_train['YearBuilt']
df_test['HouseAge'] = df_test['YrSold'] - df_test['YearBuilt']

categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
if 'SalePrice' in numeric_features:
    numeric_features.remove('SalePrice')
if 'Id' in numeric_features:
    numeric_features.remove('Id')

scaler = StandardScaler()
df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
df_test[numeric_features] = scaler.transform(df_test[numeric_features])

from sklearn.feature_selection import SelectKBest, f_regression
X = df_train.drop(['SalePrice', 'Id'], axis=1)
y = df_train['SalePrice']
selector = SelectKBest(f_regression, k=50)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features")

df_train_final = df_train[selected_features + ['SalePrice', 'Id']]
df_test_final = df_test[selected_features + ['Id']]

df_train_final.to_csv(os.path.join(RESULTS_DIR, 'train_processed.csv'), index=False)
df_test_final.to_csv(os.path.join(RESULTS_DIR, 'test_processed.csv'), index=False)

print("Final shape:", df_train_final.shape, df_test_final.shape)