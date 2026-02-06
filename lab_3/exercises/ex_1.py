import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paths
DATA_DIR = '../docs/dataset'
RESULTS_DIR = '../docs/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(df_train.shape, df_test.shape)
print(df_train.head())
print(df_train.describe())
print(df_train.info())
print(df_train.isnull().sum())
print(df_train['SalePrice'].describe())
print(df_train['SalePrice'].skew(), df_train['SalePrice'].kurtosis())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df_train['SalePrice'].hist(bins=50, ax=axes[0,0])
axes[0,0].set_title('SalePrice Distribution')
df_train['SalePrice'].plot(kind='box', ax=axes[0,1])
axes[0,1].set_title('SalePrice Boxplot')
stats.probplot(df_train['SalePrice'], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot')
np.log1p(df_train['SalePrice']).hist(bins=50, ax=axes[1,1])
axes[1,1].set_title('Log SalePrice Distribution')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'price_analysis.png'), dpi=100)
plt.close()

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
correlation = df_train[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'), dpi=100)
plt.close()

price_corr = correlation['SalePrice'].sort_values(ascending=False)
print(price_corr.head(10))

categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols[:5]:
    print(df_train[col].value_counts().head())

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(numeric_cols[:9]):
    if col != 'SalePrice':
        df_train[col].hist(bins=30, ax=axes[i])
        axes[i].set_title(col)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'numeric_distributions.png'), dpi=100)
plt.close()

for col in numeric_cols:
    if col != 'SalePrice':
        skew = df_train[col].skew()
        if abs(skew) > 1:
            print(f"{col}: {skew:.2f}")