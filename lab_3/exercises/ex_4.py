import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '../docs/results'

df_train = pd.read_csv(os.path.join(RESULTS_DIR, 'train_processed.csv'))
df_test = pd.read_csv(os.path.join(RESULTS_DIR, 'test_processed.csv'))

X = df_train.drop(['SalePrice', 'Id'], axis=1)
y = df_train['SalePrice']
X_test = df_test.drop(['Id'], axis=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'ridge': Ridge(alpha=10),
    'lasso': Lasso(alpha=0.0005),
    'elastic': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'gbr': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'xgb': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'lgb': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
    'cat': CatBoostRegressor(iterations=200, learning_rate=0.05, depth=5, random_state=42, verbose=0)
}

predictions = {}
for name, model in models.items():
    model.fit(X, y)
    predictions[name] = model.predict(X_test)
    print(f"{name} trained")

weights_arithmetic = {
    'ridge': 0.1,
    'lasso': 0.15,
    'elastic': 0.1,
    'rf': 0.1,
    'gbr': 0.15,
    'xgb': 0.2,
    'lgb': 0.15,
    'cat': 0.05
}

ensemble_arithmetic = sum(predictions[name] * weight for name, weight in weights_arithmetic.items())

ensemble_geometric = np.exp(sum(np.log(predictions[name]) * weight for name, weight in weights_arithmetic.items()))

ranks = {name: pd.Series(pred).rank() for name, pred in predictions.items()}
ensemble_rank = sum(ranks[name] * weight for name, weight in weights_arithmetic.items())
ensemble_rank = pd.Series(ensemble_rank).rank().values

train_meta = np.column_stack([
    models['ridge'].predict(X),
    models['lasso'].predict(X),
    models['xgb'].predict(X),
    models['lgb'].predict(X),
    models['cat'].predict(X)
])

test_meta = np.column_stack([
    predictions['ridge'],
    predictions['lasso'],
    predictions['xgb'],
    predictions['lgb'],
    predictions['cat']
])

meta_model = Ridge(alpha=1)
meta_model.fit(train_meta, y)
ensemble_stacking = meta_model.predict(test_meta)

blending_train = np.zeros((X.shape[0], len(models)))
blending_test = np.zeros((X_test.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    for train_idx, val_idx in kfold.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        blending_train[val_idx, i] = model.predict(X_val_fold)
    
    model.fit(X, y)
    blending_test[:, i] = model.predict(X_test)

blending_model = Ridge(alpha=1)
blending_model.fit(blending_train, y)
ensemble_blending = blending_model.predict(blending_test)

final_ensemble = 0.4 * ensemble_arithmetic + 0.3 * ensemble_stacking + 0.3 * ensemble_blending

submission_arithmetic = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': np.expm1(ensemble_arithmetic)})
submission_arithmetic.to_csv(os.path.join(RESULTS_DIR, 'submission_arithmetic.csv'), index=False)

submission_stacking = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': np.expm1(ensemble_stacking)})
submission_stacking.to_csv(os.path.join(RESULTS_DIR, 'submission_stacking.csv'), index=False)

submission_blending = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': np.expm1(ensemble_blending)})
submission_blending.to_csv(os.path.join(RESULTS_DIR, 'submission_blending.csv'), index=False)

submission_final = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': np.expm1(final_ensemble)})
submission_final.to_csv(os.path.join(RESULTS_DIR, 'submission_final.csv'), index=False)

print("All ensembles saved")