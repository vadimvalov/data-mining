import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '../docs/results'

df_train = pd.read_csv(os.path.join(RESULTS_DIR, 'train_processed.csv'))
df_test = pd.read_csv(os.path.join(RESULTS_DIR, 'test_processed.csv'))

X = df_train.drop(['SalePrice', 'Id'], axis=1)
y = df_train['SalePrice']
X_test = df_test.drop(['Id'], axis=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfold))
    return rmse.mean()

lr = LinearRegression()
print("LinearRegression RMSE:", rmse_cv(lr, X, y))

ridge = Ridge(alpha=10)
print("Ridge RMSE:", rmse_cv(ridge, X, y))

lasso = Lasso(alpha=0.0005)
print("Lasso RMSE:", rmse_cv(lasso, X, y))

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
print("DecisionTree RMSE:", rmse_cv(dt, X, y))

rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
print("RandomForest RMSE:", rmse_cv(rf, X, y))

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
print("GradientBoosting RMSE:", rmse_cv(gbr, X, y))

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
print("XGBoost RMSE:", rmse_cv(xgb_model, X, y))

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
print("LightGBM RMSE:", rmse_cv(lgb_model, X, y))

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)
print("Best XGBoost params:", grid_search.best_params_)
print("Best XGBoost RMSE:", np.sqrt(-grid_search.best_score_))

best_xgb = grid_search.best_estimator_
best_xgb.fit(X, y)
xgb_pred = best_xgb.predict(X_test)

lasso.fit(X, y)
lasso_pred = lasso.predict(X_test)

lgb_model.fit(X, y)
lgb_pred = lgb_model.predict(X_test)

ensemble_pred = 0.5 * xgb_pred + 0.3 * lasso_pred + 0.2 * lgb_pred

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': np.expm1(ensemble_pred)
})
submission.to_csv(os.path.join(RESULTS_DIR, 'submission.csv'), index=False)

print("Submission saved")