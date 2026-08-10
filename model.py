"""
Model development and evaluation for Sydney Housing Price Prediction.
Three models: Linear Regression (Ridge), Random Forest, Gradient Boosting
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import joblib

df = pd.read_csv('data/cleaned_suburbs.csv')

FEATURES_NUM = ['bedrooms', 'bathrooms', 'car_spaces', 'land_size_m2',
                'building_size_m2', 'property_age', 'Storeys', 'Pool', 'renovated',
                'desc_word_count', 'luxury_word_count']
FEATURES_CAT = ['suburb', 'property_type']
TARGET = 'sale_price'

data = df.dropna(subset=FEATURES_NUM + FEATURES_CAT + [TARGET]).reset_index(drop=True)
print("Modelling dataset shape:", data.shape)

X = data[FEATURES_NUM + FEATURES_CAT]
y = data[TARGET]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), FEATURES_NUM),
    ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT)
])

models = {
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    base_pipe = Pipeline([('prep', preprocessor), ('model', model)])
    # log1p target transform: sale prices are strongly right-skewed (Mosman outliers
    # up to $23M), so we model log(price) and inverse-transform predictions back.
    pipe = TransformedTargetRegressor(regressor=base_pipe, func=np.log1p, inverse_func=np.expm1)
    cv = cross_validate(pipe, X, y, cv=kf,
                         scoring={'mae': 'neg_mean_absolute_error',
                                  'rmse': 'neg_root_mean_squared_error',
                                  'r2': 'r2'},
                         return_train_score=True)
    results[name] = {
        'test_MAE': -cv['test_mae'].mean(),
        'test_RMSE': -cv['test_rmse'].mean(),
        'test_R2': cv['test_r2'].mean(),
        'train_R2': cv['train_r2'].mean(),
    }

res_df = pd.DataFrame(results).T
print(res_df)
res_df.to_csv('data/cv_results.csv')

# Train final model (best by test_R2) on train/test split for error analysis
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=0.2, random_state=42)

best_name = res_df['test_R2'].idxmax()
print("Best model:", best_name)
best_pipe = TransformedTargetRegressor(
    regressor=Pipeline([('prep', preprocessor), ('model', models[best_name])]),
    func=np.log1p, inverse_func=np.expm1)
best_pipe.fit(X_train, y_train)
preds = best_pipe.predict(X_test)

err_df = data.loc[idx_test, ['property_id', 'suburb', 'address', 'sale_price', 'property_type',
                              'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2']].copy()
err_df['predicted_price'] = preds
err_df['abs_error'] = (err_df['sale_price'] - err_df['predicted_price']).abs()
err_df['pct_error'] = err_df['abs_error'] / err_df['sale_price'] * 100
err_df = err_df.sort_values('abs_error', ascending=False)
err_df.to_csv('data/test_predictions.csv', index=False)
print(err_df.head(5))

# Fit best model on FULL data for deployment
final_pipe = TransformedTargetRegressor(
    regressor=Pipeline([('prep', preprocessor), ('model', models[best_name])]),
    func=np.log1p, inverse_func=np.expm1)
final_pipe.fit(X, y)
joblib.dump(final_pipe, 'app/model.joblib')
joblib.dump({'features_num': FEATURES_NUM, 'features_cat': FEATURES_CAT,
             'suburbs': sorted(data['suburb'].unique().tolist()),
             'property_types': sorted(data['property_type'].unique().tolist()),
             'best_model_name': best_name},
            'app/model_meta.joblib')
print("Saved model to app/model.joblib")
