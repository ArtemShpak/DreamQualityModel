import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor

from src import columns

# Завантаження даних
ds = pd.read_csv('../../data/encoded_data.csv')

# Опис змінних
y = ds[columns.y_column]
X = ds[columns.x_columns]

# Поділ датасету на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Ініціалізаця Lasso методу
lasso = Lasso()
lasso.fit(X_train, y_train)

# Передбачення моделлю
y_pred = lasso.predict(X_test)
print("Lasso Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print('----------------------------------------------------------------------')

# Поділ датасету на тренувальні та тестувальні для валідації
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)

# Підготовка kfold валідації
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Навчання kfold валідацією
for train, test in kfold.split(X_train):
    lasso = Lasso(tol=0.0001, max_iter=1000, alpha=3)
    lasso.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = lasso.predict(X_train.iloc[test])

    print("k-fold set metrics:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_train.iloc[test], y_pred)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_train.iloc[test], y_pred)}")
    print(f"R² Score: {r2_score(y_train.iloc[test], y_pred)}")

# Перевірка на тестовому датасеті
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_rem)

print("Test set metrics:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_rem, y_pred)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_rem, y_pred)}")
print(f"R² Score: {r2_score(y_rem, y_pred)}")
print('-----------------------------------------------------------------------------')

# Тренування моделі для виведення списку важливості колонок
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Отримуємо список
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. {X.columns[indices[f]]} ({importances[indices[f]]})")
