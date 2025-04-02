import warnings

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src import columns

warnings.simplefilter('ignore')

# Завантаження даних
ds = pd.read_csv('../../data/encoded_data.csv')

# Опис змінних
y = ds[columns.y_column]
X = ds[columns.x_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Список моделей для навчання
models = {
    'LinearRegression' : LinearRegression(),
    'Lasso' :Lasso(),
    'Ridge' :Ridge(),
    'XGBRegressor' :XGBRegressor(),
    'DecisionTreeRegressor' :DecisionTreeRegressor(),
    'SVR': SVR(),
    'KNeighborsRegressor':KNeighborsRegressor(),
    'RandomForestRegressor' : RandomForestRegressor()
    }

#Метод для виведення результатів моделей
regressors = dict()
for name, model in models.items():
    print('training ',name)
    regressor = model
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
    print('MAPE: ',metrics.mean_absolute_percentage_error(y_test, y_pred))
    print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
    print('R2', metrics.r2_score(y_test, y_pred))
    regressors[name] = regressor
    print('------------------------------------------------------------------------')



