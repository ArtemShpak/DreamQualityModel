import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.simplefilter('ignore')

from src import columns

# Завантаження даних
ds = pd.read_csv('../../data/encoded_data.csv')

# Опис змінних
y = ds[columns.y_column]
X = ds[columns.x_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Створення моделі Lasso
lasso = Lasso()

# Визначимо сітку гіперпараметрів для пошуку
param_grid = {
    'alpha': np.logspace(-4, 0, 50),      # Значення регуляризації
    'max_iter': [1000, 5000, 10000],      # Кількість ітерацій
    'tol': [1e-4, 1e-3, 1e-2]             # Точність
}

# Налаштування пошуку по сітці з крос-валідацією
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Виконання пошуку на тренувальних даних
grid_search.fit(X_train, y_train)

# Виведення найкращхи параметрів та відповідну похибку
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated score: {-grid_search.best_score_:.2f}")

