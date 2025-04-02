import pickle

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src import columns, hyperparameters

#Завантаження датасету
ds = pd.read_csv('../data/train_data.csv')

#Видалення непотрібної колонки з ID
ds=ds.drop(columns=['Student_ID'])

#One Hot Encoding для об'єктів
def one_hot_encode_multiple(ds, column_names):
    ds_encoded = pd.get_dummies(ds, columns=column_names)

    for column in column_names:
        for col in ds_encoded.columns:
            if column in col:
                ds_encoded[col] = ds_encoded[col].astype(int)

    return ds_encoded

ds = one_hot_encode_multiple(ds, columns.columns_to_encode)

ds.info()

#Маштабування даних від 0 до 1
def scale_and_plot(dm, column_name):
    scaler = MinMaxScaler()
    dm[column_name] = scaler.fit_transform(dm[[column_name]])


for column in columns.x_columns:
    scale_and_plot(ds, column)

X = ds[columns.x_columns]
y = ds[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

ls = Lasso(**hyperparameters.params)
ls.fit(X_train, y_train)
y_pred = ls.predict(X_test)
print("Test set metrics:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

filename = 'finalized_model.sav'
pickle.dump(ls, open(filename, 'wb'))


