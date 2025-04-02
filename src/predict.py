import pickle
import random

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


import hyperparameters
import columns

ds = pd.read_csv('../data/test_data.csv')
print('New data size', ds.shape)


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


X = ds[columns.x_columns]

# load the model and predict
rf = pickle.load(open('finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['Sleep_Quality'] = rf.predict(X)
ds.to_csv('../data/prediction_results.csv', index=False)