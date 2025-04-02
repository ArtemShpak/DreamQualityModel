import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import columns

#Завантаження датасету
ds = pd.read_csv('../../data/student_sleep_patterns.csv')
print(ds)

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

#Збереження закодованих даних
ds.to_csv("../../data/encoded_data.csv", index=False)

#Завантаження закодованого датасету
ds = pd.read_csv('../../data/encoded_data.csv')

#Маштабування даних від 0 до 1
def scale_and_plot(dm, column_name):
    scaler = MinMaxScaler()
    dm[column_name] = scaler.fit_transform(dm[[column_name]])


for column in columns.columns_to_scale:
    scale_and_plot(ds, column)

#Збереження маштабованих даних
ds.to_csv("../../data/scaled_data.csv", index=False)

