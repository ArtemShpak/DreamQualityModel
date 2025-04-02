import pandas as pd
from sklearn.model_selection import train_test_split
from src import columns

ds = pd.read_csv('../../data/student_sleep_patterns.csv')
#Поділ даних на тренувальні та тестові дані 80/20
X_train, X_test = train_test_split(ds, train_size=0.8)
X_train.to_csv('../../data/train_data.csv', index=False)

X_test[columns.columns_to_split].to_csv('../../data/test_data.csv', index=False)