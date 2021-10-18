# -*- coding: utf-8 -*-
"""Day-5 Data Preprocessing Assigment.ipynb

Author: Logeswaran K

Original file is located at
    https://colab.research.google.com/drive/1TRZVTxVyNxsIMBYDlq256D5JM73jyvl3
"""

from google.colab import drive
drive.mount('/content/drive/')

import numpy as np
import pandas as pd

dataset=pd.read_csv('/content/drive/MyDrive/AI/Training/Notes/Dataset/bank.csv')

dataset

dataset.isnull().any()

dataset.isnull().sum()

dataset.info()

dataset.columns

x=dataset.iloc[:,4:16]
y=dataset.iloc[:,16:17]
y

x

#convert dataframe to array
x_ary=x.values

y_ary=y.values

x_ary

y_ary

x.shape

y.shape

dataset["default"].unique()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer([("one",OneHotEncoder(),[0])],remainder="passthrough")
x_ary_trans=ct.fit_transform(x_ary)
x_ary_trans

x_ary_trans.shape

dataset["housing"].unique()

ct=ColumnTransformer([("one",OneHotEncoder(),[0,2,3,4,6,11])],remainder="passthrough")
x_ary_trans=ct.fit_transform(x_ary)
x_ary_trans

x_ary_trans.shape

ct=ColumnTransformer([("one",OneHotEncoder(),[0])],remainder="passthrough")
y_ary_trans=ct.fit_transform(y_ary)
y_ary_trans

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_ary_trans,y_ary_trans,test_size=0.2,random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape
