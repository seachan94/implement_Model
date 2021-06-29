
import config
from dataprocessing import Dataprocessing

import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
warnings.filterwarnings(action='ignore')


col = ["a","b","c","d","e","f","g"]
da = [[1,2,3,"q","w","e",1],[4,5,6,"r","t","y",2],[7,8,9,"u","i","o",3]]
f = pd.DataFrame(da,columns=col)
x = f.drop("g",axis = 1)
y = f[["g"]]

cdata = Dataprocessing(x)
data = cdata.run()

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print(x_train.values)
print(tf.cast(y_train,tf.float32))

train_ = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_train.values,tf.float32),
    tf.cast(y_train,tf.float32))
).shuffle(30000).batch(config.BATCH_SIZE)
print(x_train.shape[0])

def get_data(data,y_col):
    x = data.drop(y_col,axis = 1)
    y = data[[y_col]]
    processing = Dataprocessing(x)
    data=  processing.run()

    x_train,x_test,y_train,y_test = train_test_split(data,y,test_size = 0.2, stratify= y)

    #change tensor input

    #shuffle 크기 데이터 집단을 섞음
    #Batch_size = 만큼 추출

    train = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(x_train,tf.float32), tf.cast(y_train,tf.float32)
        )
    ).shuffle(x_train.shape[0]).batch(config.BATCH_SIZE)

    test = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(x_test,tf.float32),tf.cast(y_test,tf.float32)
        )

    ).shuffle(10).batch(config.BATCH_SIZE)

    return train, test