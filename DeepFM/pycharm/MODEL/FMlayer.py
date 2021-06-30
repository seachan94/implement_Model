import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from pycharm import config

from pycharm import dataprocessing
import pandas as pd

class FMlayer(tf.keras.layers.Layer):
    def __init__(self):

        super(FMlayer,self).__init__()

        self.embedding_size = config.EMBEDDING_SIZE
        self.num_feature = len(config.Modified_Field_Index)
        self.num_field  =len(config.FIELDS)
        self.field_index = config.Modified_Field_Index

        self.initializer = tf.keras.initializers.HeNormal()

        #initalization weight
        self.w = tf.Variable(
            tf.random.normal(
            shape = [self.num_feature],
            mean = 0,
            stddev= 1
        ),name = 'w')

        self.V = tf.Variable(
            tf.random.normal(
                shape = (self.num_field,config.EMBEDDING_SIZE),
                mean = 0,
                stddev= 1
            ),name ="v"
        )

    def call(self,inputs):

        x_batch = tf.reshape(inputs,[-1,self.num_feature,1])
        embeds = tf.nn.embedding_lookup(params = self.V, ids = self.field_index)

        #embedding layer 통과
        new_inputs = tf.math.multiply(x_batch,embeds)# 배치별 데이터의 feature들을 embedding 하여 표기한다 -> 각 feature마다 embedding값 되는 집합들의 모음


        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.w, inputs), axis=1, keepdims=False)
        # sigma(wx)
        #각 데이터 마다 가중치값 확보


        vxsquare = tf.square(tf.reduce_sum(new_inputs,axis = [1,2]))
        eachsquare = tf.reduce_sum(tf.square(new_inputs, axis = [1,2]))
        interactions = 0.5 *tf.subtract(vxsquare,eachsquare)




        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])

        y_fm = tf.concat([linear_terms, interactions], 1)

        return y_fm, new_inputs


