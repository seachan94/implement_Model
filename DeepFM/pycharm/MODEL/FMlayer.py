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

        # <w,x>
        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.w, inputs), axis=1, keepdims=False)
        print("linear_terms", linear_terms, "\n")
        print("linear_termsa", tf.square(tf.reduce_sum(new_inputs, [1, 2])), "\n")
        print("linear_terms b", tf.reduce_sum(tf.square(new_inputs), [1, 2]), "\n")


        # (batch_size, )
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(new_inputs, [1, 2])),
            tf.reduce_sum(tf.square(new_inputs), [1, 2])
        )

        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])

        y_fm = tf.concat([linear_terms, interactions], 1)

        return y_fm, new_inputs



config.CONTINUE_FIELD = ["a","b","c"]
config.Categorical_FIELD = ["d","e","f"]
config.FIELDS = config.CONTINUE_FIELD +config.Categorical_FIELD

col = ["a","b","c","d","e","f"]

da = [[1,2,3,"q","w","e"],[4,5,6,"r","t","y"],[7,8,9,"u","i","o"]]

f = pd.DataFrame(da,columns=col)


cls = dataprocessing.Dataprocessing(f,False)

a = cls.run()

b = FMlayer.call()

print("sechan ", inputs)
print("secha y", y_fm)
print("seacacha ", new_inputs)

