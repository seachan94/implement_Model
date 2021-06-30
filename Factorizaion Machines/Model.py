

from lib import *
from config import *


class FMmodel(tf.keras.Model):
    def __init__(self,data_size):
        super(FMmodel, self).__init__()

        self.feature_num = len(Field) #변환 전 data feature
        self.modified_num = len(Modified_feature) # 변환 후 data feature
        self.data_nume = len(data_size)
        initalizer = tf.random_normal_initializer(mean = 0.0, stddev = 0.1, seed = 0)
        self.w = tf.Variable(
           initalizer(shape = self.modified_num, dtype = tf.float32)
        )
        self.v = tf.Variable(
            initalizer(shape = [self.modified_num,self.data_num],dtype = tf.float32)
        )


    def call(self,inputs):


        linear = tf.reduce_sum(
            tf.math.multiply(inputs,self.w)
            ,axis = 1
        )

        vx = tf.math.square(tf.math.matmul(inputs,self.v))
        vx2 = tf.math.matmul(tf.square(inputs),tf.square(self.v))

        interaction = 0.5 * tf.reduce_sum(vx-vx2,1)


        return tf.math.sigmoid(linear+interaction)
