import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from pycharm import config
class FMlayer(tf.keras.layers.Layer):
    def __init__(self):

        super(FMlayer,self).__init__()

        self.embedding_size = None
        self.num_feature = len(config.Modified_Categorical_FIELD)
        self.num_field  =len(config.FIELDS)
        self.field_index
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

