import tensorflow as tf

class BaseModel(object):

    def __init__(self, FLAGS):
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS

    def _init_parameters(self):
        with tf.name_scope('weights'):
            self.W_1 = tf.get_variable(name='weight_1', shape=(self.FLAGS.num_mov, 512),
                                    initializer=self.weight_initializer)
            self.W_2 = tf.get_variable(name='weight_2', shape=(512, 512),
                                    initializer=self.weight_initializer)
            self.W_3 = tf.get_variable(name='weight_3', shape=(512, 1024),
                                    initializer=self.weight_initializer)
            self.W_4 = tf.get_variable(name='weight_4', shape=(1024, 512),
                                    initializer=self.weight_initializer)
            self.W_5 = tf.get_variable(name='weight_5', shape=(512, 512),
                                    initializer=self.weight_initializer)
            self.W_6 = tf.get_variable(name='weight_6', shape=(512, self.FLAGS.num_mov),
                                    initializer=self.weight_initializer)

        with tf.name_scope('biases'):
            self.b1 = tf.get_variable(name='bias_1', shape=(512), initializer=self.bias_initializer)
            self.b2 = tf.get_variable(name='bias_2', shape=(512), initializer=self.bias_initializer)
            self.b3 = tf.get_variable(name='bias_3', shape=(1024), initializer=self.bias_initializer)
            self.b4 = tf.get_variable(name='bias_4', shape=(512), initializer=self.bias_initializer)
            self.b5 = tf.get_variable(name='bias_5', shape=(512), initializer=self.bias_initializer)


    def forward(self, x):
        '''Makes one forward pass and predicts network outputs.'''
        with tf.name_scope('inference'):
            a1 = tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1), self.b1))
            a2 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2), self.b2))
            a3 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3), self.b3))
            a3 = tf.nn.dropout(a3, rate=0.8)
            a4 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a3, self.W_4), self.b4))
            a5 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a4, self.W_5), self.b5))
            a6 = tf.matmul(a5, self.W_6)

        return a6
