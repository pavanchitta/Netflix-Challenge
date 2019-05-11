import tensorflow as tf
tfe = tf.contrib.eager

class BaseModel(object):

    def __init__(self, FLAGS):
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS
        self.session = tf.Session()

    def get_variables(self):
        return self.model.trainable_variables
        #return [self.W_1, self.W_2, self.W_3, self.b1, self.b2, self.b3, self.b4, self.b5]

    def _init_parameters(self):
        tf.enable_resource_variables()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512,
            activation=tf.nn.selu, input_shape=(self.FLAGS.num_mov,), name='w1'),
            tf.keras.layers.Dense(512, activation=tf.nn.selu, name='w2'),
            tf.keras.layers.Dense(1024, activation=tf.nn.selu),
            tf.keras.layers.Dropout(rate=0.8),
            tf.keras.layers.Dense(512, activation=tf.nn.selu),
            tf.keras.layers.Dense(512, activation=tf.nn.selu),
            tf.keras.layers.Dense(self.FLAGS.num_mov, activation=tf.nn.selu)
        ])

        # with tf.name_scope('weights'):
        #     # self.W_1 = tfe.Variable(tf.random_normal([self.FLAGS.num_mov, 512], mean=0.0, stddev=0.2), name='weight_1')
        #     # self.W_2 = tfe.Variable(tf.random_normal([512, 512], mean=0.0, stddev=0.2), name='weight_2')
        #     # self.W_3 = tfe.Variable(tf.random_normal([512, 1024], mean=0.0, stddev=0.2), name='weight_3')
        #     self.W_1 = tf.get_variable(shape=[self.FLAGS.num_mov, 512], name='weight_1')
        #     self.W_2 = tf.get_variable(shape=[512, 512], name='weight_2')
        #     self.W_3 = tf.get_variable(shape=[512, 1024], name='weight_3')
        #
        #     self.W_4 = tfe.Variable(tf.random_normal([1024, 512], mean=0.0, stddev=0.2), name='weight_4')
        #     self.W_5 = tfe.Variable(tf.random_normal([512, 512], mean=0.0, stddev=0.2), name='weight_5')
        #     self.W_6 = tfe.Variable(tf.random_normal([512, self.FLAGS.num_mov], mean=0.0, stddev=0.2), name='weight_6')
        #
        # with tf.name_scope('biases'):
        #     self.b1 = tf.Variable(tf.zeros([512]), name='bias_1')
        #     self.b2 = tfe.Variable(tf.zeros([512]), name='bias_2')
        #     self.b3 = tfe.Variable(tf.zeros([1024]), name='bias_3')
        #     self.b4 = tfe.Variable(tf.zeros([512]), name='bias_4')
        #     self.b5 = tfe.Variable(tf.zeros([512]), name='bias_5')


    def forward(self, x):
        '''Makes one forward pass and predicts network outputs.'''
        return self.model(x)
        # with tf.name_scope('inference'):
        #     a1 = tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1), self.b1))
        #     a2 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2), self.b2))
        #     a3 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3), self.b3))
        #     a3 = tf.nn.dropout(a3, rate=0.8)
        #     a4 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a3, tf.transpose(self.W_3)), self.b4))
        #     a5 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a4, tf.transpose(self.W_2)), self.b5))
        #     a6 = tf.nn.selu(tf.matmul(a5, tf.transpose(self.W_1)))

        return a6
