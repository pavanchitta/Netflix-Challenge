import tensorflow as tf
import numpy as np
tfe = tf.contrib.eager

class BaseModel(object):

    def __init__(self, FLAGS):
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS
        self.session = tf.Session()

    def get_variables(self):
        # return self.model.trainable_variables
        return [self.W_1, self.W_2, self.W_3, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6]
        # return [self.W_1, self.b1, self.b2]

    def _init_parameters(self):
        tf.enable_resource_variables()

        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(512,
        #     activation=tf.nn.selu, input_shape=(self.FLAGS.num_mov,), name='w1'),
        #     tf.keras.layers.Dense(512, activation=tf.nn.selu, name='w2'),
        #     tf.keras.layers.Dense(1024, activation=tf.nn.selu),
        #     tf.keras.layers.Dropout(rate=0.8),
        #     tf.keras.layers.Dense(512, activation=tf.nn.selu),
        #     tf.keras.layers.Dense(512, activation=tf.nn.selu),
        #     tf.keras.layers.Dense(self.FLAGS.num_mov, activation=tf.nn.selu)
        # ])

        with tf.name_scope('weights'):
            # self.W_1 = tfe.Variable(tf.random_normal([self.FLAGS.num_mov, 512], mean=0.0, stddev=0.2), name='weight_1')
            # self.W_2 = tfe.Variable(tf.random_normal([512, 512], mean=0.0, stddev=0.2), name='weight_2')
            # self.W_3 = tfe.Variable(tf.random_normal([512, 1024], mean=0.0, stddev=0.2), name='weight_3')
            initializer = tf.contrib.layers.xavier_initializer()
            self.W_1 = tf.get_variable(shape=[self.FLAGS.num_mov, 512],
            initializer=initializer, name='weight_1')
            self.W_2 = tf.get_variable(shape=[512, 512], name='weight_2', initializer=initializer)
            self.W_3 = tf.get_variable(shape=[512, 1024], name='weight_3', initializer=initializer)

            # self.W_4 = tfe.Variable(tf.random_normal([1024, 512], mean=0.0, stddev=0.2), name='weight_4')
            # self.W_5 = tfe.Variable(tf.random_normal([512, 512], mean=0.0, stddev=0.2), name='weight_5')
            # self.W_6 = tfe.Variable(tf.random_normal([512, self.FLAGS.num_mov], mean=0.0, stddev=0.2), name='weight_6')

        with tf.name_scope('biases'):
            self.b1 = tf.get_variable(shape=[512], name='bias_1', initializer=self.bias_initializer)
            self.b2 = tf.get_variable(shape=[512], name='bias_2', initializer=self.bias_initializer)
            self.b3 = tf.get_variable(shape=[1024], name='bias_3', initializer=self.bias_initializer)
            self.b4 = tf.get_variable(shape=[512], name='bias_4', initializer=self.bias_initializer)
            self.b5 = tf.get_variable(shape=[512], name='bias_5', initializer=self.bias_initializer)
            self.b6 = tf.get_variable(shape=[17770], name='bias_6', initializer=self.bias_initializer)

    def pred_with_RMSE(self, test_set, pred_set):
        test_set = test_set.repeat(1)
        batch_size = 1280
        test_iterator = test_set.batch(batch_size).make_one_shot_iterator()

        batched_predictions = pred_set.batch(batch_size)
        pred_iterator = batched_predictions.make_one_shot_iterator()
        pred_batch = pred_iterator.get_next()
        dense_batch = tf.sparse.to_dense(pred_batch)


        curr_preds = self.forward_pred(dense_batch)

        row_batch = None
        curr_movies = None
        curr_ratings = None
        batch_count = 0

        total_predictions = []
        actual = []

        try:
            while True:
                predictions = []
                row_batch = test_iterator.get_next()

                test_preds = tf.gather_nd(curr_preds, row_batch.indices)

                total_predictions = tf.concat([total_predictions, test_preds], 0)
                actual = tf.concat([actual, row_batch.values], 0)

                batch_count += batch_size

                pred_batch = pred_iterator.get_next()
                curr_preds = self.forward_pred(tf.sparse.to_dense(pred_batch))

        except tf.errors.OutOfRangeError:
            pass

        total_error = tf.math.sqrt(tf.losses.mean_squared_error(total_predictions, actual))


        return total_predictions, total_error

    def pred_for_sub(self, test_set, pred_set, submit=True, filename="submission.csv"):
        test_set = test_set.repeat(1)
        batch_size = 1280
        test_iterator = test_set.batch(batch_size).make_one_shot_iterator()

        batched_predictions = pred_set.batch(batch_size)
        pred_iterator = batched_predictions.make_one_shot_iterator()
        pred_batch = pred_iterator.get_next()
        dense_batch = tf.sparse.to_dense(pred_batch)


        curr_preds = self.forward_pred(dense_batch)

        batch_count = 0

        total_predictions = []

        try:
            while True:
                curr_movies = test_iterator.get_next()[0]

                test_preds = tf.gather_nd(curr_preds, curr_movies)

                total_predictions = tf.concat([total_predictions, test_preds], 0)

                batch_count += batch_size

                pred_batch = pred_iterator.get_next()
                curr_preds = self.forward_pred(tf.sparse.to_dense(pred_batch))

        except tf.errors.OutOfRangeError:
            pass

        if submit:
            submission = total_predictions.numpy()
            np.savetxt(sub_filename, submission, delimiter=",")

        return total_predictions

    def forward_pred(self, x):
        '''Makes one forward pass and predicts network outputs.'''
        # return self.model(x)
        with tf.name_scope('inference'):
            a1 = tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1), self.b1))
            a2 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2), self.b2))
            a3 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3), self.b3))
            a4 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a3, tf.transpose(self.W_3)), self.b4))
            a5 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a4, tf.transpose(self.W_2)), self.b5))
            a6 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a5, tf.transpose(self.W_1)), self.b6))

        return a6

    def forward(self, x):
        '''Makes one forward pass and predicts network outputs.'''
        # return self.model(x)
        with tf.name_scope('inference'):
            a1 = tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1), self.b1))
            a2 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2), self.b2))
            a3 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3), self.b3))
            a3 = tf.nn.dropout(a3, rate=0.8)
            a4 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a3, tf.transpose(self.W_3)), self.b4))
            a5 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a4, tf.transpose(self.W_2)), self.b5))
            a6 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a5, tf.transpose(self.W_1)), self.b6))


        return a6
