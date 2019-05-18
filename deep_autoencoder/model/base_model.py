import tensorflow as tf
tfe = tf.contrib.eager

class BaseModel(object):

    def __init__(self, FLAGS):
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS
        self.session = tf.Session()

    def get_variables(self):
        #return self.model.trainable_variables
        return [self.W_1, self.W_2, self.W_3, self.b1, self.b2, self.b3, self.b4, self.b5]

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
            self.W_1 = tf.get_variable(shape=[self.FLAGS.num_mov, 512], name='weight_1')
            self.W_2 = tf.get_variable(shape=[512, 512], name='weight_2')
            self.W_3 = tf.get_variable(shape=[512, 1024], name='weight_3')

            # self.W_4 = tfe.Variable(tf.random_normal([1024, 512], mean=0.0, stddev=0.2), name='weight_4')
            # self.W_5 = tfe.Variable(tf.random_normal([512, 512], mean=0.0, stddev=0.2), name='weight_5')
            # self.W_6 = tfe.Variable(tf.random_normal([512, self.FLAGS.num_mov], mean=0.0, stddev=0.2), name='weight_6')

        with tf.name_scope('biases'):
            self.b1 = tf.Variable(tf.zeros([512]), name='bias_1')
            self.b2 = tfe.Variable(tf.zeros([512]), name='bias_2')
            self.b3 = tfe.Variable(tf.zeros([1024]), name='bias_3')
            self.b4 = tfe.Variable(tf.zeros([512]), name='bias_4')
            self.b5 = tfe.Variable(tf.zeros([512]), name='bias_5')

    def better_pred(self, test_set, pred_set):
        test_set = test_set.repeat(1)
        batch_size = 1280
        test_iterator = test_set.make_one_shot_iterator()

        batched_predictions = pred_set.batch(batch_size)
        pred_iterator = batched_predictions.make_one_shot_iterator()
        pred_batch = pred_iterator.get_next()
        dense_batch = tf.sparse.to_dense(pred_batch)


        curr_preds = self.forward(dense_batch)

        row_batch = test_iterator.get_next()
        curr_movies = row_batch[0]
        batch_count = 0





        total_predictions = []
        try:
            while True:
                predictions = []
                for i in range(batch_size):
                    predictions = tf.concat([tf.reshape(tf.gather(curr_preds[i], curr_movies), [-1]), predictions], 0)
                    row_batch = test_iterator.get_next()
                    curr_movies = row_batch[0]

                pred_batch = pred_iterator.get_next()
                curr_preds = self.forward(tf.sparse.to_dense(pred_batch))

                batch_count += batch_size
                print(batch_count)



                total_predictions = tf.concat([total_predictions, predictions], 0)

        except tf.errors.OutOfRangeError:
            pass

        print(total_predictions)

        return total_predictions

    def pred_with_RMSE(self, test_set, pred_set):
        test_set = test_set.repeat(1)
        batch_size = 1280
        test_iterator = test_set.make_one_shot_iterator()

        batched_predictions = pred_set.batch(batch_size)
        pred_iterator = batched_predictions.make_one_shot_iterator()
        pred_batch = pred_iterator.get_next()
        dense_batch = tf.sparse.to_dense(pred_batch)


        curr_preds = self.forward(dense_batch)

        row_batch = test_iterator.get_next()
        curr_movies = row_batch[0]
        curr_ratings = row_batch[1]
        batch_count = 0

        total_predictions = []
        RMSE = 0
        try:
            while True:
                predictions = []
                for i in range(batch_size):
                    test_preds = tf.reshape(tf.gather(curr_preds[i], curr_movies), [-1])

                    error = tf.reduce_sum(tf.square(tf.subtract(test_preds, curr_ratings)))

                    RMSE = tf.add(RMSE, error)

                    predictions = tf.concat([tf.reshape(tf.gather(curr_preds[i], curr_movies), [-1]), predictions], 0)
                    row_batch = test_iterator.get_next()
                    curr_movies = row_batch[0]
                    curr_ratings = row_batch[1]

                pred_batch = pred_iterator.get_next()
                curr_preds = self.forward(tf.sparse.to_dense(pred_batch))

                batch_count += batch_size
                print(batch_count)

                total_predictions = tf.concat([total_predictions, predictions], 0)

        except tf.errors.OutOfRangeError:
            pass

        total_error = tf.math.sqrt(tf.math.divide(RMSE, tf.to_float(tf.size(total_predictions))))

        return total_predictions, total_error





    def predict(self, test_set, pred_set):
        test_set = test_set.repeat(1)
        batch_size = 1000
        batch_test = test_set.batch(2000)
        test_iterator = batch_test.make_one_shot_iterator()

        batched_predictions = pred_set.batch(batch_size)
        pred_iterator = batched_predictions.make_one_shot_iterator()
        pred_batch = pred_iterator.get_next()

        curr_preds = self.forward(pred_batch[1])
        row_batch = test_iterator.get_next()
        j = 0
        user, movie = row_batch[0], row_batch[1]

        total_predictions = []
        count = 0
        try:
            while True:
                for i in range(batch_size):
                    count += 1

                    if count % 2000 == 0:

                        print(count)

                    def body(predictions, j, user, movie):
                        predictions = tf.concat([[curr_preds[i][movie[j] - 1]], predictions], 0)
                        j += 1
                        if j == 2000:
                            j = 0
                            row_batch = test_iterator.get_next()
                            user, movie = row_batch[0], row_batch[1]

                        return predictions, j, user, movie

                    def cond(predictions, j, user, movie):

                        return tf.equal(user[j], pred_batch[0][i])

                    predictions, j, user, movie = tf.while_loop(cond, body, [[], j, user, movie])
                    total_predictions = tf.concat([total_predictions, predictions], 0)

                pred_batch = pred_iterator.get_next()
                curr_preds = self.forward(pred_batch[1])

        except tf.errors.OutOfRangeError:
            pass

        return total_predictions

    def forward(self, x):
        '''Makes one forward pass and predicts network outputs.'''
        #return self.model(x)
        with tf.name_scope('inference'):
            a1 = tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1), self.b1))
            a2 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2), self.b2))
            a3 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3), self.b3))
            a3 = tf.nn.dropout(a3, rate=0.8)
            a4 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a3, tf.transpose(self.W_3)), self.b4))
            a5 = tf.nn.selu(tf.nn.bias_add(tf.matmul(a4, tf.transpose(self.W_2)), self.b5))
            a6 = tf.nn.selu(tf.matmul(a5, tf.transpose(self.W_1)))

        return a6