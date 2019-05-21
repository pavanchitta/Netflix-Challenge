from model.base_model import BaseModel
import tensorflow as tf
from tensorflow import contrib

class ModelParams():
    def __init__(self, l2_reg, lambda_, num_movies):
        self.l2_reg = l2_reg
        self.lambda_ = lambda_
        self.num_mov = num_movies
        self.num_epochs = 50
        self.learning_rate = 0.001

class TrainModel(BaseModel):

    def __init__(self, FLAGS):

        super(TrainModel,self).__init__(FLAGS)

        self._init_parameters()

    def loss(self, inputs, target):
        '''Compute the error on a forward pass, and return those predictions.'''

        # mask = tf.not_equal(inputs, 0.0)
        # non_zero_array = tf.boolean_mask(inputs, mask)
        # print(non_zero_array)
        with tf.name_scope('loss'):

            predictions = self.forward(inputs)
        # mask2 = tf.not_equal(predictions, 0.0)
        # non_zero_array2 = tf.boolean_mask(predictions, mask2)
        # print(non_zero_array2)
            num_train_labels = tf.count_nonzero(inputs, dtype=tf.float32)

        # Sets outputs to 0 where corresponding inputs are 0
            predictions = tf.where(tf.equal(inputs, 0.0), tf.zeros_like(predictions), predictions)
        #print(tf.count_nonzero(inputs))
            loss = tf.scalar_mul((1.0/num_train_labels), tf.reduce_sum(tf.square(tf.subtract(predictions, target)), axis=0))
            MSE = tf.reduce_sum(loss)

        return loss, MSE

    # def compute_loss(self, predictions, labels, num_labels):
    #     with tf.name_scope('loss'):
    #         loss_op = tf.math.divide(tf.reduce_sum(tf.square(tf.subtract(predictions, labels))), num_labels)

    def grad(self, inputs, target):
        with tf.GradientTape() as tape:
            loss_val, MSE = self.loss(inputs, target)

        gradient = tape.gradient(loss_val, self.get_variables())


        #print(tf.shape(target)[0])
        #print(gdiv)

        return loss_val, gradient, MSE

    # def optimizer(self, inputs):
    #     predictions = self.forward(inputs)
    #     num_train_labels = tf.count_nonzero(inputs, dtype=float32)
    #     predictions = tf.where(tf.equal(inputs, 0.0), tf.zeros_like(predicitions), predictions)
    #     loss = self.compute_loss(predictions, inputs, num_train_labels)
    #
    #     train_op = tf.train.MomentumOptimizer(self.FLAGS.learning_rate, 0.9).minimize(loss)
    #     return train_op, loss


    def train(self, dataset, probe_set, train_for_preds):
        print(self.FLAGS.learning_rate)
        optimizer = tf.train.MomentumOptimizer(self.FLAGS.learning_rate, 0.9)
        global_step = tf.Variable(0)
        batched_dataset = dataset.batch(128)
        iterator = batched_dataset.make_one_shot_iterator()
        total_movies = 460000


        for epoch in range(self.FLAGS.num_epochs):
            tf.keras.backend.set_learning_phase(1)
            batch_count = 0
            print()
            print("Epoch " + str(epoch + 1) + "/" + str(self.FLAGS.num_epochs))
            prog_bar = tf.keras.utils.Progbar(460000, stateful_metrics=["val_rmse"])

            try:
                while True:
                    batch = iterator.get_next()
                    batch_count += 1
                    # if batch_count % 100 == 0:
                    #     print(self.W_1)
                    #     print(self.b1)
                    #     print(self.b6)

                    dense_batch = tf.sparse.to_dense(batch)

                    # First forward pass
                    # predictions = self.forward(dense_batch)
                    loss, grads, MSE = self.grad(dense_batch, dense_batch)


                    prog_bar.update(128 * batch_count, values=[("train_loss", MSE)])

                    # First backward pass
                    optimizer.apply_gradients(zip(grads, self.get_variables()), global_step)
                    # Second forward pass
                    # _, grads2 = self.grad(predictions, predictions)
                    # Second backward pass
                    # optimizer.apply_gradients(zip(grads2, self.get_variables()), global_step)

                    # End of epoch

            except tf.errors.OutOfRangeError:
                ds = dataset.shuffle(460000)
                batched_dataset = ds.batch(128)
                iterator = batched_dataset.make_one_shot_iterator()


            tf.keras.backend.set_learning_phase(0)
            predictions, RMSE = self.pred_with_RMSE(probe_set, train_for_preds)
            prog_bar.update(128 * batch_count, values=[("val_rmse", RMSE)])

        saver = tf.contrib.eager.Saver(self.get_variables())
        saver.save("modelmodel")
