from model.base_model import BaseModel
import tensorflow as tf
from tensorflow import contrib

class ModelParams():
    def __init__(self, l2_reg, lambda_, num_movies):
        self.l2_reg = l2_reg
        self.lambda_ = lambda_
        self.num_mov = num_movies
        self.num_epochs = 2
        self.learning_rate = 0.005

class TrainModel(BaseModel):

    def __init__(self, FLAGS):

        super(TrainModel,self).__init__(FLAGS)

        self._init_parameters()

    def loss(self, inputs):
        '''Compute the error on a forward pass, and return those predictions.'''

        # mask = tf.not_equal(inputs, 0.0)
        # non_zero_array = tf.boolean_mask(inputs, mask)
        # print(non_zero_array)

        predictions = self.forward(inputs)
        # mask2 = tf.not_equal(predictions, 0.0)
        # non_zero_array2 = tf.boolean_mask(predictions, mask2)
        # print(non_zero_array2)
        num_train_labels = tf.count_nonzero(inputs, dtype=tf.float32)

        # Sets outputs to 0 where corresponding inputs are 0
        predictions = tf.where(tf.equal(inputs, 0.0), tf.zeros_like(predictions), predictions)
        #print(tf.count_nonzero(inputs))


        with tf.name_scope('loss'):
            loss = tf.math.divide(tf.reduce_sum(tf.square(tf.subtract(predictions, inputs))), num_train_labels)

        print(loss)

        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss += self.FLAGS.lambda_ * l2_loss

        return loss

    def grad(self, inputs):
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs)

        return tape.gradient(loss_val, self.get_variables())


    def train(self, dataset):
        optimizer = tf.train.MomentumOptimizer(self.FLAGS.learning_rate, 0.9)
        global_step = tf.Variable(0)
        batched_dataset = dataset.batch(128)
        #print(dataset[10])
        iterator = batched_dataset.make_one_shot_iterator()
        batch_count = 0
        for epoch in range(self.FLAGS.num_epochs):
            # Figure out a way to iterate over the dataset, use batches
            try:
                while True:
                    batch = iterator.get_next()
                    batch_count += 1
                    if (batch_count % 100 == 0):
                        print(batch_count)

                    dense_batch = tf.sparse.to_dense(batch, validate_indices=False)


                    # First forward pass
                    predictions = self.forward(dense_batch)
                    grads = self.grad(dense_batch)

                    # First backward pass
                    optimizer.apply_gradients(zip(grads, self.get_variables()))
                    # Second forward pass
                    grads2 = self.grad(predictions)
                    # Second backward pass
                    optimizer.apply_gradients(zip(grads2, self.get_variables()))
                    #self.model.save('model_1')
                    #break

                    # End of epoch
            except tf.errors.OutOfRangeError:
                pass
