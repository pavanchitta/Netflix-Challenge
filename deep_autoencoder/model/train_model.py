from model.base_model import BaseModel
import tensorflow as tf
from tensorflow import contrib

class ModelParams():
    def __init__(self, l2_reg, lambda_, num_movies):
        self.l2_reg = l2_reg
        self.lambda_ = lambda_
        self.num_mov = num_movies
        self.num_epochs = 2
        self.learning_rate = 0.1

class TrainModel(BaseModel):

    def __init__(self, FLAGS):

        super(TrainModel,self).__init__(FLAGS)

        self._init_parameters()

    def loss(self, inputs):
        '''Compute the error on a forward pass, and return those predictions.'''

        predictions = self.forward(inputs)

        mask = tf.where(tf.not_equal(inputs, 0.0)) # Indices of 0 values in the training set
        num_train_labels = tf.cast(tf.div(tf.size(mask), 2), dtype=tf.float32) # Number of non-zero values

        # Sets outputs to 0 where corresponding inputs are 0
        predictions = tf.where(tf.equal(inputs, 0.0), predictions, tf.zeros_like(predictions))

        with tf.name_scope('loss'):
            loss = tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions, inputs))), num_train_labels)

        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss += self.FLAGS.lambda_ * l2_loss

        print(predictions)
        return loss

    def grad(self, inputs, predictions):
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs)

        return tape.gradient(loss_val, self.get_variables())


    def train(self, dataset):
        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        global_step = tf.Variable(0)
        batched_dataset = dataset.batch(1)
        iterator = batched_dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        for epoch in range(self.FLAGS.num_epochs):
            # Figure out a way to iterate over the dataset, use batches
            try:
                while True:
                    dense_batch = tf.sparse.to_dense(batch)
                    # First forward pass
                    predictions = self.forward(dense_batch)
                    grads = self.grad(dense_batch, predictions)

                    # First backward pass
                    optimizer.apply_gradients(zip(grads, self.get_variables()))
                    # Second forward pass
                    pred2 = self.forward(predictions)
                    grads2 = self.grad(predictions, pred2)
                    # Second backward pass
                    optimizer.apply_gradients(zip(grads2, self.get_variables()))
                    # End of epoch
            except tf.errors.OutOfRangeError:
                pass
