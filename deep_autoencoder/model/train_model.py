from model.base_model import BaseModel
import tensorflow as tf
from tensorflow import contrib


class TrainModel(BaseModel):

    def __init__(self, FLAGS, name_scope):

        super(TrainModel,self).__init__(FLAGS)

        self._init_parameters()

    def loss(self, inputs, predictions):
        '''Compute the error on a forward pass, and return those predictions.'''
        mask = tf.where(inputs, 0.0), tf.zeros_like(inputs), inputs) # Indices of 0 values in the training set
        num_train_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float32) # Number of non-zero values
        bool_mask = tf.cast(mask, dtype=tf.bool)
        # Sets outputs to 0 where corresponding inputs are 0
        predictions = tf.where(bool_mask, predictions, tf.zeros_like(predictions))

        with tf.name_scope('loss'):
            loss = tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions, inputs))), num_labels)

        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss += self.FLAGS.lambda_ * l2_loss

        return loss

    def grad(self, inputs, predictions):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, predictions)
        return tape.gradient(loss, tf.trainable_variables())


    def train(self, inputs, labels):
        tfe = contrib.eager

        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        global_step = tf.Variable(0)

        for epoch in range(self.FLAGS.num_epochs):
            # Figure out a way to iterate over the dataset, use batches
            for batch in dataset:
                # First forward pass
                predictions = self.forward(batch)
                grads = grad(batch, predictions)
                # First backward pass
                optimizer.apply_gradients(zip(grads, tf.trainable_variables(), global_step))
                # Second forward pass
                pred2 = self.forward(predictions)
                grads2 = grad(predictions, pred2)
                # Second backward pass
                optimizer.apply_gradients(zip(grads2, tf.trainable_variables(), global_step))
                # End of epoch








        return train_op, RMSE_loss
