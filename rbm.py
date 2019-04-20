import tensorflow as tf

class RBM:
    def __init__(self, visible, hidden):
        self._model = tf.Graph()

        # Initialize the weights randomly
        self._weights = tf.Variable(tf.truncated_normal([visible, hidden]), name="weights")
        self._hidden_bias = tf.Variable(tf.constant(0.1, shape=[hidden]))
        self._visible_bias = tf.Variable(tf.constant(0.1, shape=[visible]))

    def forward_prop(self, vis_input):
        return tf.nn.sigmoid(tf.matmul(vis_input, self._weights)) + self._hidden_bias

    def backward_prop(self, hidden_input):
        return self.tf.nn.sigmoid(tf.matmul(hidden_input, tf.transpose(self._weights)))

    def sample_hidden(self, vis):
        probs = forward_prop(vis)
        sample = tf.math.min(0, tf.sign(probs - tf.random_uniform(tf.shape(probs))))
        return sample

    def sample_vis(self, hidden):
        probs = backward_prop(hidden)
        sample = tf.math.min(0, tf.sign(hidden - tf.random_uniform(tf.shape(probs))))
        return sample
