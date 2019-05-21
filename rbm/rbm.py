import tensorflow as tf
from tensorflow import contrib
tfe = tf.contrib.eager

class RBM:
    def __init__(self, n_visible, n_hidden, k, batch_size, weight_decay, momentum):
        self.n_visible = n_visible
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.momentum = momentum
        self.weight_decay = weight_decay
        # k is number of iterations of Gibbs sampling for CD_k
        self.k = k
        self.num_rat = 5

        self.lr = 0.01 / self.batch_size
        # if momentum:
        #     self.momentum = tf.placeholder(tf.float32)
        # else:
        #     self.momentum = 0.0
        # Initialize weights and biases
        initial_weights = tf.truncated_normal([n_visible, self.num_rat, n_hidden], stddev=0.01)
        initial_weights = tf.reduce_mean(initial_weights, axis=1)
        initial_weights = tf.stack([initial_weights, initial_weights, initial_weights, initial_weights, initial_weights], axis=1)
        self.weights = tfe.Variable(initial_weights, name="weights")

        self.hidden_bias = tfe.Variable(tf.constant(0.1, shape=[n_hidden]), name='h_bias')
        self.visible_bias = tfe.Variable(tf.constant(0.1, shape=[self.num_rat, n_visible]), name='v_bias')

        # Initialize momentum velocities
        # self.w_v = tfe.Variable(tf.zeros([n_visible, n_hidden, num_rat]), dtype=tf.float32)
		# self.hb_v = tfe.Variable(tf.zeros([n_hidden]), dtype=tf.float32)
		# self.vb_v = tfe.Variable(tf.zeros([n_visible, num_rat]), dtype=tf.float32)

    def forward_prop(self, visible):
        '''Computes a vector of probabilities hidden units are set to 1 given
        visible unit states'''

        pre_activation = tf.add(tf.tensordot(visible, self.weights, [[1, 2], [1, 0]]), self.hidden_bias)
        return tf.nn.sigmoid(pre_activation)


    def backward_prop(self, hidden):
        '''Computes a vector of probabilities visible units are set to 1 given
        hidden unit states'''
        return tf.nn.softmax(tf.add(tf.transpose(tf.tensordot(hidden, self.weights, [[1], [2]]), perm=[0, 2, 1]), self.visible_bias), axis=1)

    def sample_h_given_v(self, visible_sample, binary=True):
        '''Sample a hidden unit vector based on state probabilities given visible unit states.'''
        hidden_probs = self.forward_prop(visible_sample)
        if binary:
        # Relu is just to convert -1 to 0 for sampled vector
            return tf.nn.relu(tf.sign(tf.subtract(hidden_probs, tf.random_uniform(tf.shape(hidden_probs)))))
        else:
            return hidden_probs

    def sample_v_given_h(self, hidden_sample, binary=False):
        '''Sample a visible unit vector based on state probabilities given hidden unit states.'''
        visible_probs = self.backward_prop(hidden_sample)
        # Relu is just to convert -1 to 0 for sampled vector
        if binary:
            return tf.nn.relu(tf.sign(tf.subtract(visible_probs, tf.random_uniform(tf.shape(visible_probs)))))
        else:
            return visible_probs



    def CD_k(self, visibles, mask):
        '''Contrastive divergence with k steps of Gibbs Sampling.'''
        orig_hidden = self.sample_h_given_v(visibles)
        # k steps of alternating parallel updates
        for i in range(self.k):
            if i == 0:
                hidden_samples = orig_hidden
            visible_samples = self.sample_v_given_h(hidden_samples) * mask
            if i == self.k - 1:
                hidden_samples = self.sample_h_given_v(visible_samples, binary=False)
            else:
                hidden_samples = self.sample_h_given_v(visible_samples, binary=True)







        w_grad_pos = tf.reduce_sum(tf.transpose(tf.tensordot(visibles, orig_hidden, [[0], [0]]), perm=[1, 0, 2]), axis=1)
        #print(w_grad_pos)
            # Second term, based on reconstruction from Gibbs Sampling
        w_neg_grad = tf.reduce_sum(tf.transpose(tf.tensordot(visible_samples, hidden_samples, [[0], [0]]), perm=[1, 0, 2]), axis=1)
        w_grad_tot = tf.subtract(w_grad_pos, w_neg_grad)

        # Calculate total gradient, accounting for expectation
        w_grad_tot = tf.stack([w_grad_tot, w_grad_tot, w_grad_tot, w_grad_tot, w_grad_tot], axis=1)
        # Bias gradients
        hb_grad = tf.reduce_sum(tf.subtract(orig_hidden, hidden_samples), axis=0)
        vb_grad = tf.reduce_sum(tf.subtract(visibles, visible_samples), axis=0)

        return w_grad_tot, hb_grad, vb_grad

    def learn(self, visibles):

        reduced_visibles = tf.reduce_sum(visibles, axis=1)

        mask = tf.where(tf.equal(reduced_visibles, 0), tf.zeros_like(reduced_visibles), tf.ones_like(reduced_visibles))
        mask = tf.stack([mask, mask, mask, mask, mask], axis=1)

        weight_grad, hidden_bias_grad, visible_bias_grad = self.CD_k(visibles, mask)
        return [weight_grad, hidden_bias_grad, visible_bias_grad]
        # Compute new velocities
        # new_w_v = self.momentum * self.w_v + self.lr * weight_grad
        # new_hb_v = self.momentum * self.hb_v + self.lr * hidden_bias_grad
        # new_vb_v = self.momentum * self.vb_v + self.lr * visible_bias_grad

        # new_w_v = self.lr * weight_grad
        # new_hb_v = self.lr * hidden_bias_grad
        # new_vb_v = self.lr * visible_bias_grad
        # Update parameters
        # self.weights = tf.add(self.weights, tf.scalar_mul(-self.lr, weight_grad))
        # self.hidden_bias = tf.add(self.hidden_bias, tf.scalar_mul(-self.lr, hidden_bias_grad))
        # self.visible_bias = tf.add(self.visible_bias, tf.scalar_mul(-self.lr, visible_bias_grad))
        # Update velocities
        # update_w_v = tf.assign(self.w_v, new_w_v)
        # update_hb_v = tf.assign(self.hb_v, new_hb_v)
        # update_vb_v = tf.assign(self.vb_v, new_vb_v)
        #return [update_w, update_hb, update_vb]



    def train(self, dataset, epochs, probe_set, probe_train):
        # Computation graph definition
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        optimizer = tf.contrib.opt.MomentumWOptimizer(self.weight_decay, self.lr, self.momentum)
        # Main training loop, needs adjustments depending on how training data is handled
        #print(self.visible_bias)
        for _ in range(epochs):
            num_pts = 0

            try:
                while True:
                    print(num_pts)
                    if num_pts % 100 == 0:

                        print(self.weights)
                        print(self.visible_bias)
                        print(self.hidden_bias)
                    if num_pts % 2000 == 0 and num_pts != 0:

                        #print(self.visible_bias)
                        #print(num_pts)
                        self.pred_with_RMSE(probe_set, probe_train)

                    training_point = iterator.get_next()
                    x = tf.sparse.to_dense(training_point, default_value = -1)

                    x_hot = tf.one_hot(x, self.num_rat, axis=1)
                    grads = self.learn(x_hot)
                    optimizer.apply_gradients(zip(grads, [self.weights, self.hidden_bias, self.visible_bias]))
                    num_pts += 1
            except tf.errors.OutOfRangeError:
                ds = dataset.shuffle(460000)
                batched_dataset = ds.batch(self.batch_size)
                iterator = batched_dataset.make_one_shot_iterator
        predictions, RMSE = self.pred_with_RMSE(probe_set, probe_train)
        print(predictions)
        print(RMSE)


    def pred_with_RMSE(self, test_set, pred_set):
        test_set = test_set.repeat(1)
        test_set = test_set.batch(self.batch_size)
        test_iterator = test_set.make_one_shot_iterator()
        pred_set = pred_set.batch(self.batch_size)
        pred_iterator = pred_set.make_one_shot_iterator()

        training_point = pred_iterator.get_next()
        x = tf.sparse.to_dense(training_point, default_value = -1)
        x_hot = tf.one_hot(x, self.num_rat, axis=1)
        batch_count = 0
        curr_preds = self.forward(x_hot)


        batch_count = 0
        total_predictions = []
        actual = []
        try:
            while True:
                batch_count += 1

                row_batch = test_iterator.get_next()
                test_preds = tf.gather_nd(curr_preds, row_batch.indices)
                print(test_preds)

                RMSE = tf.math.sqrt(test_preds, row_batch.values)
                print(RMSE)

                if batch_count % 10 == 0:

                    print(test_preds)


                total_predictions = tf.concat([total_predictions, test_preds], 0)
                actual = tf.concat([actual, row_batch.values], 0)
                batch_count += self.batch_size
                training_point = pred_iterator.get_next()
                x = tf.sparse.to_dense(training_point, default_value = -1)
                x_hot = tf.one_hot(x, self.num_rat, axis=1)
                curr_preds = self.forward(x_hot)

        except tf.errors.OutOfRangeError:
            pass

        total_error = tf.math.sqrt(tf.losses.mean_squared_error(total_predictions, actual))

        return total_predictions, total_error


    def forward(self, visibles):
        hidden_samples = self.sample_h_given_v(visibles)
        visible_samples = self.sample_v_given_h(hidden_samples)
        scale = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
        return tf.reduce_sum(tf.tensordot(scale, visible_samples, [[1], [1]]), axis=0)
