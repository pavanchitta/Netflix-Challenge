import tensorflow as tf
from rbm import RBM

tf.enable_eager_execution()

def _parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = tf.add(split[1:][1::2], -1)
    dense_shape = [17770]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

def _user_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = tf.add(split[1:][1::2], -1)
    dense_shape = [17770]

    return {0: split[0], 1: tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)}

def _test_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = split[1:][1::2]
    dense_shape = [17770]

    return {0: split[0], 1: tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)}

dset = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_qual_edited.dta")
train_4_probe = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_pred_edited.dta")
probe = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/probe_edited.dta")
full_train = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data/train.tf.dta")

# test_set = tf.data.TextLineDataset("/home/CS156b-Netflix/data/probe.dta")

dataset = dset.map(_user_parse_function)
train_4_probe = train_4_probe.map(_user_parse_function)
full_train = full_train.map(_user_parse_function)
probe = probe.map(_test_parse_function)
#test_set = test_set.map(_test_parse_function)

#user_index_dataset = dset.map(_user_parse_function)

# a = model.predict(test_set, user_index_dataset)
# print(a)

rbm = RBM()
rbm.train(dataset, 50, probe, train_4_probe)

########### Submission ###############
exit(0)
saver = tf.contrib.eager.Saver(rbm.get_variables())
saver.restore("models522/rbm")

print("Predicting")
test_set = tf.data.TextLineDataset("/home/ubuntu/CS156b-Netflix/deep_autoencoder/qual_edited.dta")
test_set = test_set.map(_test_parse_function)
rbm.pred_for_sub(test_set, dataset, True, "rbm_probe_qual.txt")
# print("Created submission for test set")
# rbm.pred_for_sub(full_train, full_train, True, "rbm_train.txt")
