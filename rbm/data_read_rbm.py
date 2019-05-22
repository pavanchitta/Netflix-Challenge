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

    return (tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape))

def _test_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = split[1:][1::2]
    dense_shape = [17770]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

dset = tf.data.TextLineDataset("/home/ubuntu/CS156b-Netflix/deep_autoencoder/train_4_qual_edited.dta")
train_4_probe = tf.data.TextLineDataset("/home/ubuntu/CS156b-Netflix/deep_autoencoder/train_4_pred_edited.dta")
probe = tf.data.TextLineDataset("/home/ubuntu/CS156b-Netflix/deep_autoencoder/probe_edited.dta")
#test_set = tf.data.TextLineDataset("/Users/vigneshv/code/CS156b-Netflix/data/probe.dta")

dataset = dset.map(_user_parse_function)
train_4_probe = train_4_probe.map(_user_parse_function)
probe = probe.map(_test_parse_function)
#test_set = test_set.map(_test_parse_function)

#user_index_dataset = dset.map(_user_parse_function)

#a = model.predict(test_set, user_index_dataset)
#print(a)

rbm = RBM(17770, 100, 1, 100, 0, 0.9)
rbm.train(dataset, 0, probe, train_4_probe)
