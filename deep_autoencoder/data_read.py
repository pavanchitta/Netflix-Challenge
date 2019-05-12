import tensorflow as tf
from model import train_model

tf.enable_eager_execution()

def _parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = tf.to_float(split[1:][1::2])
    dense_shape = [17770]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

def _user_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    values = tf.to_float(split[1:][1::2])
    dense_shape = [17770]

    return { 0: split[0], 1: tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape) }

def _test_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    return split

dset = tf.data.TextLineDataset("/Users/vigneshv/code/CS156b-Netflix/deep_autoencoder/train.tf.dta")
test_set = tf.data.TextLineDataset("/Users/vigneshv/code/CS156b-Netflix/data/probe.dta")

dataset = dset.map(_parse_function)
test_set = test_set.map(_test_parse_function)

user_index_dataset = dset.map(_user_parse_function)

model = train_model.TrainModel(train_model.ModelParams(False,1,17770))
a = model.predict(test_set, user_index_dataset)
print(a)
# model.train(dataset)
