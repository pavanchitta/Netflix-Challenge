import tensorflow as tf
from model import train_model

tf.enable_eager_execution()

def _parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[::2], -1), [-1,1])
    values = tf.to_float(split[1::2])
    dense_shape = [17770]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

dataset = tf.data.TextLineDataset("/Users/vigneshv/code/CS156b-Netflix/deep_autoencoder/train.tf.dta")
dataset = dataset.map(_parse_function)

model = train_model.TrainModel(train_model.ModelParams(False,1,17770))
model.train(dataset)
