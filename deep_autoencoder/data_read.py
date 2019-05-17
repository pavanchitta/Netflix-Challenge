import tensorflow as tf
from model import train_model

tf.enable_eager_execution()

# def _parse_function(array):
#     '''For training.'''
#     split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
#     indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])
#
#
#     values = tf.to_float(split[1:][1::2])
#     dense_shape = [17770]
#
#     return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

def _user_parse_function(array):
    '''For predicting'''
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])

    sess = tf.Session()
    with sess.as_default():
        print(indices.eval())

    values = tf.to_float(split[1:][1::2])
    dense_shape = [17770]
    tensor = tf.sparse.to_dense(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape))

    return tensor

def _test_parse_function(array):
    '''Prep probe data.'''
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    movies = tf.to_float(tf.math.add(split[1:][::2], -1));
    ratings = tf.to_float(split[1:][1::2])

    return { 0: movies, 1: ratings }

dset = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_pred_edited.dta")
test_set = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/probe_edited.dta")
model = train_model.TrainModel(train_model.ModelParams(False,1,17770))

#dataset = dset.map(_parse_function)
test_set = test_set.map(_test_parse_function)

predict_user_dataset = dset.map(_user_parse_function)


predictions = model.better_pred(test_set, predict_user_dataset)
#model.train(dataset)
