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


    values = tf.to_float(split[1:][1::2])
    dense_shape = [17770]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


def _test_parse_function(array):
    '''Prep probe data.'''
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    movies = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])
    ratings = tf.to_float(split[1:][1::2])

    return tf.SparseTensor(indices=movies, values=ratings, dense_shape=[17770])



dset = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_pred_edited.dta")
probe_set = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/probe_edited.dta")
model = train_model.TrainModel(train_model.ModelParams(False,1,17770))

dataset = dset.map(_user_parse_function)

probe_set = probe_set.map(_test_parse_function)

predict_user_dataset = dset.map(_user_parse_function)

model.train(dataset, probe_set, predict_user_dataset)
########### Submission ###############

test_set = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/qual_edited.dta")
dset_train_for_qual = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_qual_edited.dta")
test_set = test_set.map(_test_parse_function)
dset_train_for_qual = dset_train_for_qual.map(_user_parse_function)
model.pred_for_sub(test_set, dset_train_for_qual)
