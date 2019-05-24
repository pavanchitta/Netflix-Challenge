import tensorflow as tf
from cond_rbm import CondRBM
from cond_rbm_factor import CondFactorRBM

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

def _qual_parse_function(array):
    split = tf.strings.to_number(tf.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(tf.math.add(split[1:][::2], -1), [-1,1])


    return tf.SparseTensor(indices=indices, values=split[1:][::2], dense_shape=[17770])

dset = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_qual_edited.dta")
train_4_probe = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/train_4_pred_edited.dta")
probe = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/probe_edited.dta")
qual_4_probe = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/qual_for_probe_edited.dta")
full_train = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data/train.tf.dta")



qual = tf.data.TextLineDataset("/Users/matthewzeitlin/Desktop/CS156b-Netflix/data_processing/qual_edited.dta")
qual = qual.map(_qual_parse_function)

# test_set = tf.data.TextLineDataset("/home/CS156b-Netflix/data/probe.dta")

dataset = dset.map(_user_parse_function)
train_4_probe = train_4_probe.map(_user_parse_function)
full_train = full_train.map(_user_parse_function)
probe = probe.map(_test_parse_function)
qual_4_probe = qual_4_probe.map(_qual_parse_function)
#test_set = test_set.map(_test_parse_function)

dataset = tf.data.Dataset.zip((dataset, qual))
probe_dataset = tf.data.Dataset.zip((train_4_probe, qual_4_probe))
#user_index_dataset = dset.map(_user_parse_function)

# a = model.predict(test_set, user_index_dataset)
# print(a)

rbm = CondFactorRBM()
rbm.train(dataset, 20, probe, probe_dataset)


########### Submission ###############





print("Predicting")
test_set = tf.data.TextLineDataset("/home/ubuntu/CS156b-Netflix/deep_autoencoder/qual_edited.dta")
test_set = test_set.map(_test_parse_function)
rbm.pred_for_sub(probe, probe_dataset)
print("Created submission for test set")
# rbm.pred_for_sub(full_train, full_train, True, "cond_rbm_train.txt")
