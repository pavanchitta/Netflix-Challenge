import tensorflow as tf
from model import train_model

dataset = tf.data.TextLineDataset("/Users/vigneshv/code/CS156b-Netflix/data/train.dta")
iterator = dataset.make_one_shot_iterator()
sess = tf.Session()

model = train_model.TrainModel(train_model.ModelParams(False, 0, 17770))
model.train(dataset)
