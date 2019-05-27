import keras
import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

SINGLE_EMBEDDING = 100
DOUBLE_EMBEDDING = 50

user_input = Input(shape=(1,), dtype='int32', name='user')
movie_input = Input(shape=(1,), dtype='int32', name='movie')
date_input = Input(shape=(1,1), dtype='float32', name='date')

user_embedding = Embedding(output_dim=SINGLE_EMBEDDING, input_dim=458293, input_length=1, 
    embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), 
    embeddings_regularizer=keras.regularizers.l2(0.0))(user_input)

movie_embedding = Embedding(output_dim=SINGLE_EMBEDDING, input_dim=17770, input_length=1,
    embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    embeddings_regularizer=keras.regularizers.l2(0.0))(movie_input)

user_embedding_2 = Embedding(output_dim=DOUBLE_EMBEDDING, input_dim=458293, input_length=1,
    embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    embeddings_regularizer=keras.regularizers.l2(0.0))(user_input)
movie_embedding_2 = Embedding(output_dim=DOUBLE_EMBEDDING, input_dim=17770, input_length=1,
    embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    embeddings_regularizer=keras.regularizers.l2(0.0))(movie_input)

merged_layer = keras.layers.Multiply()([user_embedding_2, movie_embedding_2])

total_input = keras.layers.concatenate([user_embedding, movie_embedding, merged_layer, date_input])
x = Dense(200, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.0001, seed=None), activation='relu')(total_input)
x = Dropout(0.2)(x)
output = Dense(1, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.0001, seed=None), activation='relu', name='rating')(x)

model = Model(inputs=[user_input, movie_input, date_input], outputs=[output])
model.load_weights("train-nnmf-weights-02-0.8437.hdf5")
adam = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error')

train_df = pd.read_csv("/home/ubuntu/DeepRec/data/train2.dta", sep=" ", names=["user_id", "item_id", "time", "rating"])
test_df = pd.read_csv("/home/ubuntu/DeepRec/data/probe.dta", sep=" ", names=["user_id", "item_id", "time", "rating"])
# qual_df = pd.read_csv("/home/ubuntu/DeepRec/data/qual.dta", sep=" ", names=["user_id", "item_id", "time"])

# qusers = qual_df["user_id"].values - 1
# qmovies = qual_df["item_id"].values - 1
# qdates = qual_df["time"].values / 2243

# preds = np.reshape(model.predict([qusers, qmovies, np.reshape(qdates, (-1, 1, 1))]), (-1))
# np.savetxt("nnmf-0.8437-preds.txt", preds)

users = train_df["user_id"].values - 1
movies = train_df["item_id"].values - 1
ratings = train_df["rating"].values
dates = train_df["time"].values / 2243

users_test = test_df["user_id"].values - 1
movies_test = test_df["item_id"].values - 1
ratings_test = test_df["rating"].values
dates_test = test_df["time"].values / 2243

filepath="train-nnmf-weights-{epoch:02d}-{val_loss:.4f}-2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit([users, movies, np.reshape(dates, (-1, 1, 1))], [np.reshape(ratings, (-1, 1, 1))], epochs=10, batch_size=16384, 
    validation_data=([users_test, movies_test, np.reshape(dates_test, (-1, 1, 1))], [np.reshape(ratings_test, (-1, 1, 1))]), callbacks=callbacks_list)
