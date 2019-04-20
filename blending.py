# Input preds as a matrix of ratings for each of the models, with each column
# as the predictions from 1 model. Input rmses as an array of scoreboard rmse
# for each model.

import numpy as np

DATA_LEN = 2749898
RATINGS_SQUARED = np.square(3.84358) * DATA_LEN

def quiz_blend(preds, rmses):
    pred_squared = np.sum(np.square(preds), axis=0)
    squared_errors = np.square(rmses) * DATA_LEN
    coefs = 0.5 * (pred_squared + RATINGS_SQUARED - squared_errors)
    coefs = np.matmul(np.linalg.inv(np.matmul(np.transpose(preds), preds)), coefs)
    blended_preds = np.matmul(preds, coefs)
    return blended_preds
