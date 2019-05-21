import numpy as np
import os

DATA_LEN = 2749898
RATINGS_SQUARED = np.square(3.84358) * DATA_LEN

def quiz_blend(preds, rmses):
    pred_squared = np.sum(np.square(preds), axis=0)
    squared_errors = np.square(rmses) * DATA_LEN
    coefs = 0.5 * (pred_squared + RATINGS_SQUARED - squared_errors)
    coefs = np.matmul(np.linalg.inv(np.matmul(np.transpose(preds), preds)), coefs)
    blended_preds = np.matmul(preds, coefs)
    return coefs, blended_preds

rmses = []
preds = []
models = []

count = 1
for filename in os.listdir(os.getcwd() + '/preds'):
    if filename.endswith(".txt"):
        print(count)
        count += 1
        models.append(filename + '     ')
        pred = np.loadtxt('preds/' + filename)
        preds.append(pred)
        rmse = round(int(filename.split('_')[-1].split('.')[0]) * 10**(-5), 5)
        rmses.append(rmse)

preds = np.column_stack(preds)
coefs, blended_preds = quiz_blend(preds, rmses)
model_coefs = np.column_stack((coefs, models))
np.savetxt('blended_preds.txt', blended_preds, fmt='%.5f')
np.savetxt('model_coefs.txt', model_coefs, delimiter=" ", fmt="%s")
