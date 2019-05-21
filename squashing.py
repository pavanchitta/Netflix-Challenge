import numpy as np
import os

count = 1
for filename in os.listdir(os.getcwd() + '/to_fix'):
    if filename.endswith(".txt"):
        print(count)
        count += 1
        pred = np.loadtxt('to_fix/' + filename)
        pred[pred < 1] = 1
        pred[pred > 5] = 5
        np.savetxt(filename, pred, fmt='%.5f')
