import model as md
import numpy as np
from dataload import load_csv

train = np.array(load_csv("data/mnist_train.csv", 1))
test = np.array(load_csv("data/mnist_test.csv", 5))

import matplotlib.pyplot as plt
for i in [2, 3, 5, 10, 20, 35, 50]:
    
    X_train = np.array([np.full((2), 1)])
    y_train = np.array([np.full((i), 1)])
    y_train[0][0] = 0

    model = md.Model([X_train.shape[1], 16, y_train.shape[1]])
    model.train(X_train, y_train, alpha=1)
    print(model.calculate_and_predict(X_train[0]))

    plt.plot(model.debug_cost_list, label=f"output neurons = {i}")

plt.legend()
plt.show()