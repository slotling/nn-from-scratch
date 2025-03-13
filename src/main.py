import model as md
import numpy as np
from dataload import load_csv

train = np.array(load_csv("data/mnist_train.csv", 1000))
test = np.array(load_csv("data/mnist_test.csv", 5))

X_train = train[:, 1:] / 255
X_test = test[:, 1:] / 255
y_train_raw = train[:, 0]
y_test_raw = test[:, 0]
y_train = np.zeros((y_train_raw.shape[0], y_train_raw.max()+1))
y_test = np.zeros((y_test_raw.shape[0], y_train_raw.max()+1))
y_train[np.arange(y_train_raw.shape[0]), y_train_raw] = 1
y_test[np.arange(y_test_raw.shape[0]), y_test_raw] = 1

models = []

import matplotlib.pyplot as plt
for alpha in [2]:
    model = md.Model([
        md.LayerDeclaration(X_train.shape[1]),
        md.LayerDeclaration(16),
        md.LayerDeclaration(16),
        md.LayerDeclaration(y_train.shape[1], activation_function="softmax")
    ])
    model.train(X_train, y_train, alpha=alpha, epoches=50)
    print(model.calculate_and_predict(X_test[0]), y_test[0])
    plt.plot(model.debug_cost_list, label=f"cost")
plt.plot(model.debug_acc_list, label=f"acc")
plt.legend()
plt.show()