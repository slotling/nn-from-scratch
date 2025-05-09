import model as md
import numpy as np
from dataload import load_csv
import matplotlib.pyplot as plt

train = np.array(load_csv("data/mnist_train.csv", 1000))
test = np.array(load_csv("data/mnist_test.csv", 100))

X_train = train[:, 1:] / 255
X_test = test[:, 1:] / 255
y_train_raw = train[:, 0]
y_test_raw = test[:, 0]
y_train = np.zeros((y_train_raw.shape[0], y_train_raw.max()+1))
y_test = np.zeros((y_test_raw.shape[0], y_train_raw.max()+1))
y_train[np.arange(y_train_raw.shape[0]), y_train_raw] = 1
y_test[np.arange(y_test_raw.shape[0]), y_test_raw] = 1

model = md.Model([
    md.LayerDeclaration(X_train.shape[1]),
    md.LayerDeclaration(16),
    md.LayerDeclaration(16),
    md.LayerDeclaration(y_train.shape[1], activation_function="softmax")
], md.TrainDeclaration(alpha=0.5, epoches=500))
model.train(X_train, y_train)
model.util_write_params()
print(model.calculate_and_predict(X_test[0]), y_test[0])
plt.plot(model.debug_cost_list, label=f"cost")
plt.legend()
plt.show()

cost, acc = model.evaluate(X_test, y_test)
print(cost, acc)