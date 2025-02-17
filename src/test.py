import model as md
import numpy as np
from dataload import load_csv

X_train = np.array([[1]])
y_train = np.array([[0]])
model = md.Model([1, 1, 1])
model.train(X_train, y_train)

print(model.predict(X_train[0]), y_train[0])

import matplotlib.pyplot as plt
plt.plot(model.debug_cost_list)
plt.show()