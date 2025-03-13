import model as md
import numpy as np

X_train = np.array(np.full((2, 2), 0))
y_train = np.array(np.full((2, 100), 0))
y_train[0][0] = 1

X_train[1][0] = 1
y_train[1][1] = 1

# import matplotlib.pyplot as plt

# model = md.Model([
#     md.LayerDeclaration(X_train.shape[1]),
#     md.LayerDeclaration(16),
#     md.LayerDeclaration(16),
#     md.LayerDeclaration(y_train.shape[1], activation_function="softmax")
# ])
# model.train(X_train, y_train, alpha=1, epoches=1500)
# model.util_write_params()

# plt.plot(model.debug_cost_list, label=f"alpha = {1}")

# plt.legend()
# plt.show()

model = md.Model([
    md.LayerDeclaration(X_train.shape[1]),
    md.LayerDeclaration(16),
    md.LayerDeclaration(16),
    md.LayerDeclaration(y_train.shape[1], activation_function="softmax")
], preload_folder="parameters/2025-03-13 20.07.59")

print(model.calculate_and_predict(X_train[0]))
print(model.calculate_and_predict(X_train[1]))