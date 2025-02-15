import numpy as np

mat = np.array([1,2,3,4]).reshape(2, -1)
vec = np.array([1,2]).reshape(-1)
print(np.dot(mat, vec))