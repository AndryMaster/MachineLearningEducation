import matplotlib.pyplot as plt
from time import time
import numpy as np

from som import SOM

start = time()

colors = np.array(
     [[0., 0., 1.],
      [0., 0., 0.95],
      [0., 0.05, 1.],
      [0., 1., 0.],
      [0., 0.95, 0.],
      [0., 1, 0.05],
      [0.98, 0., 0.],
      [1., 0.05, 0.],
      [1., 0., 0.05],
      [1., 1., 0.]])

# data = np.array([[1., 0., 0., 0.],
#                  [0., 1., 0., 0.],
#                  [0., 0., 1., 0.],
#                  [0., 0., 0., 1.],
#                  [0., 0.05, 0., 0.9],
#                  [0.1, 0., 1., 0.05],
#                  [0.05, 0.95, 0.05, 0.],
#                  [1., 0., 0., 0.],
#                  [1., 1., 0., 0.],
#                  [0., 0., 1., 0.95],
#                  [0.9, 1., 0.05, 0.],
#                  [0.05, 0.1, 0.95, 1.],
#                  [0.05, 1., 1., 0.],
#                  [0., 0.9, 0.95, 0.05],
#                  [1., 1., 1., 1.],
#                  [0.9, 0.95, 0.98, 1.]])

# colors = np.array(  # 7
#      [[0., 0., 1.],
#       [0., 0., 0.95],
#       [0., 0.05, 1.],
#       [0., 1., 0.],
#       [0., 0.95, 0.],
#       [0., 1, 0.05],
#       [1., 0., 0.],
#       [1., 0.05, 0.],
#       [1., 0., 0.05],
#       [1., 1., 0.05],
#       [0.9, 0.95, 0.],
#       [0.05, 1., 0.9],
#       [0., 0.9, 0.95],
#       # [1., 0., 1.],
#       # [0.95, 0.05, 1.],
#       [1., 1., 1.]])

# colors = np.array(
#     [[0., 0., 1.],
#      [0.05, 0., 0.9],
#      [0.1, 0., 0.95],
#      [1., 0., 0.],
#      [1., 0., 0.05]])

SOM.num_iters = 1500
som = SOM(16, 16, 3)
som.train(colors)

print(f"\nTime: {time() - start :.2f}")

plt.imshow(som.centroid_grid)
plt.show()
