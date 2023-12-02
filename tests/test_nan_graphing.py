import numpy as np
import matplotlib.pyplot as plt

arr = np.eye(10)
arr_nans = arr.copy()
arr_nans[arr_nans == 0] = np.nan
# Fixed 10 x 10 maze
big_maze = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 3, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["white", "black", "lightseagreen", "lawngreen"])


plt.imshow(big_maze, cmap=cmap)

x = np.arange(0, 10)
y = np.arange(0, 10)

plt.pcolormesh(x, y, arr_nans)
plt.show()
