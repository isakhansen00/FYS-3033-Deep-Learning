import numpy as np
from skimage.data import camera


x = np.array(camera()).reshape(1, 1, 512, 512)
w = np.concatenate((np.array([[1, 2, 1], [0, 0, 0],
                              [-1, -2, -1]]).reshape(1, 1, 3, 3),
                    np.array([[1, 0, -1], [2, 0, -2],
                              [1, 0, -1]]).reshape(1, 1, 3, 3)))

print(f'Image Shape: {x.shape}')
print(f'Filter Shape: {w.shape}')