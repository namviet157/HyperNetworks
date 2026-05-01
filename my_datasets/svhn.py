import numpy as np
from scipy.io import loadmat
import os


class SVHN(object):
    def __init__(self, path="data/svhn"):
        self.num_classes = 10

        train_data = loadmat(os.path.join(path, "train_32x32.mat"))
        test_data = loadmat(os.path.join(path, "test_32x32.mat"))

        # Load images (32x32x3)
        x_train = train_data['X']
        x_test = test_data['X']

        # Convert from (32,32,3,N) -> (N,32,32,3)
        self.x_train = np.transpose(x_train, (3, 0, 1, 2)).astype(np.float32) / 255.0
        self.x_test = np.transpose(x_test, (3, 0, 1, 2)).astype(np.float32) / 255.0

        # Labels
        y_train = train_data['y'].reshape(-1)
        y_test = test_data['y'].reshape(-1)

        # SVHN dùng label "10" cho số 0 → convert về 0
        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0

        # One-hot
        self.y_train = np.eye(self.num_classes, dtype=np.float32)[y_train]
        self.y_test = np.eye(self.num_classes, dtype=np.float32)[y_test]