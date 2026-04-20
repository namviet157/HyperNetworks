import numpy as np
from tensorflow.keras.datasets import cifar10


class Cifar10(object):
    def __init__(self):
        self.num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        self.x_train = x_train.astype(np.float32) / 255.0
        self.x_test = x_test.astype(np.float32) / 255.0

        y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
        self.y_train = np.eye(self.num_classes, dtype=np.float32)[y_train]
        self.y_test = np.eye(self.num_classes, dtype=np.float32)[y_test]
