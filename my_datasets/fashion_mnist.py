import numpy as np
from tensorflow.keras.datasets import fashion_mnist


class FashionMnist(object):
    def __init__(self):
        self.num_classes = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Normalize + reshape (28x28x1)
        self.x_train = (np.expand_dims(x_train, axis=3).astype(np.float32)) / 255.0
        self.x_test = (np.expand_dims(x_test, axis=3).astype(np.float32)) / 255.0

        # One-hot labels
        y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)

        self.y_train = np.eye(self.num_classes, dtype=np.float32)[y_train]
        self.y_test = np.eye(self.num_classes, dtype=np.float32)[y_test]