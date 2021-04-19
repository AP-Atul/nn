import numpy as np


class mul:
    @staticmethod
    def forward(w, x):
        return np.dot(x, w)

    @staticmethod
    def backward(w, x, dz):
        dw = np.dot(np.transpose(x), dz)
        dx = np.dot(dz, np.transpose(w))
        return dw, dx


class add:
    @staticmethod
    def forward(x, b):
        return x + b

    @staticmethod
    def backward(x, dz):
        dx = dz * np.ones_like(x)
        db = np.dot(np.ones((1, dz.shape[0]), dtype=np.float64), dz)
        return db, dx
