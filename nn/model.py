import numpy as np
from . import mul, add, softmax, tanh


class Model:
    def __init__(self, layers_dim, epochs=2000, epsilon=0.01, reg=0.01, debug=False):
        (self.b, self.W, self.epochs,
         self.epsilon, self.reg, self.debug) = (list(), list(), epochs,
                                                epsilon, reg, debug)
        for i in range(len(layers_dim) - 1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i + 1]) / np.sqrt(layers_dim[i]))
            self.b.append(np.random.randn(layers_dim[i + 1]).reshape(1, layers_dim[i + 1]))

    def loss(self, x, y):
        data = x
        for i in range(len(self.W)):
            m = mul.forward(self.W[i], data)
            a = add.forward(m, self.b[i])
            data = tanh.forward(a)

        return softmax.loss(data, y)

    def predict(self, x):
        data = x
        for i in range(len(self.W)):
            m = mul.forward(self.W[i], data)
            a = add.forward(m, self.b[i])
            data = tanh.forward(a)

        return np.argmax(softmax.predict(data), axis=1)

    def train(self, x, y):
        for epoch in range(self.epochs):
            # Forward propagation
            data = x
            forward = [(None, None, data)]
            for i in range(len(self.W)):
                m = mul.forward(self.W[i], data)
                a = add.forward(m, self.b[i])
                data = tanh.forward(a)
                forward.append((m, a, data))

            # Back propagation
            dtanh = softmax.diff(forward[len(forward) - 1][2], y)
            for i in range(len(forward) - 1, 0, -1):
                dadd = tanh.backward(forward[i][1], dtanh)
                db, dmul = add.backward(forward[i][0], dadd)
                dw, dtanh = mul.backward(self.W[i - 1], forward[i - 1][2], dmul)
                # Add regularization terms (b1 and b2 don't have regularization terms)
                dw += self.reg * self.W[i - 1]
                # Gradient descent parameter update
                self.b[i - 1] += - self.epsilon * db
                self.W[i - 1] += - self.epsilon * dw

            if self.debug and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" % (epoch, self.loss(x, y)))
