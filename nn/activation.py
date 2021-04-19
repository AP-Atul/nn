import numpy as np


class sigmoid:
    @staticmethod
    def forward(x):
        return 1.0 / (1.0 + np.exp(- x))

    @staticmethod
    def backward(x, top_diff):
        output = sigmoid.forward(x)
        return (1.0 - output) * output * top_diff


class tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x, top_diff):
        output = tanh.forward(x)
        return (1.0 - np.square(output)) * top_diff


class softmax:
    @staticmethod
    def predict(x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def loss(x, y):
        num_examples = x.shape[0]
        prob = softmax.predict(x)
        logprob = -np.log(prob[range(num_examples), y])
        data_loss = np.sum(logprob)
        return 1. / num_examples * data_loss

    @staticmethod
    def diff(x, y):
        num_examples = x.shape[0]
        prob = softmax.predict(x)
        prob[range(num_examples), y] -= 1
        return prob
