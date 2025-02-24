import numpy as np

class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def fit(self, X):
        for i in range(self.n):
            for j in range(self.n):
                self.W[i,j] = sum([X[k][i] * X[k][j] for k in range(len(X))]) / self.n
        np.fill_diagonal(self.W, 0)

    def train(self, patterns):
        for pattern in patterns:
            flat_pattern = pattern.flatten()
            self.W += np.outer(flat_pattern, flat_pattern)
        np.fill_diagonal(self.W, 0)
        self.W /= len(patterns)

    def recall(self, x, steps=1000):
        x_flat = x.flatten()
        for _ in range(steps):
            x_flat = np.sign(self.W @ x_flat)
        return x_flat.reshape(x.shape)

if __name__ == "__main__":
    X = np.array([[1, 1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, 1]])
    hopfield = Hopfield(4)
    hopfield.train(X)
    print(hopfield.W)

    print(hopfield.recall(np.array([1, 1, 1, 1])))
