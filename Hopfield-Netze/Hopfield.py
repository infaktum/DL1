import numpy as np

class Hopfield():
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def fit(self, X):
        for i in range(self.n):
            for j in range(self.n):
                self.W[i,j] = sum([X[k][i] * X[k][j] for k in range(len(X))]) / self.n
        np.fill_diagonal(self.W, 0)

    def recall(self, x, steps=100):
        x_flat = x.flatten()
        for _ in range(steps):
            x_flat = np.sign(self.W @ x_flat)
        return x_flat.reshape(4)

if __name__ == "__main__":
    X = np.array([[1, 1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, 1]])
    hopfield = Hopfield(4)
    hopfield.fit(X)
    print(hopfield.W)

    print(hopfield.recall(np.array([1, 1, -1, -1])))
