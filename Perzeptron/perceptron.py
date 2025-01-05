import numpy as np

class Perceptron:
    def __init__(self,N):
        self.w = np.zeros(N)
        self.b = 0.

    def fit(self, X, Y, alpha=0.01, max_iter=10):
        for _ in range(max_iter):
            for x, y in zip(X, Y):
                o = self.output(x)
                self.w += alpha * (y - o) * x
                self.b += alpha * (y - o)
        
    def output(self, x):
        return int(np.dot(self.w,x) + self.b > 0)

    def __str__(self):
        return f'w = {self.w}, b = {self.b}'

