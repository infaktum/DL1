import numpy as np

class Perceptron:
    def __init__(self,N):
        self.w = np.zeros(N)
        self.b = 0.
        
    def output(self, x):
        return int(np.dot(self.w,x) + self.b > 0)
    
    def train(self,X, Y,alpha=0.01,maxiter=10):
        for _ in range(maxiter):
            for x,y in zip(X,Y):
                o = self.output(x)  
                self.w += alpha * (y - o) * x
                self.b += alpha * (y - o)

    def __str__(self):
        return f'w = {self.w}, b = {self.b}'

