import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gallery(images, rows, cols, cmap=None):
    for n in range(rows * cols):          
        plt.subplot(rows, cols,n+1)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)          
        plt.imshow(images[n], cmap) 


def draw_vector(v, o = (0,0), c = 'black'):
    l = np.linalg.norm(v)
    plt.arrow(o[0],o[1],v[0],v[1], head_width=0.05, head_length=0.05, fc='c', ec='c')     

def plot_separator(x1_start,x1_end,w,b):
    try:
        line =  [  [x1_start,x1_end  ], [- (b + w[0] * x1_start) / w[1],- (b + w[0] * x1_end)   / w[1] ]]
        plt.plot(line[0],line[1],c='black')    
    except Exception:
        pass;

def plot_weights(w):
    w_scaled = 0.2 * w / np.linalg.norm(w)
    plt.arrow(0.5,0.5,w_scaled[0],w_scaled[1], head_width=0.05, head_length=0.05, fc='black', ec='black') 

def plot_dots(p,X):

    dots = [255 * p.predict(x) for x in X] 
    plt.scatter(X[:,0],X[:,1],cmap=cm.coolwarm, c = dots);     
    
def plot_perceptron(Y):
    X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
    plt.figure(figsize=(15,10))
    x1_start = x2_start = -0.25
    x1_end = x2_end = 1.25
    density = 0.03    
    input = np.array([[x1,x2] for x1 in np.arange(x1_start,x1_end,density) for x2 in np.arange(x2_start,x2_end,density)])
    
    for n,y in enumerate(Y):
        p = Perzeptron(2)
        p.fit(X,y)        
   
        ax = plt.subplot(2,3,n+1)
        ax.axis([x1_start,x1_end,x1_start,x1_end])
        plt.title(f'y= {y}, w,b = {p.w}, {p.b}')
        
        colors = [p.forward(x) * 255 for x in input]        
        plt.scatter(input[:,0],input[:,1],cmap=cm.coolwarm,c=colors); 

        plot_dots(p,X)        
        plot_separator(x1_start,x1_end,p.w,p.b)
        plot_weights(p.w)
        

class Perzeptron:
    def __init__(self,n):
        self.w = np.zeros(n)
        self.b = 0.
    
    def predict(self,x):
        return 1 if self.forward(x) > 0 else 0
    
    def fit(self,x_train,y_train,epsilon=0.01,iter=100):
        for _ in range(iter):
            for x,y in zip(x_train, y_train):
                self.hebb(x,y,epsilon)

    def forward(self, x):
        return np.sum(self.w * x) + self.b        
        
    def hebb(self,x,y,epsilon):
        o = self.predict(x)  
        self.w += epsilon * (y - o) * x
        self.b += epsilon * (y - o)

    def score(self, x, y):
        return np.sum([self.predict(x) == y]) / len(x)

    def __str__(self):
        return f'Gewichte: {self.w}, Bias: {self.b}'