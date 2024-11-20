import numpy as np


def loss(out,exp):
    """
    Die Verlustfunktion
    """
    return np.sum((out - exp)**2) / len(out)

# Mögliche Transferfunktionen

def sigmoid(x):
    """Die logistische Funktion"""
    return 1. / (1. + np.exp(-x))

def softmax(x):
    # Funktioniert auch mit sehr großen Exponenten
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def leaky_relu(x):
    """ Die Leaky-ReLU-Funktion"""
    return np.maximum(0.2 * x,x)

def output(w,x,transfer = sigmoid):
    return sigmoid(np.sum(np.dot(w,x)))

# Die Ableitungen unserer Funktionen  

def d_loss(out,expd):
    """Ableitung der Verlustfunktion"""
    return 2 * np.sum(out - expd) / len(out)
    
def d_sigmoid(x):
    sigm = sigmoid(x)
    return sigm * (1 - sigm)

def d_leaky_relu(x):
    return 0.2 if x < 0 else 1


