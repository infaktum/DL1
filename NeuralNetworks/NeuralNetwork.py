import numpy as np

def get_scaler(min_scale = 0.,max_scale = 1.):
    return lambda record: (np.asfarray(record[1:]) / 255.0 * max_scale) + min_scale

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def relu(x):
    return np.max(0,x)


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, nodes, transfer=sigmoid):
        self.nodes = nodes
        self.weights = [np.array([np.random.normal(loc=0.0, scale=pow(nodes[k], -0.5),
                                                   size=(self.nodes[k], nodes[k + 1]))])
                        for k in range(len(nodes) - 1)]
        self.biases = np.zeros(len(self.nodes))
        self.transfer = transfer
        pass

    def feedforward(self, input):
        layer1 = sigmoid(np.dot(input, self.weights[0]) + self.biases[0])
        output = sigmoid(np.dot(layer1, self.weights[1]) + self.biases[1])
        return output, layer1

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def save(self, file):
        '''Speichert die Gewichte des Netzwerks'''
        with open(file + '.npy', 'wb') as f:
            np.save(f, self.weights, allow_pickle=True)
            np.save(f, self.biases, allow_pickle=True)

    def load(self, file):
        '''Lädt die Gewichte des Netzwerks'''
        with open(file + '.npy', 'rb') as f:
            self.weights = np.load(f)
            self.biases = np.load(f)

    def __str__(self):
        return "KNN (Geometrie = {0}, Transfer: {1})".format(self.nodes, self.transfer.__name__)


if __name__ == "__main__":
    nn = NeuralNetwork([4,2, 1], transfer=sigmoid)
    print(nn)
    print(nn.weights[0], nn.weights[1])
