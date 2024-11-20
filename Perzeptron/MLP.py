import numpy as np





def get_scaler(min_scale=0., max_scale=1.):
    def scaler(record: np.ndarray) -> np.ndarray:
        """ Skaliert ein übergebenes NumPy Array auf einen vorgegebenen Bereich """
        min_val, max_val = np.min(record), np.max(record)
        return ((max_scale - min_scale) / (max_val - min_val) * record) + (min_scale - min)

    return scaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return x#np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)


class MLP:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.wih = np.random.randn(hidden_size, input_size) * 0.01
        self.who = np.random.randn(output_size, hidden_size) * 0.01
        #self.wih = np.zeros([hidden_size, input_size])
        #self.who = np.zeros([output_size, hidden_size])
        self.hidden_input = self.hidden_output = self.final_input = self.final_output = None
        pass


    def forward(self, x: np.ndarray):
        self.hidden_input = np.dot(self.wih,x)
        self.hidden_output = relu(self.hidden_input)

        self.final_input = np.dot(self.who,self.hidden_output )
        self.final_output = relu(self.final_input)
        #print(self.final_input,self.final_output)
        return self.final_output

    def backward(self, x: np.ndarray, y: np.ndarray, output, lr):
        output_error = y - output
        hidden_error = np.dot(output_error, self.who) * d_relu(self.hidden_input)

        self.who += lr * np.outer( output_error, self.hidden_output)
        self.wih += lr * np.dot( hidden_error, x)
        #self.who += lr * np.dot(output_error,self.hidden_output )
        #self.wih += lr * np.dot(x.T, hidden_error)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05,epochs=100) -> None:
        for _ in range(epochs):
            for x, t in zip(X, y):
                output = self.forward(x)
                self.backward(x, t, output, lr)

    def predict(self, inputs: np.ndarray) -> int:
        return np.argmax(self.forward(inputs))

    def __str__(self) -> str:
        return f'in -> hidden: {np.array2string(self.wih)}\nhidden -> out: {np.array2string(self.who)}'



if __name__ == "__main__":
    # XOR-Datensatz
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0,1],[0,1],[0,1],[1,0]])

    mlp = MLP(2, 2, 2)
    print(mlp)

    mlp.fit(X, y)
    print(mlp)

    for x, t in zip(X,y):
        y = mlp.predict(x)
        print(f"Input: {x}, target: {y}, Predicted: {y}")


#    X = np.array([[x, y] for x in [0, 1] for y in [0, 1]])
#    Y = np.array([[1-y,y] for y in [1, 1, 0, 0]])
#    mlp = MLP(2,2,2)
#    mlp.train(X, Y)
#    for x in X:
#        result = mlp.forward(x)
#        print(f'input: {x} ,output: {result} -> {np.argmax(result)}')
