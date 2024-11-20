import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings("ignore")

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.wih = np.random.randn(input_size, hidden_size) * 0.01
        self.who = np.random.randn(hidden_size, output_size) * 0.01

    def forward(self, x):
        self.hidden_input = np.dot(x, self.wih)
        self.hidden_output = relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.who)
        self.final_output = relu(self.final_input)
        return self.final_output

    def backward(self, x, y, output, lr):
        output_error = y - output
        hidden_error = np.dot(output_error, self.who.T) * d_relu(self.hidden_input)

        self.who += lr * np.dot(self.hidden_output.T, output_error)
        self.wih += lr * np.dot(x.T, hidden_error)

    def train(self, x, y, lr=0.01, epochs=1000):
        for _ in range(epochs):
            print('.',end='')
            for xi, yi in zip(x, y):
                output = self.forward(xi)
                self.backward(xi, yi, output, lr)

    def predict(self, x):
        return self.forward(x)

# Load and preprocess the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0  # Normalize the data to [0, 1]
y = np.eye(10)[y.astype(int)]  # One-hot encode the labels

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the MLP
mlp = MLP(input_size=784, hidden_size=64, output_size=10)
mlp.train(X_train[:50], y_train[:50], lr=0.01, epochs=10)

# Predict and evaluate the model
y_pred = np.argmax(mlp.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy: {accuracy}")