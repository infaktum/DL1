
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


digits = load_digits()
X, y, images = digits.data, digits.target, digits.images
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X,y)

y_pred = perceptron.predict(X_test)

# Calculate the accuracy score
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score:.2f}')

