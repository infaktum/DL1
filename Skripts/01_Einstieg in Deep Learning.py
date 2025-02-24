
from sklearn.datasets import load_digits

digits = load_digits()
X, y, images = digits.data, digits.target, digits.images

from dl1 import gallery

print(f"Anzahl Datensätze : {len(X)}")
rows, cols = 4, 8
gallery(images,rows, cols,cmap='Blues') 
print(f"y: {y[:rows * cols]}")

from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier( max_iter=1000)


sep = 1400
X_train, X_test, y_train, y_test = X[:sep], X[sep:], y[:sep], y[sep:]

mlpc.fit(X_train,y_train)

n = 20
print(f'Erwartet: \t{y_test[:n]}\nTatsächlich:  \t{mlpc.predict(X_test[:n])}')


print(f'Score: {mlpc.score(X_test,y_test) :0.1%} ')




idx = mlpc.predict(X) != y
print(y[idx])
gallery(images[idx],5,8,cmap='Reds')  





