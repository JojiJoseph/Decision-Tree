from cart import CART
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X_data, y_data = load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

# print(X_test.shape)

cart = CART()
cart.fit(X_train, y_train)
print(np.mean(cart.predict(X_test).astype(int) == y_test))

clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(X_train, y_train)
print(np.mean(clf.predict(X_test).astype(int) == y_test))

cart = CART()
cart.fit(X_train, y_train)
print(np.mean(cart.predict(X_train).astype(int) == y_train))

clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(X_train, y_train)
print(np.mean(clf.predict(X_train).astype(int) == y_train))

"Regression Test"

X_data = np.linspace(0,1,100).reshape((-1,1))
y_data = np.sin(2*np.pi*X_data).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

cart_reg = CART(impurity_function="mse")

cart_reg.fit(X_train, y_train)

# X_test

y_pred = cart_reg.predict(X_test)

import matplotlib.pyplot as plt

plt.plot(X_data.flatten(), y_data)
plt.scatter(X_test.flatten(), y_pred)
plt.plot()
plt.show()

def test_impurity():
    cart = CART()
    assert cart.calc_impurity([1,1,1,1]) == 0
    assert cart.calc_impurity([1,2,1,2]) == 0.5