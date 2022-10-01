from cart import CART, get_explanation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X_data, y_data = load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=1)

print("Testing custom CART")
cart = CART(impurity_function="gini")
cart.fit(X_train, y_train)
print("Training accuracy:", np.mean(cart.predict(X_train).astype(int) == y_train))
print("Evaluation accuracy:", np.mean(cart.predict(X_test).astype(int) == y_test))

print("\nTesting sklearn Decision Tree")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Training accuracy:", np.mean(clf.predict(X_train).astype(int) == y_train))
print("Evaluation accuracy:", np.mean(clf.predict(X_test).astype(int) == y_test))


def test_impurity():
    cart = CART()
    assert cart.calc_impurity([1,1,1,1]) == 0
    assert cart.calc_impurity([1,2,1,2]) == 0.5