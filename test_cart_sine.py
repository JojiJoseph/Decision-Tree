from cart import CART, get_explanation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import export_text

X_data = np.linspace(0,1,10).reshape((-1,1))
y_data = np.sin(2*np.pi*X_data).flatten()

cart_reg = CART(impurity_function="mse", max_depth=3)
cart_reg.fit(X_data, y_data)

X_test = np.linspace(0,1,1000).reshape((-1,1))
y_test = np.sin(2*np.pi*X_test).flatten()
y_pred1 = cart_reg.predict(X_test)

cart_reg = CART(impurity_function="mse", max_depth=10)

cart_reg.fit(X_data, y_data)

X_test = np.linspace(0,1,1000).reshape((-1,1))
y_test = np.sin(2*np.pi*X_test).flatten()
y_pred2 = cart_reg.predict(X_test)

plt.figure("Custom CART")
plt.plot(X_test.flatten(), y_test)
plt.plot(X_test.flatten(), y_pred1, label="max depth=3")
plt.plot(X_test.flatten(), y_pred2, label="max depth=10")
plt.legend()
plt.show(block=False)

reg = DecisionTreeRegressor(max_depth=3)
reg.fit(X_data, y_data)



X_test = np.linspace(0,1,1000).reshape((-1,1))
y_test = np.sin(2*np.pi*X_test).flatten()
y_pred1 = reg.predict(X_test)

reg = DecisionTreeRegressor(max_depth=10)
reg.fit(X_data, y_data)
y_pred2 = reg.predict(X_test)
plt.figure("Sklearn CART")
plt.plot(X_test.flatten(), y_test)
plt.plot(X_test.flatten(), y_pred1, label="max depth=3")
plt.plot(X_test.flatten(), y_pred2, label="max depth=10")
plt.legend()
plt.show()

print("\nExplantion by sklearn")
print(export_text(reg))

print("\nExplantion by custom CART")
print(get_explanation(cart_reg))

print(get_explanation(cart_reg) == export_text(reg))