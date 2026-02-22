import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
import numpy as np
np.set_printoptions(linewidth=200)

data = fetch_openml('mnist_784', version=1)
X, y = data.data, data.target
print("Wczytanie danych")

# print(y[0])
print(np.array(X.iloc[4]).reshape(28,28))





# model = DecisionTreeClassifier(random_state=42)
# model.fit(X, y)
# print("Model wytrenowany")
#
# joblib.dump(model, 'model.joblib')
# print("Model zapisany!")

