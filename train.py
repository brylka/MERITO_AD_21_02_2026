import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

data = load_digits()
X, y = data.data, data.target
print("Wczytanie danych")

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
print("Model wytrenowany")

joblib.dump(model, 'model.joblib')
print("Model zapisany!")

