from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

xgb_classifier = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.3,
    random_state=42)
xgb_classifier.fit(X_train, y_train)

train_acc = xgb_classifier.score(X_train, y_train)
test_acc = xgb_classifier.score(X_test, y_test)

print(f"Dokładność na zbiorze treningowym: {train_acc}")
print(f"Dokładność na zbiorze testowym:    {test_acc}")
print(f"Różnica:                           {train_acc - test_acc}")

cv_scores = cross_val_score(xgb_classifier, X, y, cv=10)
print(f"Walidacja krzyżowa:")
print(f"Średnia dokładność: {cv_scores.mean()}")
print(f"Odchylenie std:     {cv_scores.std()}")
