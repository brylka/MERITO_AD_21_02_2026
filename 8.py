from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_digits
import time
import pandas as pd

data = load_digits()
X, y = pd.DataFrame(data.data), data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

n_es = [10, 20, 50, 100, 200, 300, 400, 500, 1000]

print(f"{'ilość drzew':<25} {'D.treningowa':<15} {'D.testowa':<15} {'Średnia CV':<15} {'std':<15} {'czas':<10}")

for n_e in n_es:
    model = xgb.XGBClassifier(n_estimators=n_e, learning_rate=0.1, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    tree_cv_scores = cross_val_score(model, X, y, cv=10)
    stop = time.time()

    tree_train_acc = model.score(X_train, y_train)
    tree_test_acc = model.score(X_test, y_test)

    print(f"{n_e:<25} {tree_train_acc:<15.2f} {tree_test_acc:<15.2f} {tree_cv_scores.mean():<15.2f} {tree_cv_scores.std():<15.2f} {stop - start:<15.2f}")
