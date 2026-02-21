from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_digits
import time

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    'POJEDYNCZE DRZEWO': DecisionTreeClassifier(random_state=42),
    'LAS DRZEW (100)': RandomForestClassifier(n_estimators=100, random_state=42),
    'LAS DRZEW (200)': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBOOST (100)': xgb.XGBClassifier(n_estimators=100, random_state=42),
    'XGBOOST (200)': xgb.XGBClassifier(n_estimators=200, random_state=42),
    'LIGHTGBM (100)': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'LIGHTGBM (200)': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
}
#print(f"{'model':<25} {'D.treningowa':<15} {'D.testowa':<15} {'Średnia CV':<15} {'std':<15} {'czas':<10}")
text = f"{'model':<25} {'D.treningowa':<15} {'D.testowa':<15} {'Średnia CV':<15} {'std':<15} {'czas':<10}\n"
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    stop = time.time()

    tree_train_acc = model.score(X_train, y_train)
    tree_test_acc = model.score(X_test, y_test)
    tree_cv_scores = cross_val_score(model, X, y, cv=10)

    # print(f"{name:<25} {tree_train_acc:<15.2f} {tree_test_acc:<15.2f} {tree_cv_scores.mean():<15.2f} {tree_cv_scores.std():<15.2f} {stop - start:<15.2f}")
    text += f"{name:<25} {tree_train_acc:<15.2f} {tree_test_acc:<15.2f} {tree_cv_scores.mean():<15.2f} {tree_cv_scores.std():<15.2f} {stop - start:<15.2f}\n"

print(text)