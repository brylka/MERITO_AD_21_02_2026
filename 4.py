from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42)
random_forest.fit(X_train, y_train)

tree_train_acc = single_tree.score(X_train, y_train)
tree_test_acc = single_tree.score(X_test, y_test)

print("POJEDYNCZE DRZEWO")
print(f"Dokładność na zbiorze treningowym: {tree_train_acc}")
print(f"Dokładność na zbiorze testowym:    {tree_test_acc}")
print(f"Różnica:                           {tree_train_acc - tree_test_acc}")

tree_cv_scores = cross_val_score(single_tree, X, y, cv=10)
print(f"Walidacja krzyżowa:")
print(f"Średnia dokładność: {tree_cv_scores.mean()}")
print(f"Odchylenie std:     {tree_cv_scores.std()}")

rf_train_acc = random_forest.score(X_train, y_train)
rf_test_acc = random_forest.score(X_test, y_test)

print("LAS DRZEW:")
print(f"Dokładność na zbiorze treningowym: {rf_train_acc}")
print(f"Dokładność na zbiorze testowym:    {rf_test_acc}")
print(f"Różnica:                           {rf_train_acc - rf_test_acc}")

rf_cv_scores = cross_val_score(random_forest, X, y, cv=10)
print(f"Walidacja krzyżowa:")
print(f"Średnia dokładność: {rf_cv_scores.mean()}")
print(f"Odchylenie std:     {rf_cv_scores.std()}")