from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.datasets import load_digits
import pandas as pd

data = load_digits()
X, y = pd.DataFrame(data.data), data.target

param_grid = {
    "n_estimators": [10, 20, 50, 100, 200, 300, 400, 500, 1000],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
}

model = xgb.XGBClassifier(random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', verbose=1)
grid_search.fit(X, y)

print(f"Najlepsze parametry:  {grid_search.best_params_}")
print(f"Najlepsza dokładność: {grid_search.best_score_}")

