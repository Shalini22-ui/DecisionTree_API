from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib
x,y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(x, y)
joblib.dump(model, "tree_model.pkl")
print("model saved successfully")