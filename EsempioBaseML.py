from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Caricamento dataset
iris = load_iris()

# Creazione modello
model = DecisionTreeClassifier()

# Addestramento
model.fit(iris.data, iris.target)

print("Modello addestrato correttamente")
