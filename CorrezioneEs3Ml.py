# Importo i moduli necessari
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Carico il dataset Wine
wine = load_wine()

# X contiene le caratteristiche dei vini
X = wine.data

# y contiene la classe corretta di ogni vino
y = wine.target

# 2. Standardizzo le caratteristiche
scaler = StandardScaler()

# Trasformo i dati portandoli su una scala comune
X_standardizzato = scaler.fit_transform(X)

# 3. Divido i dati in training set e test set
X_train, X_test, y_train, y_test = train_test_split(
    X_standardizzato,
    y,
    test_size=0.30,
    random_state=42
)

# 4. Creo e addestro il modello Decision Tree
modello = DecisionTreeClassifier(random_state=42)

# Il modello impara dai dati di training
modello.fit(X_train, y_train)

# 5. Uso il modello per fare previsioni sui dati di test
y_pred = modello.predict(X_test)

# Stampo le metriche di valutazione
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 6. Creo la matrice di confusione
matrice = confusion_matrix(y_test, y_pred)

# Visualizzo la matrice di confusione
display = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=wine.target_names
)

display.plot()
plt.title("Matrice di confusione - Dataset Wine")
plt.show()
