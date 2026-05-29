# ==========================
# DATASET WINE - SCIKIT LEARN
# ==========================

# Importazione librerie
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 1. CARICAMENTO DATASET
wine = load_wine()

# Caratteristiche e classi
X = wine.data
y = wine.target

# Creo un DataFrame per leggere meglio i dati
df = pd.DataFrame(X, columns=wine.feature_names)

print("Prime righe del dataset:")
print(df.head())

# 2. ESPLORAZIONE DEL DATASET

# Numero di campioni per classe
print("\nCampioni per classe:")
print(pd.Series(y).value_counts())

# Statistiche di base
print("\nStatistiche:")
print(df.describe())

# Grafico distribuzione classi
plt.figure(figsize=(6,4))
pd.Series(y).value_counts().sort_index().plot(kind="bar")
plt.title("Distribuzione classi")
plt.xlabel("Classe")
plt.ylabel("Numero campioni")
plt.show()

# Standardizzazione dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. RIDUZIONE DIMENSIONALITA' CON PCA

# Riduco a 2 componenti
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Scatter plot
plt.figure(figsize=(8,6))

for classe in range(3):
    plt.scatter(
        X_pca[y == classe, 0],
        X_pca[y == classe, 1],
        label=f"Classe {classe}"
    )

plt.title("PCA - Dataset Wine")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.show()

# 4. TRAIN E TEST SET

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# 5. RANDOM FOREST

modello = RandomForestClassifier(random_state=42)

# Addestramento
modello.fit(X_train, y_train)

# Previsioni
y_pred = modello.predict(X_test)

# 6. VALUTAZIONE MODELLO

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. IMPORTANZA DELLE FEATURE

importanze = modello.feature_importances_

# Creo DataFrame ordinato
feature_importance = pd.DataFrame({
    "Feature": wine.feature_names,
    "Importanza": importanze
})

feature_importance = feature_importance.sort_values(
    by="Importanza",
    ascending=False
)

print("\nImportanza feature:")
print(feature_importance)

# Grafico
plt.figure(figsize=(10,5))
plt.bar(feature_importance["Feature"],
        feature_importance["Importanza"])
plt.xticks(rotation=90)
plt.title("Importanza delle feature")
plt.show()

# 8. MATRICE DI CONFUSIONE

matrice = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(matrice)

plt.title("Matrice di Confusione")
plt.colorbar()

plt.xlabel("Classe Predetta")
plt.ylabel("Classe Reale")

plt.show()

print("\nMatrice di Confusione:")
print(matrice)

# 9. OTTIMIZZAZIONE CON GRID SEARCH

parametri = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    parametri,
    cv=5
)

grid.fit(X_train, y_train)

print("\nMigliori parametri trovati:")
print(grid.best_params_)

print("\nMiglior punteggio:")
print(grid.best_score_)
