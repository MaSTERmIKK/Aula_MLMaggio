# Definiamo le distribuzioni da cui "pescare" casualmente i valori dei parametri
param_distributions = {
    'C': uniform(0.1, 10),        # uniform da 0.1 a 10
    'kernel': ['linear', 'rbf', 'poly'],  # valori categorici
    'gamma': uniform(0.0001, 0.1) # uniform tra 0.0001 e 0.1
}

# Istanziamo RandomizedSearchCV
# - n_iter=10 indica quante combinazioni casuali testare
# - cv=5 indica la 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,     
    cv=5,
    scoring='accuracy',
    random_state=42,  # per rendere riproducibili i risultati
    n_jobs=-1         # usa tutti i core disponibili per velocizzare
)

# Eseguiamo la ricerca
random_search.fit(X, y)

# Stampa dei migliori parametri e dello score
print("Migliori parametri trovati:", random_search.best_params_)
print("Migliore score (media in CV):", random_search.best_score_)
