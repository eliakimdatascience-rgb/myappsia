import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pickle

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Importation des 12 algorithmes du cours
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    BaggingRegressor, 
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor
)
# ==============================================================================
# 1. CHARGEMENT DES BIBLIOTHÈQUES ET DES DONNÉES
# ==============================================================================

# Chargement adapté à la structure réelle du fichier (séparateur ',' et décimales '.')
# Chargement avec le chemin absolu vers ton dossier de données
df = pd.read_csv(r"E:\COURS INSSEDS\data set\Premier_League_fr.csv")

print("=== STRUCTURE DU JEU DE DONNÉES ===")
print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
print(df.dtypes)

# ==============================================================================
# 2. PRÉTRAITEMENT DES DONNÉES
# ==============================================================================

# --- Traitement des Doublons ---
nb_doublons = df.duplicated().sum()
print(f"\nNombre de doublons détectés : {nb_doublons}")
df = df.drop_duplicates()

# --- Sélection intelligente des variables numériques pour les graphiques ---
# On exclut les variables de classement (Rank) et les niveaux de ligue pour le boxplot
colonnes_num = df.select_dtypes(include=[np.number]).columns.tolist()
colonnes_a_tracer = [col for col in colonnes_num if not col.startswith('Rang_') and col != 'Niveau_Championnat']

# Calcul dynamique de la grille de subplots (4 colonnes, lignes calculées selon le besoin)
n_cols = 4
n_rows = (len(colonnes_a_tracer) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
axes = axes.flatten()

# Palette de couleurs variées
colors = sns.color_palette("husl", len(colonnes_a_tracer))

for i, col in enumerate(colonnes_a_tracer):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=colors[i], color="black"),
                    medianprops=dict(color="darkred", linewidth=1.5))
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].tick_params(labelsize=8)

# Supprimer les axes vides en trop
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Boxplots des variables clés - Avant traitement des outliers", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# --- Traitement sélectif des Outliers par Winsorisation ---
def winsorize_iqr(series, threshold=1.5):
    """Winsorisation basée sur la règle de Tukey (IQR x threshold)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return series.clip(lower=lower, upper=upper)

# On applique uniquement sur les vraies métriques de performance continue (ex: Cartons, Minutes)
# ATTENTION : Ne pas écraser les buts des meilleurs attaquants ! On peut choisir de ne winsoriser que la discipline.
colonnes_a_winsoriser = ['Cartons_Jaunes', 'Cartons_Rouges', 'Score_Discipline']
df[colonnes_a_winsoriser] = df[colonnes_a_winsoriser].apply(lambda col: winsorize_iqr(col, threshold=1.5))
print(f"\nWinsorisation appliquée avec succès sur : {colonnes_a_winsoriser}")

# --- Sélection intelligente des variables numériques pour les graphiques ---
# On exclut les variables de classement (Rank) et les niveaux de ligue pour le boxplot
colonnes_num = df.select_dtypes(include=[np.number]).columns.tolist()
colonnes_a_tracer = [col for col in colonnes_num if not col.startswith('Rang_') and col != 'Niveau_Championnat']

# Calcul dynamique de la grille de subplots (4 colonnes, lignes calculées selon le besoin)
n_cols = 4
n_rows = (len(colonnes_a_tracer) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
axes = axes.flatten()

# Palette de couleurs variées
colors = sns.color_palette("husl", len(colonnes_a_tracer))

for i, col in enumerate(colonnes_a_tracer):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=colors[i], color="black"),
                    medianprops=dict(color="darkred", linewidth=1.5))
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].tick_params(labelsize=8)

# Supprimer les axes vides en trop
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Boxplots des variables clés -  apres traitement des outliers", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# --- Analyse des données manquantes ---
print(f"\nNombre de lignes incomplètes : {df.isnull().any(axis=1).sum()}")
print(f"Proportion de lignes incomplètes : {df.isnull().any(axis=1).mean():.4f}")

# Visualisation des valeurs manquantes (si le dataset est complet, la matrice sera pleine)
fig, ax = plt.subplots(figsize=(10, 5))
msno.matrix(df, ax=ax, sparkline=False, fontsize=9)
plt.title("Carte de complétude des données", fontsize=12, fontweight='bold')
plt.show()

# 3. Analyse descriptive de la variable cible et répartition des classes
print("Statistiques descriptives de la Note d'Efficacité :")
print(df['Note_Efficacite'].describe())

print("\nRépartition de la Catégorie de Performance :")
print(df['Categorie_Performance'].value_counts())

# 4. Matrice de corrélation entre les métriques "Par 90" et l'Efficiency Rating
metrics_90 = ['Buts_Par_90', 'Passes_Decisives_Par_90', 'Contributions_But_Par_90', 'Note_Efficacite']
corr_matrix = df[metrics_90].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation (Métriques par 90 min vs Note d'Efficacité)")
plt.show()

# Note : On remarque une colinéarité parfaite ou très forte entre "Contributions_But_Par_90" 
# et la somme (Buts_Par_90 + Passes_Decisives_Par_90). Il faudra en exclure une pour éviter la multicolinéarité !

# ------------------------------------------------------------------------------
# 3. PRÉPARATION DE LA MATRICE DES PRÉDICTEURS (X) ET DE LA CIBLE (Y)
# ------------------------------------------------------------------------------
# Définition de la variable cible
Y = df['Note_Efficacite'].values

# Exclusion des métriques colinéaires, qualitatives cibles ou identifiants
colonnes_a_exclure = [
    'Note_Efficacite', 'Categorie_Performance', 'Joueur','Equipe' ,
    'Contributions_But_Par_90', 'Contributions_Offensives', 
    'Date_Collecte_Donnees', 'Saison','Competition', 'Pays', 'Nom_Championnat'
]

# Conserver uniquement les colonnes réellement existantes
cols_to_drop = [c for c in colonnes_a_exclure if c in df.columns]
X_df = df.drop(columns=cols_to_drop)

# Dummy / One-Hot Encoding pour les variables catégorielles (Postes, Clubs, etc.)
X_encoded = pd.get_dummies(X_df, drop_first=True)
X = X_encoded.values

# Définition de la Validation Croisée K-Fold (K=10)
kf = KFold(n_splits=10, shuffle=True, random_state=7)

# ------------------------------------------------------------------------------
# 3. LISTE DES 12 MODÈLES DE RÉGRESSION
# ------------------------------------------------------------------------------
def get_models():
    models = []
    models.append(('RESEAU NET', MLPRegressor(max_iter=1000, random_state=42)))
    models.append(('RLM', LinearRegression()))
    models.append(('RIDGE', Ridge()))
    models.append(('LASSO', Lasso()))
    models.append(('ELASTICNET', ElasticNet()))
    models.append(('KKPV', KNeighborsRegressor()))
    models.append(('ARBRE', DecisionTreeRegressor(random_state=42)))
    models.append(('SVM', SVR()))
    models.append(('BAGGING', BaggingRegressor(random_state=42)))
    models.append(('ADABOOST', AdaBoostRegressor(random_state=42)))
    models.append(('EXTRA_TREE', ExtraTreesRegressor(n_estimators=100, random_state=42)))
    models.append(('BOOSTING', GradientBoostingRegressor(n_estimators=100, random_state=42)))
    return models

# ------------------------------------------------------------------------------
# 4. COMPARAISON AVANT STANDARDISATION
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("--- ÉVALUATION AVANT STANDARDISATION (RMSE) ---")
print("="*50)

models = get_models()
results_raw = []
names_raw = []

for name, model in models:
    # Calcul du RMSE à partir de neg_mean_squared_error
    rmse_scores = np.sqrt(-cross_val_score(model, X, Y, cv=kf, scoring='neg_mean_squared_error'))
    results_raw.append(rmse_scores)
    names_raw.append(name)
    print(f"{name:12s}: {rmse_scores.mean():.6f} ({rmse_scores.std():.6f})")

# Visualisation graphique (Boxplot) avant standardisation
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Comparaison des Algorithmes - Données Brutes (Avant Standardisation)")
ax = fig.add_subplot(111)
ax.boxplot(results_raw, vert=False)
ax.set_yticklabels(names_raw)
plt.xlabel("RMSE")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 5. COMPARAISON APRÈS STANDARDISATION (TRANSFORMATION DES DONNÉES)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("--- ÉVALUATION APRÈS STANDARDISATION (RMSE) ---")
print("="*50)

# Application de la standardisation
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

models = get_models()
results_scaled = []
names_scaled = []

for name, model in models:
    rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, Y, cv=kf, scoring='neg_mean_squared_error'))
    results_scaled.append(rmse_scores)
    names_scaled.append(name)
    print(f"{name:12s}: {rmse_scores.mean():.6f} ({rmse_scores.std():.6f})")

# Visualisation graphique (Boxplot) après standardisation
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Comparaison des Algorithmes - Données Standardisées")
ax = fig.add_subplot(111)
ax.boxplot(results_scaled, vert=False)
ax.set_yticklabels(names_scaled)
plt.xlabel("RMSE")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 6. ENTRAÎNEMENT DU MEILLEUR MODÈLE ET PRÉDICTIONS FINALES
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("--- PRÉDICTION FINALE AVEC LE MEILLEUR MODÈLE (ExtraTrees) ---")
print("="*50)

best_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
best_model.fit(X, Y)

# Prédictions sur l'ensemble du jeu de données
predictions = best_model.predict(X)
df['predictions_extra_trees'] = predictions

# Calcul des erreurs de prédiction
errors = Y - predictions
df['prediction_errors'] = np.round(errors, 2)

# Affichage des 10 premiers résultats avec les vrais noms de colonnes
cols_affichage = ['Joueur', 'Equipe', 'Note_Efficacite', 'predictions_extra_trees', 'prediction_errors']
print(df[cols_affichage].head(10))


import pickle
# ------------------------------------------------------------------------------
# 7. ENREGISTREMENT ET CHARGEMENT DU MODÈLE (.pkl)
# ------------------------------------------------------------------------------
model_filename = 'extra_trees_regressor_premier_league.pkl'

# Sauvegarde sur disque
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"\nModèle enregistré avec succès sous '{model_filename}'.")

# Rechargement du modèle pour vérification
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Test de prédiction sur le premier individu
first_individual = X[0].reshape(1, -1)
pred_first = loaded_model.predict(first_individual)

print(f"Prédiction pour {df['Joueur'].iloc[0]} ({df['Equipe'].iloc[0]}) : {pred_first[0]:.2f}")
print(f"Valeur réelle : {Y[0]:.2f}")

# ==============================================================================
# Modélisation par Deep Learning / Perceptron Multicouche (MLP)
# Variable Cible (Y) : Note_Efficacite
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------------------
# 0. CHARGEMENT ET PRÉPARATION DES DONNÉES PREMIER LEAGUE
# ------------------------------------------------------------------------------


# Définition de la cible Y
Y = df['Note_Efficacite'].values

# Exclusion des métriques colinéaires, qualitatives cibles ou identifiants
colonnes_a_exclure = [
    'Note_Efficacite', 'Categorie_Performance', 'Joueur', 'Equipe',
    'Contributions_But_Par_90', 'Contributions_But', 
    'Date_Collecte_Donnees', 'Saison', 'Competition', 'Pays', 'Nom_Championnat',
    'predictions_extra_trees', 'prediction_errors'
]

cols_to_drop = [c for c in colonnes_a_exclure if c in df.columns]
X_df = df.drop(columns=cols_to_drop)

# Encodage One-Hot des variables catégorielles
X_encoded = pd.get_dummies(X_df, drop_first=True)
X = X_encoded.values.astype(np.float32)

input_dimension = X.shape[1]
print(f"Nombre total de prédicteurs (input_dim) après encodage : {input_dimension}")

# Separation Apprentissage / Test
X_app, X_test, Y_app, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# ------------------------------------------------------------------------------
# 1. PERCEPTRON SIMPLE (SANS COUCHE CACHÉE)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("1. PERCEPTRON SIMPLE (0 Couche Cachée)")
print("="*50)

# Un perceptron simple correspond à hidden_layer_sizes=()
model_simple = MLPRegressor(
    hidden_layer_sizes=(), 
    activation='relu', 
    solver='adam', 
    max_iter=150, 
    random_state=42
)
model_simple.fit(X_app, Y_app)

mse_simple_app = mean_squared_error(Y_app, model_simple.predict(X_app))
mse_simple_test = mean_squared_error(Y_test, model_simple.predict(X_test))

print(f"MSE Apprentissage : {mse_simple_app:.4f}")
print(f"MSE Test          : {mse_simple_test:.4f}")

# ------------------------------------------------------------------------------
# 2. PERCEPTRON MULTICOUCHE (MLP) & COURBE DE PERTE
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("2. PERCEPTRON MULTICOUCHE & COURBE D'APPRENTISSAGE")
print("="*50)

model_mlp = MLPRegressor(
    hidden_layer_sizes=(12, 12, 10, 10), 
    activation='relu', 
    solver='adam', 
    max_iter=100, 
    random_state=42
)
model_mlp.fit(X_app, Y_app)

# Affichage de la courbe de perte
plt.figure(figsize=(8, 5))
plt.plot(model_mlp.loss_curve_, label='Apprentissage (MSE)')
plt.title('Évolution de la Perte au fil des Itérations')
plt.ylabel('Perte')
plt.xlabel('Itération')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
# 3. MLP AVEC STANDARDISATION (ESSENTIEL POUR LES RÉSEAUX DE NEURONES)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("3. MLP AVEC DONNÉES STANDARDISÉES (StandardScaler)")
print("="*50)

scaler = StandardScaler().fit(X_app)
rescaledX_app = scaler.transform(X_app)
rescaledX_test = scaler.transform(X_test)

model_scaled = MLPRegressor(
    hidden_layer_sizes=(12, 12, 10, 10), 
    activation='relu', 
    solver='adam', 
    max_iter=500,           # Augmenté pour éviter le ConvergenceWarning
    early_stopping=True,    # Arrêt précoce dès que val_loss ne baisse plus
    n_iter_no_change=15,
    random_state=42
)
model_scaled.fit(rescaledX_app, Y_app)

mse_scaled_app = mean_squared_error(Y_app, model_scaled.predict(rescaledX_app))
mse_scaled_test = mean_squared_error(Y_test, model_scaled.predict(rescaledX_test))

print(f"MSE Standardisé Apprentissage : {mse_scaled_app:.4f}")
print(f"MSE Standardisé Test          : {mse_scaled_test:.4f}")

# ------------------------------------------------------------------------------
# 4. PRÉDICTIONS FINALES SUR LE DATASET COMPLET
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("4. PRÉDICTIONS FINALES SUR LE DATASET")
print("="*50)

scaler_full = StandardScaler().fit(X)
X_scaled_full = scaler_full.transform(X)

predictions = model_scaled.predict(X_scaled_full)

df['predictions_deep_learning'] = np.round(predictions, 2)
df['erreur_deep_learning'] = np.round(Y - predictions, 2)

cols_preview = ['Joueur', 'Equipe', 'Note_Efficacite', 'predictions_deep_learning', 'erreur_deep_learning']
print(df[cols_preview].head(10))

#-------------------------------------------------------------------------#
# 0. CHARGEMENT ET PRÉPARATION DES DONNÉES PREMIER LEAGUE
#-------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix



# Définition de la variable Cible Classification (Binaire)
# Si Categorie_Performance est déjà sous forme textuelle, on l'encode en 0 / 1
if 'Categorie_Performance' in df.columns:
    # Exemple : Encodage binaire (Adaptez la condition selon vos classes exactes)
    # Ex: 1 si Top Performer / Élite, 0 sinon
    Y = np.where(df['Categorie_Performance'].astype(str).str.contains('Haute|Élite|Top|1', case=False), 1, 0)
else:
    # Alternative : Seuil sur la Note_Efficacite pour créer une classe binaire
    Y = np.where(df['Note_Efficacite'] >= df['Note_Efficacite'].median(), 1, 0)

# Exclusion des variables explicatives colinéaires, id, ou métriques cibles
colonnes_a_exclure = [
    'Note_Efficacite', 'Categorie_Performance', 'Joueur', 'Equipe',
    'Contributions_But_Par_90', 'Contributions_But', 
    'Date_Collecte_Donnees', 'Saison', 'Competition', 'Pays', 'Nom_Championnat',
    'predictions_extra_trees', 'prediction_errors', 'predictions_deep_learning', 'erreur_deep_learning'
]

cols_to_drop = [c for c in colonnes_a_exclure if c in df.columns]
X_df = df.drop(columns=cols_to_drop)

# Encodage One-Hot des variables catégorielles restantes
X_encoded = pd.get_dummies(X_df, drop_first=True)
X = X_encoded.values.astype(np.float32)

# Standardisation des variables
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Définition de la validation croisée
kf = KFold(n_splits=10, shuffle=True, random_state=7)

print(f"Nombre total d'observations : {X.shape[0]}")
print(f"Nombre de prédicteurs (input_dim) : {X.shape[1]}")

#-------------------------------------------------------------------------#
# 1. COMPARAISON TOUS MODÈLES (SUR DONNÉES STANDARDISÉES)
#-------------------------------------------------------------------------#
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier, 
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier
)

models = []
models.append(('LOGIT', LogisticRegression(max_iter=1000)))
models.append(('RIDGE', RidgeClassifier()))
models.append(('KKPV', KNeighborsClassifier()))
models.append(('ARBRE', DecisionTreeClassifier(random_state=7)))
models.append(('SVM', SVC()))
models.append(('BAYES', GaussianNB()))
models.append(('RESEAU NET', MLPClassifier(max_iter=500, random_state=7)))
models.append(('BAGGING', BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=7)))
models.append(('RANDOM FOREST', RandomForestClassifier(n_estimators=100, random_state=7)))
models.append(('EXTRA_TREE', ExtraTreesClassifier(n_estimators=100, random_state=7)))
models.append(('ADABOOST', AdaBoostClassifier(n_estimators=30, random_state=7)))
models.append(('BOOSTING', GradientBoostingClassifier(n_estimators=100, random_state=7)))

# Évaluation de chaque modèle en Validation Croisée
results = []
names = []

print("\n" + "="*50)
print("PERFORMANCES DES MODÈLES (ACCURACY MOYENNE)")
print("="*50)

for name, model in models:
    cv_results = cross_val_score(model, X_scaled, Y, cv=kf, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Graphique Boxplot Comparatif
fig = plt.figure(figsize=(10, 6))
fig.suptitle('Comparaison des Algorithmes de Classification (Premier League)')
ax = fig.add_subplot(111)
ax.boxplot(results, vert=False)
ax.set_yticklabels(names)
plt.grid(True)
plt.show()

#-------------------------------------------------------------------------#
# 2. PRÉDICTION FINALE & ÉVALUATION (AVEC LE MEILLEUR MODÈLE - EX: RANDOM FOREST)
#-------------------------------------------------------------------------#
print("\n" + "="*50)
print("ENTRAÎNEMENT FINAL & MATRICE DE CONFUSION")
print("="*50)

best_model = RandomForestClassifier(n_estimators=100, random_state=7)
best_model.fit(X_scaled, Y)

predictions = best_model.predict(X_scaled)

# Ajout des prédictions et erreurs au DataFrame
df['prediction_class'] = predictions
df['erreur_classification'] = Y - predictions

# Métriques globales
accuracy = accuracy_score(Y, predictions)
auc = roc_auc_score(Y, predictions)
tn, fp, fn, tp = confusion_matrix(Y, predictions).ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Accuracy    : {accuracy:.4f} (% de bonnes prédictions global)")
print(f"AUC         : {auc:.4f} (Pouvoir discriminant)")
print(f"Spécificité : {specificity:.4f} (% de vrais négatifs bien classés)")
print(f"Sensibilité : {sensitivity:.4f} (% de vrais positifs bien classés)")

# Aperçu final des résultats par joueur
cols_preview = ['Joueur', 'Equipe', 'prediction_class', 'erreur_classification']
cols_available = [c for c in cols_preview if c in df.columns]
print("\n", df[cols_available].head(10))

# ==============================================================================
# DEEP LEARNING & MACHINE LEARNING (PREMIER LEAGUE 2023/24)
# APPLIQUÉ À UN PROBLÈME DE CLASSIFICATION DU NIVEAU DES JOUEURS
# ==============================================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ------------------------------------------------------------------------------

# Création d'une variable binaire : 1 si Joueur Top Performance (World Class / Elite), 0 sinon
top_categories = ['Classe mondiale (25+)', 'Elite (15-24)']
df['Top_Performance'] = df['Categorie_Performance'].apply(
    lambda x: 1 if x in top_categories else 0
)

# Sélection des features statistiques (explicatives)
features_cols = [
    'Apparitions',
    'Minutes',
    'Buts',
    'Passes_Decisives',
    'Buts_Par_90',
    'Passes_Decisives_Par_90',
    'Contributions_But_Par_90',
    'Score_Discipline',
]

X = df[features_cols].values
Y = df['Top_Performance'].values

print(f'Taille de la matrice X : {X.shape}')
print(
    f'Distribution de la variable cible : {np.bincount(Y)} (0: Ordinaire, 1: Top Player)'
)


# ------------------------------------------------------------------------------
# 1. PERCEPTRON SIMPLE POUR LA PRÉDICTION DE PERFORMANCE
# ------------------------------------------------------------------------------
print('\n--- 1. PERCEPTRON SIMPLE ---')

# Standardisation des variables d'entrée
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation Apprentissage / Test
X_app, X_test, Y_app, Y_test = train_test_split(
    X_scaled, Y, train_size=0.7, random_state=42, stratify=Y
)

# Créer un perceptron simple (pas de couche cachée)
model_ps = Sequential()
model_ps.add(Dense(1, input_dim=8, activation='sigmoid'))

# Compiler
model_ps.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
)

# Entraîner
model_ps.fit(
    X_app, Y_app, epochs=150, validation_data=(X_test, Y_test), batch_size=5, verbose=0
)

# Évaluer
scores_app = model_ps.evaluate(X_app, Y_app, verbose=0)
scores_test = model_ps.evaluate(X_test, Y_test, verbose=0)
print(
    "Exactitude Perceptron Simple (Train) : %.2f%%" % (scores_app[1] * 100)
)
print("Exactitude Perceptron Simple (Test)  : %.2f%%" % (scores_test[1] * 100))


# ------------------------------------------------------------------------------
# 2. PERCEPTRON MULTICOUCHE (MLP - 1 COUCHE CACHÉE) + SUIVI GRAPHIQUE
# ------------------------------------------------------------------------------
print('\n--- 2. MLP (1 COUCHE CACHÉE) ---')

X_app, X_test, Y_app, Y_test = train_test_split(
    X, Y, train_size=0.7, random_state=42, stratify=Y
)

# Standardisation séparée pour éviter le Data Leakage
scaler = StandardScaler()
X_app_scaled = scaler.fit_transform(X_app)
X_test_scaled = scaler.transform(X_test)

model_mlp1 = Sequential()
model_mlp1.add(
    Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu')
)
model_mlp1.add(Dense(1, activation='sigmoid'))

model_mlp1.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
)

history_mlp1 = model_mlp1.fit(
    X_app_scaled,
    Y_app,
    epochs=100,
    validation_data=(X_test_scaled, Y_test),
    batch_size=5,
    verbose=0,
)

scores_app = model_mlp1.evaluate(X_app_scaled, Y_app, verbose=0)
scores_test = model_mlp1.evaluate(X_test_scaled, Y_test, verbose=0)
print("MLP 1 couche (Train) : %.2f%%" % (scores_app[1] * 100))
print("MLP 1 couche (Test)  : %.2f%%" % (scores_test[1] * 100))

# Visualisation Perte & Précision
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_mlp1.history['loss'], label='Train')
plt.plot(history_mlp1.history['val_loss'], label='Test')
plt.title('Perte du Modèle (MLP 1 couche)')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mlp1.history['accuracy'], label='Train')
plt.plot(history_mlp1.history['val_accuracy'], label='Test')
plt.title('Précision du Modèle (MLP 1 couche)')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 3. MLP MULTI-COUCHES AVEC DROPOUT (LUTTE CONTRE L'OVERFITTING)
# ------------------------------------------------------------------------------
print('\n--- 3. MLP PROFOND AVEC REGULARISATION DROPOUT ---')

model_dropout = Sequential()
model_dropout.add(
    Dense(16, input_dim=8, kernel_initializer='uniform', activation='relu')
)
model_dropout.add(Dropout(0.3))  # Taux d'abandon fixé à 30%
model_dropout.add(Dense(8, activation='relu'))
model_dropout.add(Dropout(0.2))
model_dropout.add(Dense(1, activation='sigmoid'))

model_dropout.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
)

history_dropout = model_dropout.fit(
    X_app_scaled,
    Y_app,
    epochs=150,
    validation_data=(X_test_scaled, Y_test),
    batch_size=5,
    verbose=0,
)

scores_app = model_dropout.evaluate(X_app_scaled, Y_app, verbose=0)
scores_test = model_dropout.evaluate(X_test_scaled, Y_test, verbose=0)
print("MLP avec Dropout (Train) : %.2f%%" % (scores_app[1] * 100))
print("MLP avec Dropout (Test)  : %.2f%%" % (scores_test[1] * 100))


# ------------------------------------------------------------------------------
# 4. VALIDATION CROISÉE STRATIFIÉE (10-FOLD CV)
# ------------------------------------------------------------------------------
print('\n--- 4. VALIDATION CROISÉE (10-FOLD) ---')

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

scaler_cv = StandardScaler()
X_cv_scaled = scaler_cv.fit_transform(X)

for train_idx, test_idx in kfold.split(X_cv_scaled, Y):
  model_cv = Sequential()
  model_cv.add(
      Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu')
  )
  model_cv.add(Dense(8, activation='relu'))
  model_cv.add(Dense(1, activation='sigmoid'))

  model_cv.compile(
      loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
  )
  model_cv.fit(
      X_cv_scaled[train_idx],
      Y[train_idx],
      epochs=100,
      batch_size=5,
      verbose=0,
  )

  scores = model_cv.evaluate(
      X_cv_scaled[test_idx], Y[test_idx], verbose=0
  )
  cvscores.append(scores[1] * 100)

print(
    'Moyenne CV Accuracy : %.2f%% (+/- %.2f%%)'
    % (np.mean(cvscores), np.std(cvscores))
)


# ------------------------------------------------------------------------------
# 5. SAUVEGARDE DU MEILLEUR MODÈLE (CHECKPOINT)
# ------------------------------------------------------------------------------
print('\n--- 5. SAUVEGARDE DU MEILLEUR MODÈLE (CHECKPOINT) ---')

# Extension .keras obligatoire avec Keras 3+
filepath = 'meilleur_modele_premier_league.keras'

checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0
)

model_final = Sequential([
    Dense(12, input_dim=8, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model_final.compile(
    loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']
)

model_final.fit(
    X_app_scaled,
    Y_app,
    validation_data=(X_test_scaled, Y_test),
    epochs=150,
    batch_size=5,
    callbacks=[checkpoint],
    verbose=0,
)

print(f"Modèle sauvegardé avec succès sous : {filepath}")
