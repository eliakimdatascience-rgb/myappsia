import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix

# Algorithmes de Régression
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

# Algorithmes de Classification
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

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ==============================================================================

# Chargement avec chemin relatif pour le serveur
df = pd.read_csv("Premier_League_fr.csv")

print("=== STRUCTURE DU JEU DE DONNÉES ===")
print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
print(df.dtypes)

# Nettoyage des doublons
df = df.drop_duplicates()

# Winsorisation des outliers pour la discipline
def winsorize_iqr(series, threshold=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - threshold * IQR, upper=Q3 + threshold * IQR)

colonnes_a_winsoriser = ['Cartons_Jaunes', 'Cartons_Rouges', 'Score_Discipline']
cols_existantes = [c for c in colonnes_a_winsoriser if c in df.columns]
if cols_existantes:
    df[cols_existantes] = df[cols_existantes].apply(lambda col: winsorize_iqr(col))

# ==============================================================================
# 2. PRÉPARATION DE LA MATRICE (RÉGRESSION)
# ==============================================================================
Y_reg = df['Note_Efficacite'].values

colonnes_a_exclure = [
    'Note_Efficacite', 'Categorie_Performance', 'Joueur', 'Equipe',
    'Contributions_But_Par_90', 'Contributions_Offensives', 
    'Date_Collecte_Donnees', 'Saison', 'Competition', 'Pays', 'Nom_Championnat'
]

cols_to_drop = [c for c in colonnes_a_exclure if c in df.columns]
X_df = df.drop(columns=cols_to_drop)
X_encoded = pd.get_dummies(X_df, drop_first=True)
X_reg = X_encoded.values.astype(np.float32)

kf = KFold(n_splits=10, shuffle=True, random_state=7)

# ==============================================================================
# 3. ÉVALUATION ET ENTRAÎNEMENT DU MEILLEUR MODÈLE DE RÉGRESSION
# ==============================================================================
best_reg_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
best_reg_model.fit(X_reg, Y_reg)

df['predictions_extra_trees'] = best_reg_model.predict(X_reg)
df['prediction_errors'] = np.round(Y_reg - df['predictions_extra_trees'], 2)

# Sauvegarde du modèle Pickle
model_filename = 'extra_trees_regressor_premier_league.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_reg_model, file)

# ==============================================================================
# 4. CLASSIFICATION MACHINE LEARNING
# ==============================================================================
if 'Categorie_Performance' in df.columns:
    Y_class = np.where(df['Categorie_Performance'].astype(str).str.contains('Haute|Élite|Top|1', case=False), 1, 0)
else:
    Y_class = np.where(df['Note_Efficacite'] >= df['Note_Efficacite'].median(), 1, 0)

scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_reg)

best_clf_model = RandomForestClassifier(n_estimators=100, random_state=7)
best_clf_model.fit(X_class_scaled, Y_class)
df['prediction_class'] = best_clf_model.predict(X_class_scaled)

# ==============================================================================
# 5. DEEP LEARNING / RÉSEAU DE NEURONES (MLP MULTICOUCHE VIA SCIKIT-LEARN)
# ==============================================================================
top_categories = ['Classe mondiale (25+)', 'Elite (15-24)']
if 'Categorie_Performance' in df.columns:
    df['Top_Performance'] = df['Categorie_Performance'].apply(lambda x: 1 if x in top_categories else 0)
else:
    df['Top_Performance'] = Y_class

features_cols = ['Apparitions', 'Minutes', 'Buts', 'Passes_Decisives', 
                 'Buts_Par_90', 'Passes_Decisives_Par_90', 'Contributions_But_Par_90', 'Score_Discipline']

# Vérification des colonnes disponibles
features_cols = [c for c in features_cols if c in df.columns]

if features_cols:
    X_dl = df[features_cols].values
    Y_dl = df['Top_Performance'].values

    scaler_dl = StandardScaler()
    X_dl_scaled = scaler_dl.fit_transform(X_dl)

    X_app, X_test, Y_app, Y_test = train_test_split(X_dl_scaled, Y_dl, train_size=0.7, random_state=42, stratify=Y_dl)

    # Réseau MLP (2 couches cachées de 12 et 8 neurones)
    model_final = MLPClassifier(
        hidden_layer_sizes=(12, 8),
        activation='relu',
        solver='adam',
        max_iter=150,
        random_state=42
    )

    model_final.fit(X_app, Y_app)

    # Sauvegarde du modèle au format pickle
    with open('meilleur_modele_premier_league.pkl', 'wb') as file:
        pickle.dump(model_final, file)

    print("Entraînement et sauvegarde du modèle MLP scikit-learn terminés avec succès.")
