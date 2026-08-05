import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st

# TensorFlow / Keras (VOTRE REQUÊTE : Keras strictement conservé)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Scikit-Learn
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# CONFIGURATION STREAMLIT & LAYOUT
# ==============================================================================
st.set_page_config(page_title="Premier League Analytics", layout="wide")

# ==============================================================================
# BARRE LATÉRALE (SIDEBAR - DESIGN CONFORME À LA CAPTURE)
# ==============================================================================
with st.sidebar:
    # Card Auteur du Projet
    st.markdown("""
    <div style="background-color: #1a365d; padding: 18px; border-radius: 12px; color: white; font-family: sans-serif;">
        <h3 style="margin-top:0; margin-bottom: 12px; font-size: 16px; color: #f6ad55;">👨‍💻 Auteur du Projet</h3>
        <p style="margin: 4px 0; font-size: 13px;"><b>Nom :</b> Osei Poku Eliakim</p>
        <p style="margin: 4px 0; font-size: 13px;"><b>📞 Tél :</b> 0150059383</p>
        <p style="margin: 4px 0; font-size: 13px;"><b>✉️ Email :</b><br><a href="mailto:eliakimdatascience@gmail.com" style="color:#63b3ed;">eliakimdatascience@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Source des données
    st.markdown("### 📁 Source des Données")
    uploaded_file = st.file_uploader("Importer un nouveau Dataset CSV", type=["csv"])
    
    st.markdown("---")
    
    # Configuration du modèle
    st.markdown("### ⚙️ Configuration du Modèle")
    algo_choice = st.selectbox(
        "Choisissez l'algorithme à utiliser :",
        ["Random Forest (Machine Learning)", "Extra Trees Regressor", "Keras Deep Learning (Sequential)"]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    mode_utilisation = st.radio(
        "Mode d'utilisation :",
        ["Consulter un Joueur du Dataset", "Simuler une Recrue (Scouting)"]
    )

# ==============================================================================
# 1. CHARGEMENT ET TRAITEMENT DES DONNÉES (CHARGEMENT HYBRIDE LOCAL/UPLOAD)
# ==============================================================================
@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    
    # Gestion du chemin d'accès local ou relatif
    chemin_local = r"E:\COURS INSSEDS\data set\Premier_League_fr.csv"
    if os.path.exists(chemin_local):
        return pd.read_csv(chemin_local)
    elif os.path.exists("Premier_League_fr.csv"):
        return pd.read_csv("Premier_League_fr.csv")
    else:
        st.error("Fichier Premier_League_fr.csv introuvable.")
        st.stop()

df = load_data(uploaded_file)

# --- Traitement des Doublons ---
df = df.drop_duplicates()

# --- Traitement sélectif des Outliers par Winsorisation ---
def winsorize_iqr(series, threshold=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return series.clip(lower=lower, upper=upper)

colonnes_a_winsoriser = ['Cartons_Jaunes', 'Cartons_Rouges', 'Score_Discipline']
cols_existantes = [c for c in colonnes_a_winsoriser if c in df.columns]
if cols_existantes:
    df[cols_existantes] = df[cols_existantes].apply(lambda col: winsorize_iqr(col, threshold=1.5))

# ==============================================================================
# 2. ENTRAÎNEMENT KERAS (DEEP LEARNING SÉQUENTIEL)
# ==============================================================================
top_categories = ['Classe mondiale (25+)', 'Elite (15-24)']
if 'Categorie_Performance' in df.columns:
    df['Top_Performance'] = df['Categorie_Performance'].apply(lambda x: 1 if x in top_categories else 0)
else:
    df['Top_Performance'] = np.where(df['Note_Efficacite'] >= df['Note_Efficacite'].median(), 1, 0)

features_cols = [
    'Apparitions', 'Minutes', 'Buts', 'Passes_Decisives',
    'Buts_Par_90', 'Passes_Decisives_Par_90', 'Contributions_But_Par_90', 'Score_Discipline'
]
cols_k = [c for c in features_cols if c in df.columns]

X_k = df[cols_k].values
Y_k = df['Top_Performance'].values

scaler_k = StandardScaler()
X_k_scaled = scaler_k.fit_transform(X_k)

# Modèle Keras Séquentiel avec Dropout (votre code)
model_final_keras = Sequential([
    Dense(12, input_dim=len(cols_k), activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_final_keras.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_final_keras.fit(X_k_scaled, Y_k, epochs=30, batch_size=5, verbose=0)

# Sauvegarde du modèle Keras (.keras)
model_final_keras.save('meilleur_modele_premier_league.keras')

# ==============================================================================
# 3. ENTRAÎNEMENT MODÈLES MACHINE LEARNING (EXTRA TREES & RANDOM FOREST)
# ==============================================================================
Y_reg = df['Note_Efficacite'].values if 'Note_Efficacite' in df.columns else np.zeros(len(df))
colonnes_a_exclure = [
    'Note_Efficacite', 'Categorie_Performance', 'Joueur', 'Equipe',
    'Contributions_But_Par_90', 'Contributions_Offensives', 
    'Date_Collecte_Donnees', 'Saison', 'Competition', 'Pays', 'Nom_Championnat', 'Position'
]
cols_to_drop = [c for c in colonnes_a_exclure if c in df.columns]
X_df = df.drop(columns=cols_to_drop)
X_encoded = pd.get_dummies(X_df, drop_first=True)
X_reg = X_encoded.values.astype(np.float32)

best_model_et = ExtraTreesRegressor(n_estimators=100, random_state=42)
best_model_et.fit(X_reg, Y_reg)

with open('extra_trees_regressor_premier_league.pkl', 'wb') as file:
    pickle.dump(best_model_et, file)

best_model_rf = RandomForestClassifier(n_estimators=100, random_state=7)
best_model_rf.fit(X_k_scaled, Y_k)

# ==============================================================================
# 4. CORPS PRINCIPAL DE L'APPLICATION (DESIGN EXACT DU DEUXIÈME SCREENSHOT)
# ==============================================================================

# Entête principal
st.markdown("# ⚽ Premier League Sport Analytics & Decision Tool")
st.markdown("<p style='color: #718096; margin-top: -15px;'>Projet Machine Learning & Deep Learning — <i>Outil d'aide à la décision & Scouting</i></p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if mode_utilisation == "Consulter un Joueur du Dataset":
    st.markdown("## 📊 Fiche Joueur & Analyse de Performance")
    
    liste_joueurs = df['Joueur'].unique() if 'Joueur' in df.columns else []
    joueur_sel = st.selectbox("Sélectionnez un joueur :", liste_joueurs)
    
    if joueur_sel:
        row = df[df['Joueur'] == joueur_sel].iloc[0]
        
        # Structure de carte d'informations du joueur (Club, Poste, Buts/Passes D., Note Réelle)
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.caption("Club")
            st.markdown(f"### {row.get('Equipe', 'N/A')}")
            
        with c2:
            st.caption("Poste")
            st.markdown(f"### {row.get('Position', row.get('Poste', 'Attaquant'))}")
            
        with c3:
            st.caption("Buts / Passes D.")
            b = int(row.get('Buts', 0))
            p = int(row.get('Passes_Decisives', 0))
            st.markdown(f"### {b} / {p}")
            
        with c4:
            st.caption("Note Réelle")
            note_r = row.get('Note_Efficacite', 0.0)
            st.markdown(f"### {note_r:.2f} / 3.00")
            
        st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)
        
        # Section Prédiction
        st.markdown("### 🎯 Prédiction selon l'Algorithme Sélectionné")
        
        idx = df[df['Joueur'] == joueur_sel].index[0]
        
        if algo_choice == "Keras Deep Learning (Sequential)":
            pred_prob = model_final_keras.predict(X_k_scaled[idx:idx+1], verbose=0)[0][0]
            note_estimee = pred_prob * 3.0
            is_elite = pred_prob >= 0.5
        elif algo_choice == "Extra Trees Regressor":
            note_estimee = best_model_et.predict([X_reg[idx]])[0]
            is_elite = note_estimee >= 2.0
        else: # Random Forest
            pred_prob = best_model_rf.predict_proba(X_k_scaled[idx:idx+1])[0][1]
            note_estimee = pred_prob * 3.0
            is_elite = pred_prob >= 0.5

        st.markdown(f"**Note d'Efficacité estimée :** <span style='color: #48bb78; font-weight: bold;'>{note_estimee:.2f} / 3.00</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Badge de niveau (fond vert comme sur la capture d'écran)
        if is_elite:
            st.markdown("""
            <div style="background-color: #d4edda; color: #155724; padding: 12px 20px; border-radius: 8px; font-weight: bold;">
                Niveau Prédit : 🌟 World Class / Élite
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #fff3cd; color: #856404; padding: 12px 20px; border-radius: 8px; font-weight: bold;">
                Niveau Prédit : ⚽ Titulaire / En Progression
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("## 🔮 Simulation d'une Recrue (Scouting)")
    st.write("Entrez les statistiques du joueur pour calculer son score d'efficacité avec Keras / Machine Learning.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        b_sim = st.number_input("Buts", 0, 50, 10)
        p_sim = st.number_input("Passes Décisives", 0, 30, 5)
    with col_b:
        m_sim = st.number_input("Minutes", 0, 3800, 2000)
        app_sim = st.number_input("Apparitions", 0, 38, 25)
    with col_c:
        b90_sim = b_sim / (m_sim / 90) if m_sim > 0 else 0
        p90_sim = p_sim / (m_sim / 90) if m_sim > 0 else 0
        disc_sim = st.number_input("Score Discipline", 0, 50, 5)

    if st.button("Lancer la Prédiction", type="primary"):
        input_data = np.array([[app_sim, m_sim, b_sim, p_sim, b90_sim, p90_sim, b90_sim+p90_sim, disc_sim]])
        input_scaled = scaler_k.transform(input_data)
        score_k = model_final_keras.predict(input_scaled, verbose=0)[0][0]
        
        st.success(f"Probabilité de performance d'Élite (Keras) : {score_k * 100:.1f}%")
        st.info(f"Note d'Efficacité estimée : {(score_k * 3.0):.2f} / 3.00")
