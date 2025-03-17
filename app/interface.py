import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Charger le modèle CatBoost enregistré
model = joblib.load("catboost_model.pkl")

# Liste des features utilisées pour la prédiction
features = [
    'Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
    'STDs: Number of diagnosis'
]

# ---- 🎨 Amélioration du style ----
st.set_page_config(page_title="Prédiction de la Biopsy", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #dde2e8;
        }
        .stButton > button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# ---- 🏥 Titre principal ----
st.title("🔬 Prédiction de la Biopsy avec CatBoost")
st.write("Remplissez les informations à gauche et cliquez sur **Prédire** pour obtenir un diagnostic.")

# ---- 📋 Entrée utilisateur via la barre latérale ----
st.sidebar.header("📝 Entrez vos informations")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(f"{feature} :", value=0.0)

# Convertir les entrées en DataFrame
input_df = pd.DataFrame([user_input])

# ---- 🚀 Bouton de prédiction ----
if st.sidebar.button("Prédire la Biopsy"):
    prediction = model.predict(input_df)[0]  # Faire la prédiction
    proba = model.predict_proba(input_df)
    proba_risque = proba[0][0]  # Probabilité d'être dans la classe 0
    st.write(f"🟡 Probabilité du risque : {proba_risque:.2%}")
    biopsy_result = "🟢 Négatif (0)" if prediction == 1 else "🔴 Positif (1)"

    if prediction == 1:
        st.success(f"**Résultat : {biopsy_result}**\nRisque faible de cancer. 👍")
    else:
        st.error(f"**Résultat : {biopsy_result}**\nUn suivi médical est recommandé. 🏥")

    # ---- 📊 Explication SHAP ----
    st.subheader("📌 Explication de la prédiction")
    
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Affichage du graphique SHAP
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap_values[0], max_display=10)
    st.pyplot(fig)

# ---- ℹ️ Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Application développée avec CatBoost et SHAP**")
