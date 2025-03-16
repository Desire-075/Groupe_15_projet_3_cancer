import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import joblib  # pour charger un modèle sauvegardé
import shap # pour les explications shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import catboost

# Charger un modèle pré-entrainé (exemple)
# Remplace 'model.pkl' par le chemin réel de ton modèle
model = joblib.load("catboost_model.pkl")
# Créer un explainer SHAP avec ton modèle
explainer = shap.TreeExplainer(model)  # Cela fonctionnera pour de nombreux modèles, y compris les modèles linéaires ou XGBoost

# Diviser en deux colonnes pour le titre et l'image
col1, col2 = st.columns([2, 1])  # Largeur relative : 2 pour le titre et 1 pour l'image

# Ajouter le titre dans la colonne de gauche
with col1:
    st.markdown("<h1 style='text-align: left; color: #4CAF50;'>Cervical Cancer Risk Prediction</h1>", unsafe_allow_html=True)

# Ajouter l'image dans la colonne de droite
with col2:
    image_path = "C:\\Users\\LENOVO\\Desktop\\forex\\Groupe_15_projet_3_cancer\\col de l'utérus image.jpg"  # Mets ton vrai chemin ici
image = Image.open(image_path)

st.image(image, caption=" col de l'utérus")

# Continuer avec le contenu de l'affiche
st.markdown("---")  # Ligne de séparation
st.markdown("### Importance du dépistage")
st.write("Cette application est un outil de support à la décision médicale pour prédire le risque de cancer du col de l'utérus")
st.markdown("<h1 style='text-align: left; color: #4CAF50;'>veuillez remplir le formulaire suivant pour obtenir votre prediction :</h1>", unsafe_allow_html=True)
# Section 1 : Affichage des données saisies
nom = st.text_input("Nom")
age = st.number_input("Age",step=1,min_value=0,max_value=120)
nombre_partenaire = st.number_input("Nombre de partenaires sexuels", step=1, min_value=0)
premiere_rapportsexuel = st.number_input("Age du premier rapport sexuel", step=1, min_value=0)
nombre_grossesse = st.number_input("Nombre de grossesses", step=1, min_value=0)
fumeur = st.selectbox("Fumeur",("Oui","Non"))
fumeur_annee = st.number_input("Nombre d'annees de fummees", step=1, min_value=0)
fumeur_paquet = st.number_input("Nombre de paquets par annee", step=1, min_value=0)
hormones = st.selectbox("Hormones contraceptives",( "Oui","Non"))
hormones_annee = st.number_input("Nombre d'annees de prise de contraceptifs", step=1, min_value=0)
IUD = st.selectbox("Dispositif intra-utérin",("Oui","Non"))
IUD_annee = st.number_input("Nombre d'annees de port de dispositif intra-utérin", step=1, min_value=0)
STD = st.selectbox("Infection sexually transmitted diseases",("Oui","Non"))
STD_annee = st.number_input("Nombre d'annees d'infection", step=1, min_value=0)
STD_nombre = st.number_input("Nombre d'infections", step=1, min_value=0)
STD_condylomatosis = st.selectbox("Condylomatosis",("Oui","Non"))
STD_cervical_condylomatosis = st.selectbox("Cervical condylomatosis",("Oui" ,"Non"))
STD_Vaginal_condylomatosis = st.selectbox("Vaginal condylomatosis",("Oui","Non"))
STD_vulvo_perineal_condylomatosis = st.selectbox("Vulvo-perineal condylomatosis",("Oui","Non"))
STD_syphilis = st.selectbox("Syphilis",("Oui","Non"))
STD_pelvic_inflammatory_disease = st.selectbox("pelvic inflammatory disease",("Oui","Non"))
STD_genital_herpes = st.selectbox("Genital herpes",("Oui","Non"))
STD_molluscum_contagiosum = st.selectbox("Molluscum contagiosum",("Oui","Non"))
STD_AIDS = st.selectbox("AIDS",("Oui","Non"))
STD_HIV = st.selectbox("HIV",("Oui","Non"))
STD_Hepatitis_B = st.selectbox("Hepatitis B",("Oui","Non"))
STD_HPV = st.selectbox("HPV",("Oui","Non"))
STD_number_diagnostique = st.number_input("Nombre de diagnostic", step=1, min_value=0)
STD_time_diagnostique = st.number_input("Temps de diagnostique", step=1, min_value=0)
cancer_famille = st.selectbox("Cancer dans la famille",("Oui","Non"))
cancer_famille_type = st.selectbox("Type de cancer dans la famille" ,("Cervical","Breast","No","Unknown"))
cancer_famille_age = st.number_input("Age de la personne atteinte", step=1, min_value=0)
biopsy = st.selectbox("Résultat de la Biopsie", ("Positive", "Negative"))

# Encodage des données
fumeur_value = 1 if fumeur == "Oui" else 0
hormones_value = 1 if hormones == "Oui" else 0
IUD_value = 1 if IUD == "Oui" else 0
STD_value = 1 if STD == "Oui" else 0
cancer_famille_value = 1 if cancer_famille == "Oui" else 0
cancer_famille_type_dict = {"Cervical": 1, "Breast": 2, "No": 0, "Unknown": -1}
cancer_famille_type_value = cancer_famille_type_dict[cancer_famille_type]
biopsy_value = 1 if biopsy == "Positive" else 0

# Préparation des données
input_data = np.array([[age, nombre_partenaire, premiere_rapportsexuel, nombre_grossesse,
                        fumeur_value, fumeur_annee, hormones_value, hormones_annee,
                        IUD_value, STD_value, STD_nombre, 
                        cancer_famille_value, cancer_famille_type_value, cancer_famille_age,
                        biopsy_value]])

# Bouton de prédiction
if st.button("Prédire"):
    # Convertir input_data en DataFrame pour SHAP
    columns = ["age", "nombre_partenaire", "premiere_rapportsexuel", "nombre_grossesse",
               "fumeur", "fumeur_annee", "hormones", "hormones_annee",
               "IUD", "STD", "STD_nombre", 
               "cancer_famille", "cancer_famille_type", "cancer_famille_age", "biopsy_value"]
    
    input_df = pd.DataFrame(input_data, columns=columns)

    # Prédiction avec le modèle
    prediction = model.predict(input_df)
    st.success(f"Risque de cancer estimé : {int(prediction[0])}")

    # Explication avec SHAP
    shap_values = explainer(input_df)
    st.subheader("Interprétation de la prédiction avec SHAP")
# Calcul des valeurs SHAP
shap_values = explainer(input_data)

    # Affichage des explications SHAP
st.subheader("Interprétation de la prédiction avec SHAP")
    
    # Création du graphique SHAP
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values[0], max_display=5, show=False)
plt.xlabel("Impact sur la prédiction")

    # Affichage du graphique dans Streamlit
st.pyplot(fig)
# Section 5 : Footer
st.markdown("---")
st.write("*Développé par l'équipe 15 et supporté par des modèles de machine learning.*")
