import pytest
import joblib
import pandas as pd
import numpy as np
import shap
import os

# Charger le modèle CatBoost avec un chemin relatif ou de manière flexible
MODEL_PATH = os.path.join(os.getcwd(), "catboost_model.pkl")  # Utilise le chemin relatif du répertoire courant

@pytest.fixture(scope="module")
def model():
    # Vérification de l'existence du fichier avant de charger le modèle
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Le modèle n'a pas été trouvé à l'emplacement {MODEL_PATH}.")
    return joblib.load(MODEL_PATH)

# Vérifier que le modèle se charge correctement
def test_model_loading(model):
    assert model is not None, "Le modèle n'a pas pu être chargé."

# Définir une entrée de test
@pytest.fixture(scope="module")
def sample_input():
    return pd.DataFrame([{ 
        'Age': 25, 'Number of sexual partners': 2, 'First sexual intercourse': 18,
        'Num of pregnancies': 1, 'Smokes (years)': 0, 'Smokes (packs/year)': 0,
        'Hormonal Contraceptives (years)': 2, 'IUD (years)': 0,
        'STDs (number)': 0, 'STDs: Number of diagnosis': 0
    }])

# Vérifier la structure des entrées
def test_input_structure(sample_input):
    expected_columns = ['Age', 'Number of sexual partners', 'First sexual intercourse',
                        'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
                        'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
                        'STDs: Number of diagnosis']
    assert list(sample_input.columns) == expected_columns, "Les colonnes de l'entrée ne correspondent pas."

# Vérifier la prédiction du modèle
def test_model_prediction(model, sample_input):
    prediction = model.predict(sample_input)
    assert isinstance(prediction, np.ndarray), "La prédiction doit être un tableau NumPy."
    assert prediction.shape == (1,), "La prédiction doit contenir un seul élément."
    assert prediction[0] in [0, 1], "La prédiction doit être 0 ou 1."

# Vérifier l'explication SHAP
def test_shap_explanation(model, sample_input):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(sample_input)
        assert shap_values is not None, "Les valeurs SHAP n'ont pas été générées."
        assert shap_values.values.shape[0] == 1, "Les valeurs SHAP doivent correspondre à une seule observation."
        
        # Vérifier la structure exacte des SHAP values
        print("Shape des SHAP values:", shap_values.values.shape)
        
        # Générer le graphique SHAP, mais ajouter un contrôle si on est dans un environnement sans graphique
        try:
            if len(shap_values.values.shape) == 3:  # Si multi-output
                shap.waterfall_plot(shap_values[0, 0])
            else:
                shap.waterfall_plot(shap_values[0])
        except Exception as e:
            pytest.fail(f"Erreur lors de la génération du graphique SHAP: {e}")
    except Exception as e:
        pytest.fail(f"Erreur lors de l'explication SHAP: {e}")
