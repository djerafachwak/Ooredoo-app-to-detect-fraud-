from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib 
from preprocessing1 import full_preprocess

app = Flask(__name__)

# Charger ton modèle entraîné
model = joblib.load("random_forest__best_fraud_model11.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recevoir les données en JSON
        json_data = request.get_json()
        df_raw = pd.DataFrame(json_data)

        # Charger le mapping des antennes
        df_mapping = pd.read_csv("CELL-LAC-WILAYA.csv", sep=";")

        # Prétraitement complet
        df_ready = full_preprocess(df_raw, df_mapping)
        df_ready = df_ready.replace({np.nan: None})
        # Prédiction
        X = df_ready.drop(columns=["Fraud"], errors="ignore")
        predictions = model.predict(X)
        df_ready["Prediction"] = predictions

        return jsonify(df_ready.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
