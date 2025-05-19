import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Ooredoo Fraud Detection", layout="wide")
st.title("📡 Telecom Fraud Detection – Ooredoo")

uploaded_file = st.file_uploader("📤 Upload your CDR file (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Uploaded Data")
    st.dataframe(df.head())

    if st.button("🔍 Detect Fraud"):
        try:
            # 🔧 Nettoyer les NaN pour éviter erreur JSON
            df = df.fillna("missing")  # ou 0, selon le type de variable

            # 🔁 Envoyer vers API Flask
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json=df.to_dict(orient="records")
            )

            if response.status_code == 200:
                results = pd.DataFrame(response.json())
                st.success("✅ Fraud detection completed!")
                st.dataframe(results)

                if "Prediction" in results.columns:
                    st.subheader("📊 Prediction Summary")
                    st.bar_chart(results["Prediction"].value_counts())

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"🚨 Could not connect to API: {e}")
