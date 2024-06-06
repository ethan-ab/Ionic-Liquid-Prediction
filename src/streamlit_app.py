import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.DEBUG)

st.title("Prédictions des propriétés des IL")

iupac_name = st.text_input("Nom IUPAC")
temperature = st.number_input("Température (K)", value=318.15, step=1.0)
pressure = st.number_input("Pression (kPa)", value=101.325, step=1.0)

if st.button("Prédire la viscosité"):
    data = {
        'iupac_name': iupac_name,
        'temperature': temperature,
        'pressure': pressure,
        'viscosity': 0
    }

    logging.debug(f"Sending data: {data}")

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        response.raise_for_status()  # This will raise an error for bad status codes

        logging.debug(f"Response status code: {response.status_code}")
        logging.debug(f"Response content: {response.content}")

        try:
            response_json = response.json()
            logging.debug(f"Response JSON: {response_json}")
            predicted_viscosity_mPa_s = response_json.get('predicted_viscosity_mPa_s')

            if predicted_viscosity_mPa_s is not None:
                st.success(
                    f"{predicted_viscosity_mPa_s:.2f} mPa.s")
            else:
                st.error("Erreur: La réponse ne contient pas la prédiction.")

        except ValueError as e:
            logging.error(f"JSON decode error: {e}")
            st.error(f"Erreur de décodage JSON: {e}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        st.error(f"Erreur de requête: {e}")
