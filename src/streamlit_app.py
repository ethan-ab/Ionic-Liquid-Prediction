import streamlit as st
import requests
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(to right, #ece9e6, #ffffff);
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.75rem 1.5rem;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #388E3C;
        }
        .center-content {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        table {
            width: 100%;
            table-layout: fixed; /* Ensure fixed table layout */
        }
        thead th {
            text-align: center; /* Center-align table headers */
            background-color: #f4f4f9;
            font-weight: bold;
        }
        tbody td {
            text-align: center; /* Center-align table cells */
        }
        tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)



iupac_name = st.text_input("Nom IUPAC", placeholder="Entrez le nom IUPAC de l'espèce")
temperature = st.number_input("Température (K)", value=318.15, step=1.0)
pressure = st.number_input("Pression (kPa)", value=101.325, step=1.0)

# Center the button and results
with st.container():
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)

    if st.button("Prédire les propriétés"):
        data_viscosity = {
            'iupac_name': iupac_name,
            'temperature': temperature,
            'pressure': pressure,
            'viscosity': 0
        }

        data_density = {
            'iupac_name': iupac_name,
            'temperature': temperature,
            'pressure': pressure,
            'density': 0
        }

        data_conductivity = {
            'iupac_name': iupac_name,
            'temperature': temperature,
            'pressure': pressure,
            'electrical_conductivity': 0
        }

        data_melting_temperature = {
            'iupac_name': iupac_name,
            'temperature': temperature,
            'pressure': pressure,
            'melting_temperature': 0
        }

        try:
            # Prédiction de la viscosité
            response_viscosity = requests.post("http://127.0.0.1:5000/predict", json=data_viscosity)
            response_viscosity.raise_for_status()  # This will raise an error for bad status codes

            logging.debug(f"Response status code for viscosity: {response_viscosity.status_code}")
            logging.debug(f"Response content for viscosity: {response_viscosity.content}")

            response_json_viscosity = response_viscosity.json()
            logging.debug(f"Response JSON for viscosity: {response_json_viscosity}")
            predicted_viscosity_mPa_s = response_json_viscosity.get('predicted_viscosity_mPa_s')
            confidence_interval_viscosity = response_json_viscosity.get('viscosity_confidence_interval', "N/A")

            # Prédiction de la densité
            response_density = requests.post("http://127.0.0.1:5000/predict", json=data_density)
            response_density.raise_for_status()  # This will raise an error for bad status codes

            logging.debug(f"Response status code for density: {response_density.status_code}")
            logging.debug(f"Response content for density: {response_density.content}")

            response_json_density = response_density.json()
            logging.debug(f"Response JSON for density: {response_json_density}")
            predicted_density_kg_m3 = response_json_density.get('predicted_density_kg_m3')
            confidence_interval_density = response_json_density.get('density_confidence_interval', "N/A")

            # Prédiction de la conductivité électrique
            response_conductivity = requests.post("http://127.0.0.1:5000/predict", json=data_conductivity)
            response_conductivity.raise_for_status()  # This will raise an error for bad status codes

            logging.debug(f"Response status code for electrical conductivity: {response_conductivity.status_code}")
            logging.debug(f"Response content for electrical conductivity: {response_conductivity.content}")

            response_json_conductivity = response_conductivity.json()
            logging.debug(f"Response JSON for electrical conductivity: {response_json_conductivity}")
            predicted_conductivity_S_m = response_json_conductivity.get('predicted_electrical_conductivity_S_m')
            confidence_interval_conductivity = response_json_conductivity.get('electrical_conductivity_confidence_interval', "N/A")

            # Prédiction de la température de fusion
            response_melting_temperature = requests.post("http://127.0.0.1:5000/predict", json=data_melting_temperature)
            response_melting_temperature.raise_for_status()  # This will raise an error for bad status codes

            logging.debug(f"Response status code for melting temperature: {response_melting_temperature.status_code}")
            logging.debug(f"Response content for melting temperature: {response_melting_temperature.content}")

            response_json_melting_temperature = response_melting_temperature.json()
            logging.debug(f"Response JSON for melting temperature: {response_json_melting_temperature}")
            predicted_melting_temperature_K = response_json_melting_temperature.get('predicted_melting_temperature_K')
            confidence_interval_melting_temperature = response_json_melting_temperature.get('melting_temperature_confidence_interval', "N/A")

            if (predicted_viscosity_mPa_s is not None and predicted_density_kg_m3 is not None
                and predicted_conductivity_S_m is not None and predicted_melting_temperature_K is not None):
                # Create tables
                results_properties = pd.DataFrame({
                    "Propriété": ["Viscosité (mPa.s)", "Densité (kg/m3)", "Conductivité Électrique (S/m)"],
                    "Valeur": [f"{predicted_viscosity_mPa_s:.2f}", f"{predicted_density_kg_m3:.2f}",
                               f"{predicted_conductivity_S_m:.2f}"],
                    "Intervalle de confiance à 95%": [confidence_interval_viscosity, confidence_interval_density, confidence_interval_conductivity]
                })

                melting_temperature_properties = pd.DataFrame({
                    "Propriété": ["Température de Fusion (K)"],
                    "Valeur": [f"{predicted_melting_temperature_K:.2f}"],
                    "Intervalle de confiance à 95%": [confidence_interval_melting_temperature]
                })

                # Convert DataFrames to HTML with centering using inline styles
                results_properties_html = results_properties.to_html(index=False, justify='center')
                melting_temperature_properties_html = melting_temperature_properties.to_html(index=False, justify='center')

                # Style headers to be bold
                results_properties_html = results_properties_html.replace('<th>', '<th style="font-weight: bold;">')
                melting_temperature_properties_html = melting_temperature_properties_html.replace('<th>', '<th style="font-weight: bold;">')

                # Display tables
                st.subheader(f"Propriétés à {temperature} K et {pressure} kPa")
                st.markdown(results_properties_html, unsafe_allow_html=True)

                st.subheader("Propriétés du liquide ionique")
                st.markdown(melting_temperature_properties_html, unsafe_allow_html=True)

            else:
                st.error("Erreur: La réponse ne contient pas les prédictions attendues.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            st.error(f"Erreur de requête: {e}")

        except ValueError as e:
            logging.error(f"JSON decode error: {e}")
            st.error(f"Erreur de décodage JSON: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
