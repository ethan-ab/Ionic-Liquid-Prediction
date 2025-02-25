from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib
import logging

import data_preparation.input_preparation as inputprep

import src.model.viscosity.XGBOOST_viscosity as viscosity_model
import src.model.density.XGBOOST_density as density_model
from src.model.electrical_conductivity import XGBOOST_ec as conductivity_model
import src.model.melting_temperature.XGBOOST_melting_temperature as melting_temperature_model

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Load models
viscosity_model_path = '/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/viscosity/modelviscosity.pkl'
electrical_conductivity_model_path = "/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/electrical_conductivity/modelec.pkl"
density_model_path = "/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/density/modeldensity.pkl"
melting_temperature_model_path = "/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/melting_temperature/model_temperature.pkl"

model_viscosity = joblib.load(viscosity_model_path)
model_electrical_conductivity = joblib.load(electrical_conductivity_model_path)
model_density = joblib.load(density_model_path)
model_melting_temperature = joblib.load(melting_temperature_model_path)

segments = joblib.load('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/viscosity/segments.pkl')
segment_rmse = joblib.load('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/viscosity/segment_rmse.pkl')

rmse_viscosity = 10.2 * 0.001  # Example value, replace with actual RMSE
rmse_density = 0.0953948*1000   # Example value, replace with actual RMSE
rmse_electrical_conductivity = 0.946532114989638 # Example value, replace with actual RMSE
rmse_melting_temperature = 20.05736

def calculate_confidence_interval_visc(prediction):
    return viscosity_model.calculate_confidence_interval(segments, segment_rmse, prediction)

def calculate_confidence_interval(prediction, rmse, round_digits=2):
    margin_of_error = 1.96 * rmse
    lower_bound = max(0, prediction - margin_of_error)  # Ensure lower bound is at least 0
    upper_bound = prediction + margin_of_error
    if round_digits is not None:
        lower_bound = round(lower_bound, round_digits)
        upper_bound = round(upper_bound, round_digits)
    return lower_bound, upper_bound

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Data received: {data}")

        iupac_name = data['iupac_name']
        temperature = data['temperature']
        pressure = data['pressure']

        response_data = {}

        if 'viscosity' in data:
            viscosity = data['viscosity']
            df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, viscosity, "viscosity")
            app.logger.debug(f"Prepared DataFrame: {df_prepared}")

            X_test = df_prepared.drop(columns=['Viscosity Pa/s'])
            y_test_viscosity = np.log(df_prepared['Viscosity Pa/s'])

            # Use the model to predict
            predicted_log_viscosity = model_viscosity.predict(X_test)
            predicted_viscosity = np.exp(predicted_log_viscosity)
            predicted_viscosity_mPa_s = float(predicted_viscosity[0] * 1000)  # Convert to mPa.s

            # Calculate the local confidence interval
            confidence_interval = calculate_confidence_interval_visc(predicted_viscosity[0])
            lower_bound = max(0, predicted_viscosity[0] - confidence_interval)
            upper_bound = predicted_viscosity[0] + confidence_interval

            lower_bound_mPa_s = lower_bound * 1000
            upper_bound_mPa_s = upper_bound * 1000

            # Rounding to 4 decimal places
            lower_bound_rounded = round(lower_bound_mPa_s, 1)
            upper_bound_rounded = round(upper_bound_mPa_s, 1)

            app.logger.debug(f"Predicted viscosity (mPa.s): {predicted_viscosity_mPa_s}")
            response_data['predicted_viscosity_mPa_s'] = predicted_viscosity_mPa_s
            response_data['viscosity_confidence_interval'] = [lower_bound_rounded, upper_bound_rounded]

        if 'density' in data:
            density = data['density']
            df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, density, 'density')
            app.logger.debug(f"Prepared DataFrame for density: {df_prepared}")

            X_test = df_prepared.drop(columns=['Specific density kg/m3'])
            y_test = df_prepared['Specific density kg/m3']

            predicted_density = model_density.predict(X_test)
            predicted_density_kg_m3 = float(predicted_density[0])

            lower_bound, upper_bound = calculate_confidence_interval(predicted_density_kg_m3, rmse_density, 2)

            app.logger.debug(f"Predicted density (kg/m3): {predicted_density_kg_m3}")
            response_data['predicted_density_kg_m3'] = predicted_density_kg_m3
            response_data['density_confidence_interval'] = [lower_bound, upper_bound]

        if 'electrical_conductivity' in data:
            electrical_conductivity = data['electrical_conductivity']
            df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, electrical_conductivity, 'electrical_conductivity')
            app.logger.debug(f"Prepared DataFrame for electrical conductivity: {df_prepared}")

            X_test = df_prepared.drop(columns=['Electrical conductivity S/m'])
            y_test = df_prepared['Electrical conductivity S/m']

            predicted_electrical_conductivity = model_electrical_conductivity.predict(X_test)
            predicted_electrical_conductivity_S_m = float(predicted_electrical_conductivity[0]) * 10

            lower_bound, upper_bound = calculate_confidence_interval(predicted_electrical_conductivity_S_m, rmse_electrical_conductivity, 2)

            app.logger.debug(f"Predicted electrical conductivity (S/m): {predicted_electrical_conductivity_S_m}")
            response_data['predicted_electrical_conductivity_S_m'] = predicted_electrical_conductivity_S_m
            response_data['electrical_conductivity_confidence_interval'] = [lower_bound, upper_bound]

        if 'melting_temperature' in data:
            df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, None, 'melting_temperature')
            app.logger.debug(f"Prepared DataFrame for melting temperature: {df_prepared}")

            X_test = df_prepared.drop(columns=['Melting Temperature'])
            y_test = df_prepared['Melting Temperature']

            predicted_melting_temperature = model_melting_temperature.predict(X_test)
            predicted_melting_temperature_K = float(predicted_melting_temperature[0])  # Assuming model predicts in Kelvin

            lower_bound, upper_bound = calculate_confidence_interval(predicted_melting_temperature_K, rmse_melting_temperature, None)

            app.logger.debug(f"Predicted melting temperature (K): {predicted_melting_temperature_K}")
            response_data['predicted_melting_temperature_K'] = predicted_melting_temperature_K
            response_data['melting_temperature_confidence_interval'] = [int(lower_bound), int(upper_bound)]

        if not response_data:
            raise ValueError("Either 'viscosity', 'density', 'electrical_conductivity', or 'melting_temperature' must be provided in the data.")

        app.logger.debug(f"Response JSON: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
