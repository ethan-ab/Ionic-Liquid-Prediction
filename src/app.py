from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pandas as pd
import data_preparation.inputpreparation as inputprep
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

model = xgb.Booster()
model.load_model('model/model.json')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Data received: {data}")

        iupac_name = data['iupac_name']
        temperature = data['temperature']
        pressure = data['pressure']
        viscosity = data['viscosity']

        df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, viscosity)
        app.logger.debug(f"Prepared DataFrame: {df_prepared}")

        X_test = df_prepared.drop(columns=['Viscosity Pa/s'])
        y_test_viscosity = np.log(df_prepared['Viscosity Pa/s'])

        dtest = xgb.DMatrix(X_test)
        predicted_log_viscosity = model.predict(dtest)
        predicted_viscosity = np.exp(predicted_log_viscosity)
        predicted_viscosity_mPa_s = float(predicted_viscosity[0] * 1000)  # Convert to float

        app.logger.debug(f"Predicted viscosity (mPa.s): {predicted_viscosity_mPa_s}")

        response = jsonify(predicted_viscosity_mPa_s=predicted_viscosity_mPa_s)
        app.logger.debug(f"Response JSON: {response.get_json()}")

        return response
    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
