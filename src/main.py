import data_preparation.inputpreparation as inputprep
import xgboost as xgb
import numpy as np


model = xgb.Booster()
model.load_model('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/src/model/model.json')


iupac_name = "1-Octyl-3-methylimidazolium bis(trifluoromethylsulfonyl)imide"
temperature = 318.15
pressure = 101.325
viscosity = 40.36


df_prepared = inputprep.prepare_data(iupac_name, temperature, pressure, viscosity)
X_test = df_prepared.drop(columns=['Viscosity Pa/s'])
y_test_viscosity = np.log(df_prepared['Viscosity Pa/s'])


dtest = xgb.DMatrix(X_test)

predicted_log_viscosity = model.predict(dtest)
predicted_viscosity = np.exp(predicted_log_viscosity)

predicted_viscosity_mPa_s = predicted_viscosity * 1000
print(f"\nLa viscosité du {iupac_name} à {temperature} K est de \033[1m{predicted_viscosity_mPa_s[0]:.2f} mPa.s\033[0m\n")