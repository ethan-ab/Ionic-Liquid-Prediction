import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import src.data_preparation.descriptor
import src.data_preparation.preparation as preparation



# Entraînement du modèle XGBoost avec les meilleurs paramètres
# Input : X et y pour l'entraînement et la validation
# Output : Modèle entraîné et évaluations de performance
def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    best_params = {
        'colsample_bytree': 0.7,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 300,
        'subsample': 0.9
    }

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

    print(f"-----------------------")

    # Prédictions et évaluation sur les données de validation
    val_preds_log = best_model.predict(X_val)
    val_preds = np.exp(val_preds_log)
    val_mse = mean_squared_error(np.exp(y_val), val_preds)
    val_rmse = math.sqrt(val_mse)
    print(f"Validation Mean Squared Error for Viscosity: {val_mse}")
    print(f"Validation Root Mean Squared Error for Viscosity: {val_rmse}")

    # Prédictions et évaluation sur les données de test
    test_preds_log = best_model.predict(X_test)
    test_preds = np.exp(test_preds_log)
    test_mse = mean_squared_error(np.exp(y_test), test_preds)
    test_rmse = math.sqrt(test_mse)
    print(f"Test Mean Squared Error for Viscosity: {test_mse}")
    print(f"Test Root Mean Squared Error for Viscosity: {test_rmse}")
    print("------------------------------------------")
    print(f"Test Root Mean Squared Error for Viscosity in mPa.s: {test_rmse * 1000}")

    best_model.save_model("model.json")

df_cleaned = preparation.load_and_clean_data('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/IlThermo-smiled_dataset_viscosity.csv')
df_descriptors_cleaned = preparation.generate_and_clean_descriptors(df_cleaned)
df_filtered = preparation.filter_data(df_descriptors_cleaned)

X_train_v, X_val_v, X_test_v, y_train_v, y_val_v, y_test_v = preparation.prepare_final_data(df_filtered)

train_and_evaluate_model(X_train_v, y_train_v, X_val_v, y_val_v, X_test_v, y_test_v)
