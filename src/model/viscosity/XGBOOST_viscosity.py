import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import joblib
import src.data_preparation.preparation as preparation

# Fonction pour entraîner le modèle
def train_model(X_train, y_train, X_val, y_val, best_params):
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    return model

# Fonction pour évaluer le modèle
def evaluate_model(model, X, y, set_name="Validation"):
    preds_log = model.predict(X)
    preds = np.exp(preds_log)
    mse = mean_squared_error(np.exp(y), preds)
    rmse = math.sqrt(mse)
    print(f"{set_name} Mean Squared Error for Viscosity: {mse}")
    print(f"{set_name} Root Mean Squared Error for Viscosity: {rmse}")
    return preds, rmse

# Fonction pour calculer les résidus
def calculate_residuals(y_true, preds):
    return np.exp(y_true) - preds

# Fonction pour calculer les statistiques des résidus
def calculate_residual_statistics(model, X, y):
    preds_log = model.predict(X)
    preds = np.exp(preds_log)
    residuals = calculate_residuals(y, preds)
    rmse = math.sqrt(mean_squared_error(np.exp(y), preds))
    return rmse, preds, residuals

# Fonction pour diviser les données en segments réguliers
def create_segments(min_val, max_val, num_segments):
    return np.linspace(min_val, max_val, num_segments + 1)

# Fonction pour calculer les RMSE par segment
def calculate_segment_rmse(preds, residuals, segments):
    segment_rmse = []

    for i in range(len(segments) - 1):
        segment_mask = (preds >= segments[i]) & (preds < segments[i + 1])
        segment_residuals = residuals[segment_mask]
        if len(segment_residuals) > 0:
            rmse = np.sqrt(np.mean(np.square(segment_residuals)))
            segment_rmse.append(rmse)
            print(f"Segment [{segments[i]}, {segments[i + 1]}) RMSE: {rmse}")
        else:
            segment_rmse.append(0)
            print(f"Segment [{segments[i]}, {segments[i + 1]}) RMSE: 0")

    return segment_rmse

# Fonction pour calculer l'intervalle de confiance localement
def calculate_confidence_interval(segments, segment_rmse, prediction):
    segment_index = np.digitize(prediction, segments) - 1
    if segment_index >= len(segment_rmse):
        segment_index = len(segment_rmse) - 1
    rmse = segment_rmse[segment_index]

    confidence_interval = 1.96 * rmse  # pour un intervalle de confiance de 95%
    return confidence_interval

def predict_with_confidence(model, X, segments, segment_rmse):
    preds_log = model.predict(X)
    preds = np.exp(preds_log)
    lower_bounds = []
    upper_bounds = []

    for pred in preds:
        conf_interval = calculate_confidence_interval(segments, segment_rmse, pred)
        lower_bounds.append(pred - conf_interval)
        upper_bounds.append(pred + conf_interval)

    return preds, lower_bounds, upper_bounds

# Fonction pour évaluer les prédictions avec intervalles de confiance
def evaluate_predictions_with_bounds(y_true, preds, lower_bounds, upper_bounds):
    mse_preds = mean_squared_error(np.exp(y_true), preds)
    mse_lower_bounds = mean_squared_error(np.exp(y_true), lower_bounds)
    mse_upper_bounds = mean_squared_error(np.exp(y_true), upper_bounds)

    rmse_preds = math.sqrt(mse_preds)
    rmse_lower_bounds = math.sqrt(mse_lower_bounds)
    rmse_upper_bounds = math.sqrt(mse_upper_bounds)

    print(f"RMSE for predictions: {rmse_preds}")
    print(f"RMSE for lower bounds: {rmse_lower_bounds}")
    print(f"RMSE for upper bounds: {rmse_upper_bounds}")

    return rmse_preds, rmse_lower_bounds, rmse_upper_bounds

# Fonction principale pour entraîner et évaluer le modèle
def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    best_params = {
        'colsample_bytree': 0.7,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 300,
        'subsample': 0.9
    }

    model = train_model(X_train, y_train, X_val, y_val, best_params)
    print("-----------------------")

    val_preds, val_rmse = evaluate_model(model, X_val, y_val, "Validation")

    residuals = calculate_residuals(y_val, val_preds)


    min_val, max_val = np.min(val_preds), np.max(val_preds)
    segments = create_segments(min_val, max_val, 15)

    segment_rmse = calculate_segment_rmse(val_preds, residuals, segments)

    test_preds, test_rmse = evaluate_model(model, X_test, y_test, "Test")
    print("------------------------------------------")
    print(f"Test Root Mean Squared Error for Viscosity in mPa.s: {test_rmse * 1000}")

    joblib.dump(segments, "segments.pkl")
    joblib.dump(segment_rmse, "segment_rmse.pkl")

    joblib.dump(model, "modelviscosity.pkl")

    return model, residuals, val_rmse, val_preds, segments, segment_rmse

if __name__ == "__main__":

    df_cleaned = preparation.load_and_clean_data('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/IlThermo-smiled_dataset_viscosity.csv', "DeltaViscosity")
    df_descriptors_cleaned = preparation.generate_and_clean_descriptors(df_cleaned)
    df_filtered = preparation.filter_data_viscosity(df_descriptors_cleaned)
    X_train_v, X_val_v, X_test_v, y_train_v, y_val_v, y_test_v = preparation.prepare_final_data_log(df_filtered, "Viscosity Pa/s")


    model, residuals, val_rmse, val_preds, segments, segment_rmse = train_and_evaluate_model(X_train_v, y_train_v, X_val_v, y_val_v, X_test_v, y_test_v)

    preds, lower_bounds, upper_bounds = predict_with_confidence(model, X_test_v, segments, segment_rmse)

    rmse_preds, rmse_lower_bounds, rmse_upper_bounds = evaluate_predictions_with_bounds(y_test_v, preds, lower_bounds, upper_bounds)

    for pred, lower, upper in zip(preds, lower_bounds, upper_bounds):
        print(f"Prediction: {pred}, Lower Bound: {lower}, Upper Bound: {upper}")
