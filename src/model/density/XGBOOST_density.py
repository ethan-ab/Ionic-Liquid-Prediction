import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import src.data_preparation.descriptor
import src.data_preparation.preparation as preparation
from scipy import stats
import joblib


# Entraînement du modèle XGBoost avec les meilleurs paramètres
def train_model(X_train, y_train, X_val, y_val):
    best_params = {
        'colsample_bytree': 0.7428214242463981,
        'learning_rate': 0.15220821760665343,
        'max_depth': 6,
        'n_estimators': 300,
        'subsample': 0.8321948719751764
    }

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    print("Model trained successfully.")

    joblib.dump(best_model, "modeldensity.pkl")

    return best_model


# Calculate residuals and confidence interval based on validation set
def calculate_residual_statistics(model, X_val, y_val):
    val_preds = model.predict(X_val)
    residuals = y_val - val_preds

    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    return rmse, val_preds, residuals


# Function to calculate confidence interval for a specific prediction
def calculate_confidence_interval(rmse, val_preds, residuals, prediction):
    # Segment residuals based on prediction ranges
    segments = np.percentile(val_preds, [0, 25, 50, 75, 100])
    std_devs = []

    for i in range(len(segments) - 1):
        segment_mask = (val_preds >= segments[i]) & (val_preds < segments[i + 1])
        segment_residuals = residuals[segment_mask]
        if len(segment_residuals) > 0:
            std_devs.append(np.std(segment_residuals))
        else:
            std_devs.append(0)

    # Determine the segment for the given prediction
    segment_index = np.digitize(prediction, segments) - 1
    if segment_index >= len(std_devs):
        segment_index = len(std_devs) - 1
    std_dev = std_devs[segment_index]

    # Calculate confidence interval using the segment standard deviation
    confidence_interval = 1.96 * std_dev  # for 95% confidence

    return confidence_interval


# Function to evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    test_preds = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = math.sqrt(test_mse)
    print(f"Test Mean Squared Error for density: {test_mse}")
    print(f"Test Root Mean Squared Error for density: {test_rmse}")
    print("------------------------------------------")
    print(f"Test Root Mean Squared Error for density in g/cm3: {test_rmse * 0.001}")

    return test_rmse


# Function to make predictions with confidence intervals
def predict_with_confidence(model, X, rmse, val_preds, residuals):
    preds = model.predict(X)
    lower_bounds = []
    upper_bounds = []

    for pred in preds:
        conf_interval = calculate_confidence_interval(rmse, val_preds, residuals, pred)
        lower_bounds.append(pred - conf_interval)
        upper_bounds.append(pred + conf_interval)

    return preds, lower_bounds, upper_bounds


# Load and prepare data
#df_cleaned_density = preparation.load_and_clean_data('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/densitysmiled.csv', "Delta density")
#df_descriptors_cleaned_density = preparation.generate_and_clean_descriptors(df_cleaned_density)
#df_filtered_density = preparation.filter_data_density(df_descriptors_cleaned_density)

#X_train_v, X_val_v, X_test_v, y_train_v, y_val_v, y_test_v = preparation.prepare_final_data(df_filtered_density,'Specific density kg/m3')


