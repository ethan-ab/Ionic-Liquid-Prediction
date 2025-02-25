import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import src.data_preparation.descriptor
import src.data_preparation.preparation as preparation

# Function to train the stacking model
def train_stacking_model(X_train, y_train, best_xgb_params):
    estimators = [
        ('xgb', xgb.XGBRegressor(**best_xgb_params)),
        ('rf', RandomForestRegressor(n_estimators=100))
    ]
    final_estimator = LinearRegression()
    stacked_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    stacked_model.fit(X_train, y_train)
    return stacked_model

# Function to calculate residuals, RMSE, and other statistics
def calculate_residual_statistics(model, X, y):
    preds = model.predict(X)
    residuals = y - preds
    mse = mean_squared_error(y, preds)
    rmse = math.sqrt(mse)
    return rmse, preds, residuals

# Function to calculate confidence interval for a specific prediction
def calculate_confidence_interval(rmse, val_preds, residuals, prediction):
    segments = np.percentile(val_preds, [0, 25, 50, 75, 100])
    std_devs = []

    for i in range(len(segments) - 1):
        segment_mask = (val_preds >= segments[i]) & (val_preds < segments[i + 1])
        segment_residuals = residuals[segment_mask]
        if len(segment_residuals) > 0:
            std_devs.append(np.std(segment_residuals))
        else:
            std_devs.append(0)

    segment_index = np.digitize(prediction, segments) - 1
    if segment_index >= len(std_devs):
        segment_index = len(std_devs) - 1
    std_dev = std_devs[segment_index]

    confidence_interval = 1.96 * std_dev  # for 95% confidence
    return confidence_interval

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

# Function to evaluate the model
def evaluate_model(model, X, y, set_name="Validation"):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = math.sqrt(mse)
    print(f"{set_name} Mean Squared Error for Electrical Conductivity: {mse}")
    print(f"{set_name} Root Mean Squared Error for Electrical Conductivity: {rmse}")
    return preds, rmse

# Main function to train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    best_xgb_params = {
        'colsample_bytree': 0.7428214242463981,
        'learning_rate': 0.15220821760665343,
        'max_depth': 6,
        'n_estimators': 300,
        'subsample': 0.8321948719751764
    }

    stacked_model = train_stacking_model(X_train, y_train, best_xgb_params)
    print("Model trained successfully.")
    joblib.dump(stacked_model, "modelec.pkl")

    val_preds, val_rmse = evaluate_model(stacked_model, X_val, y_val, "Validation")
    residuals = y_val - val_preds

    test_preds, test_rmse = evaluate_model(stacked_model, X_test, y_test, "Test")
    print("------------------------------------------")
    print(f"Test Root Mean Squared Error for Electrical Conductivity in mS/cm: {test_rmse * 10}")

    return stacked_model, val_rmse, val_preds, residuals

# Load and prepare data
#df_cleaned_density = preparation.load_and_clean_data('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/electricalsmiled2.csv', "Delta elec")
#df_descriptors_cleaned_density = preparation.generate_and_clean_descriptors(df_cleaned_density)
#df_filtered_density = preparation.filter_data_ec(df_descriptors_cleaned_density)

#X_train_v, X_val_v, X_test_v, y_train_v, y_val_v, y_test_v = preparation.prepare_final_data(df_filtered_density, 'Electrical conductivity S/m')


# Train and evaluate the model
#model, rmse, val_preds, residuals = train_and_evaluate_model(X_train_v, y_train_v, X_val_v, y_val_v, X_test_v, y_test_v)
