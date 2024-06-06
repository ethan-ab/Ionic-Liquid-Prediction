import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import src.data_preparation.descriptor


# Chargement et nettoyage initial des données
# Input : chemin du fichier CSV
# Output : DataFrame nettoyée
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(["DeltaViscosity"], axis=1)
    df_cleaned = df.dropna()
    return df_cleaned


# Génération et nettoyage des descripteurs
# Input : DataFrame nettoyée
# Output : DataFrame nettoyée avec descripteurs
def generate_and_clean_descriptors(df_cleaned):
    df_descriptors = df_cleaned['SMILES'].apply(src.data_preparation.descriptor.process_ionic_liquid)
    valid_mask = df_descriptors.notnull()
    df_cleaned = df_cleaned[valid_mask].reset_index(drop=True)
    df_descriptors = pd.DataFrame(list(df_descriptors.dropna()))
    df_cleaned = pd.concat([df_cleaned, df_descriptors], axis=1)
    df_cleaned = df_cleaned.drop(["SMILES"], axis=1)
    df_cleaned.dropna(inplace=True)
    return df_cleaned


# Filtrage des données selon la température, pression et viscosité
# Input : DataFrame nettoyée avec descripteurs
# Output : DataFrame filtrée
def filter_data(df_cleaned):
    df_cleaned["Viscosity Pa/s"] = df_cleaned["Viscosity Pa/s"]
    temperature_filtered_df = df_cleaned[
        (df_cleaned['Temperature K'] >= 275) & (df_cleaned['Temperature K'] <= 400)
        ]
    pressure_filtered_df = temperature_filtered_df[
        (temperature_filtered_df['Pressure kPa'] >= 97) & (temperature_filtered_df['Pressure kPa'] <= 103)
        ]
    viscosity_filtered_df = pressure_filtered_df[
        (pressure_filtered_df['Viscosity Pa/s'] <= 0.140) & (pressure_filtered_df['Viscosity Pa/s'] >= 0.008)
        ]
    return viscosity_filtered_df.copy()


# Préparation des données finales pour l'entraînement
# Input : DataFrame filtrée
# Output : X, y et DataFrames pour l'entraînement et les tests
def prepare_final_data(df_filtered):
    df_filtered.drop("Components", axis=1, inplace=True)
    df_filtered.drop('Cation_LipinskiRuleOfFive', inplace=True, axis=1)
    df_filtered = df_filtered.select_dtypes(include=['number'])

    X = df_filtered.drop(columns=['Viscosity Pa/s'])
    y_viscosity = np.log(df_filtered['Viscosity Pa/s'])

    X_train_v, X_temp_v, y_train_v, y_temp_v = train_test_split(X, y_viscosity, test_size=0.2, random_state=42)
    X_val_v, X_test_v, y_val_v, y_test_v = train_test_split(X_temp_v, y_temp_v, test_size=0.5, random_state=42)

    return X_train_v, X_val_v, X_test_v, y_train_v, y_val_v, y_test_v

