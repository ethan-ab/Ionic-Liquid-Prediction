import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense
import src.data_preparation.input_preparation as inputprep
import src.data_preparation.preparation as preparation
import matplotlib.pyplot as plt
import os


def train_autoencoder(df_cleaned, model_save_path='autoencoder_model.h5'):

    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    data_numeric = df_cleaned[numeric_columns]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    input_dim = data_scaled.shape[1]
    encoding_dim = 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')


    autoencoder.fit(data_scaled, data_scaled, epochs=100, batch_size=32, shuffle=True, validation_split=0.1)

    autoencoder.save(model_save_path)


def get_cleaned_df_with_autoencoder(df_cleaned, model_save_path='autoencoder_model.h5'):
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(
            f"Le modèle {model_save_path} n'a pas été trouvé. Veuillez d'abord entraîner le modèle.")

    autoencoder = load_model(model_save_path)


    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    data_numeric = df_cleaned[numeric_columns]


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    reconstructions = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 90)

    outliers = mse > threshold

    df_cleaned_no_outliers = df_cleaned[~outliers]


    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data_scaled)), mse, c=outliers, cmap='coolwarm', alpha=0.6)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error for All Columns Data')
    plt.show()

    return df_cleaned_no_outliers

df_cleaned = preparation.load_and_clean_data('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/IlThermo-smiled_dataset_viscosity.csv', "DeltaViscosity")
train_autoencoder(df_cleaned)