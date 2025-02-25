import pandas as pd
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import src.model.viscosity.nn_viscosity as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def iupac_to_smiles(iupac_name):
    url = f'https://opsin.ch.cam.ac.uk/opsin/{iupac_name}.json'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('smiles', "")
    except requests.RequestException as e:
        print(f"Erreur lors de la requête à l'API: {e}")
        return ""


import pandas as pd
import requests
import src.data_preparation.descriptor as desc

def iupac_to_smiles(iupac_name):
    url = f'https://opsin.ch.cam.ac.uk/opsin/{iupac_name}.json'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('smiles', "")
    except requests.RequestException as e:
        print(f"Erreur lors de la requête à l'API: {e}")
        return ""

def prepare_data(iupac_name, temperature, pressure, property_value, property_type):
    smiles = iupac_to_smiles(iupac_name)
    if not smiles:
        print("Conversion to SMILES failed.")
        return pd.DataFrame()

    data = {
        "Components": [iupac_name],
        "SMILES": [smiles],
    }

    if property_type == 'viscosity':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Viscosity Pa/s"] = [property_value * 0.001]  # Convert mPa.s to Pa.s
        data["DeltaViscosity"] = [0]
    elif property_type == 'density':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Specific density kg/m3"] = [property_value]  # Density in kg/m3
    elif property_type == 'electrical_conductivity':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Electrical conductivity S/m"] = [property_value]  # Conductivity in S/m
    elif property_type == 'melting_temperature':
        data["Melting Temperature"] = [property_value]  # Melting Temperature in K

    df = pd.DataFrame(data)
    df_descriptors = df['SMILES'].apply(desc.process_ionic_liquid)
    valid_mask = df_descriptors.notnull()
    df = df[valid_mask].reset_index(drop=True)
    df_descriptors = pd.DataFrame(list(df_descriptors.dropna()))

    df = pd.concat([df, df_descriptors], axis=1)

    df = df.drop(columns=["SMILES", "Components"])
    df.drop('Cation_LipinskiRuleOfFive', inplace=True, axis=1)

    if property_type == 'viscosity':
        df = df.drop(columns=["DeltaViscosity"])

    return df

def prepare_data_nn(iupac_name, temperature, pressure, property_value, property_type):
    smiles = iupac_to_smiles(iupac_name)
    if not smiles:
        print("Conversion to SMILES failed.")
        return pd.DataFrame()

    data = {
        "Components": [iupac_name],
        "SMILES": [smiles],
    }

    if property_type == 'viscosity':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Viscosity Pa/s"] = [property_value * 0.001]  # Convert mPa.s to Pa.s
        data["DeltaViscosity"] = [0]
    elif property_type == 'density':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Specific density kg/m3"] = [property_value]  # Density in kg/m3
    elif property_type == 'electrical_conductivity':
        data["Temperature K"] = [temperature]
        data["Pressure kPa"] = [pressure]
        data["Electrical conductivity S/m"] = [property_value]  # Conductivity in S/m
    elif property_type == 'melting_temperature':
        data["Melting Temperature"] = [property_value]  # Melting Temperature in K

    df = pd.DataFrame(data)

    # Process descriptors
    df_descriptors = df['SMILES'].apply(desc.process_ionic_liquid)
    valid_mask = df_descriptors.notnull()
    df = df[valid_mask].reset_index(drop=True)
    df_descriptors = pd.DataFrame(list(df_descriptors.dropna()))

    df = pd.concat([df, df_descriptors], axis=1)

    # Extract numeric columns and process SMILES
    df_numeric = df.select_dtypes(include=[np.number])
    df_smiles = nn.tokenize_smiles(df["SMILES"].tolist())

    # Apply scaling
    df_numeric_scaled = nn.scaler.transform(df_numeric)


    # Drop unnecessary columns
    df = df.drop(columns=["SMILES", "Components", 'Cation_LipinskiRuleOfFive'])
    if property_type == 'viscosity':
        df = df.drop(columns=["DeltaViscosity"])

    return df



# Test with given data
iupac_name = "1-Decyl-3-methylimidazolium bis(trifluoromethylsulfonyl)imide"
temperature = 308.15
pressure = 101.325
viscosity = 63.29

df_prepared_viscosity = prepare_data(iupac_name, temperature, pressure, viscosity, 'viscosity')

density = 1100  # Example density value
df_prepared_density = prepare_data(iupac_name, temperature, pressure, density, 'density')

print("Prepared Data for Viscosity Prediction:")
print(df_prepared_viscosity)
print("\nPrepared Data for Density Prediction:")
print(df_prepared_density)
