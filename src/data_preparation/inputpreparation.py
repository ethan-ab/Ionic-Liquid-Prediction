import pandas as pd
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import data_preparation.descriptor as desc


def iupac_to_smiles(iupac_name):
    url = f'https://opsin.ch.cam.ac.uk/opsin/{iupac_name}.json'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('smiles', "")
    except requests.RequestException as e:
        print(f"Erreur lors de la requête à l'API: {e}")
        return ""


def prepare_data(iupac_name, temperature, pressure, viscosity):
    smiles = iupac_to_smiles(iupac_name)
    if not smiles:
        print("Conversion to SMILES failed.")
        return pd.DataFrame()

    data = {
        "Components": [iupac_name],
        "Temperature K": [temperature],
        "Pressure kPa": [pressure],
        "Viscosity Pa/s": [viscosity * 0.001],
        "DeltaViscosity": [0],
        "SMILES": [smiles],
    }

    df = pd.DataFrame(data)
    df_descriptors = df['SMILES'].apply(desc.process_ionic_liquid)
    valid_mask = df_descriptors.notnull()
    df = df[valid_mask].reset_index(drop=True)
    df_descriptors = pd.DataFrame(list(df_descriptors.dropna()))

    df = pd.concat([df, df_descriptors], axis=1)

    df = df.drop(columns=["SMILES", "DeltaViscosity", "Components"])
    df.drop('Cation_LipinskiRuleOfFive', inplace=True, axis=1)

    return df


# Test with given
iupac_name = "1-Decyl-3-methylimidazolium bis(trifluoromethylsulfonyl)imide"
temperature = 308.15
pressure = 101.325
viscosity = 63.29

df_prepared = prepare_data(iupac_name, temperature, pressure, viscosity)




