import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, MolSurf


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES strings
    descriptors = {
        'AtomCount': mol.GetNumAtoms(),
        'BondCount': mol.GetNumBonds(),
        'ElementCount': len(set(atom.GetSymbol() for atom in mol.GetAtoms())),
        'HBondAcceptorCount': Descriptors.NumHAcceptors(mol),
        'HBondDonorCount': Descriptors.NumHDonors(mol),
        'HybridizationRatio_SP3': sum(1 for atom in mol.GetAtoms() if
                                      atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3) / mol.GetNumAtoms(),
        'RotatableBondCount': Descriptors.NumRotatableBonds(mol),
        'TopologicalPolarSurfaceArea': Descriptors.TPSA(mol),
        'ExactMolWeight': rdMolDescriptors.CalcExactMolWt(mol),
        'MolMR': Descriptors.MolMR(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'CarbonCount': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'),
        'RingCount': Descriptors.RingCount(mol),
        'LipinskiRuleOfFive': int(all([
            Descriptors.MolWt(mol) < 500,
            Descriptors.NumHAcceptors(mol) <= 10,
            Descriptors.NumHDonors(mol) <= 5,
            Descriptors.MolLogP(mol) < 5
        ])),
        'VeberRule': int(all([
            Descriptors.TPSA(mol) <= 140,
            Descriptors.NumRotatableBonds(mol) <= 10
        ])),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NHOHCount': Descriptors.NHOHCount(mol),
        'NOCount': Descriptors.NOCount(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
        'ALogP': Descriptors.MolLogP(mol),
        'XLogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'PolarSurfaceArea': Descriptors.TPSA(mol),
        'MolecularVolume': Descriptors.MolMR(mol),
        'BalabanIndex': Descriptors.BalabanJ(mol),
        'KierHallAlpha': Descriptors.Kappa1(mol)

    }
    return descriptors


def process_ionic_liquid(smiles):
    try:
        cation_smiles, anion_smiles = smiles.split('.')
    except ValueError:
        return None
    cation_descriptors = calculate_descriptors(cation_smiles)
    anion_descriptors = calculate_descriptors(anion_smiles)
    if cation_descriptors and anion_descriptors:
        combined_descriptors = {}
        for key in cation_descriptors:
            combined_descriptors[f'Cation_{key}'] = cation_descriptors[key]
            combined_descriptors[f'Anion_{key}'] = anion_descriptors[key]
        return combined_descriptors
    return None