import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from joblib import Parallel, delayed
import time

# def convert_list_of_smiles_to_morgan_fingerprints(smiles_list):
#         # Convert SMILES to RDKit molecules
#         molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
#         # Create a Morgan fingerprint generator
#         morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
#         # Generate Morgan fingerprints
#         fingerprints = [morgan_gen.GetFingerprint(mol) for mol in molecules]
    
#         # Convert fingerprints to a NumPy array of bit vectors
#         fingerprints_array = np.array([np.array(fp) for fp in fingerprints])
        
#         return fingerprints_array

def convert_smiles_to_morgan_fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = morgan_gen.GetFingerprint(molecule)
    return np.array(fingerprint)

def convert_list_of_smiles_to_morgan_fingerprints(smiles_list, n_jobs=-1):
    # n_jobs=-1 nutzt alle verfügbaren Kerne
    fingerprints = Parallel(n_jobs=n_jobs)(delayed(convert_smiles_to_morgan_fingerprint)(smiles) for smiles in smiles_list)
    return np.array(fingerprints)

def remove_right_df_from_left_df(left_df, right_df):
        return left_df[~left_df['SMILES'].isin(right_df['SMILES'])]

def log_and_save(message,log_file):
    log_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
    #log_file.write(message + "\n")
    log_file.flush()  # Ensure it's written to the file immediately

def log_list(list_data,log_file):
    message = "["
    for i in range(len(list_data)):
        message += str(list_data[i])
        message += ", "
    message += "]"
    log_and_save(message,log_file)     

