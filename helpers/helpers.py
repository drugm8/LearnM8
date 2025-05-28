import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from joblib import Parallel, delayed
import time
import os
import json
import hashlib


def convert_smiles_to_morgan_fingerprint(smiles):
    """
    Converts a SMILES string to a Morgan fingerprint.
    Args:
        smiles (str): The SMILES string of the molecule.
    Returns:
        np.ndarray: The Morgan fingerprint of the molecule. 
    """
    molecule = Chem.MolFromSmiles(smiles)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = morgan_gen.GetFingerprint(molecule)
    return np.array(fingerprint)

def convert_list_of_smiles_to_morgan_fingerprints(smiles_list, n_jobs=-1):
    """
    Converts a list of SMILES strings to Morgan fingerprints using parallel processing.
    Args:
        smiles_list (list): A list of SMILES strings.
        n_jobs (int): The number of jobs to run in parallel. Default is -1, which uses all available cores.
    Returns:
        np.ndarray: An array of Morgan fingerprints corresponding to the input SMILES strings.
    """
    if os.cpu_count() > 32: #safeguard for hpc cluster
        n_jobs = 32
    else:
        n_jobs = os.cpu_count()
    # n_jobs=-1 nutzt alle verfügbaren Kerne
    fingerprints = Parallel(n_jobs=n_jobs)(delayed(convert_smiles_to_morgan_fingerprint)(smiles) for smiles in smiles_list)
    return np.array(fingerprints)

def remove_right_df_from_left_df(left_df, right_df):
        return left_df[~left_df['ID'].isin(right_df['ID'])]

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

def initialize_logging(script_name):
    filename = os.path.basename(script_name).split(".")[0]
    if not os.path.exists("./runs/"+filename):
        os.makedirs("./runs/"+filename)
    log_file_path = "./runs/"+filename+"/"+"log_"+str(time.strftime("%Y-%m-%d %H:%M:%S"))+".txt"
    return open(log_file_path, "w")

def hash_params(dictionary):
    # Sort the dictionary by keys to ensure consistent ordering
    sorted_dict = dict(sorted(dictionary.items()))
    
    # Convert the sorted dictionary to a JSON string
    json_string = json.dumps(sorted_dict, sort_keys=True)
    
    # Create a hash object (using SHA-256 in this example)
    hash_object = hashlib.sha256()
    
    # Update the hash object with the JSON string encoded as UTF-8
    hash_object.update(json_string.encode('utf-8'))
    
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()