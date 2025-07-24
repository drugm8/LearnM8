"""Query functions for active learning compound selection in molecular screening."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
import multiprocessing as mp
from itertools import combinations
from functools import partial

# Constants for clustering and similarity calculations
MAX_COMPOUNDS_FOR_CLUSTERING = 35000
DEFAULT_CLUSTERING_SEED = 42
MAX_COMPOUNDS_PER_CLUSTER = 10
CLUSTER_SAMPLING_FRACTION = 0.5
TANIMOTO_SIMILARITY_CUTOFF = 0.4


def greedy_query_function(compound_data, batch_size, seed, score_direction='higher'):
    """
    Greedy query function that selects the top compounds based on their estimated values.
    
    Selects compounds with the best 'estimation' values according to the scoring direction,
    representing an exploitation strategy in active learning for molecular screening.
    
    Args:
        compound_data (pd.DataFrame): Input DataFrame containing compounds with columns:
                                    - 'ID': Compound identifier
                                    - 'SMILES': Molecular structure representation
                                    - 'estimation': Predicted score/value for ranking
        batch_size (int): Number of compounds to select from the pool
        seed (int): Random seed for reproducibility (not used in greedy selection)
        score_direction (str): 'higher' for higher-is-better scores, 'lower' for lower-is-better scores
        
    Returns:
        pd.DataFrame: DataFrame containing the top `batch_size` compounds sorted by 
                     their 'estimation' values according to score_direction
    """
    input_compounds = compound_data.copy()
    
    # Sort compounds by estimation values according to score direction
    # For 'lower' direction (e.g., docking scores), ascending=True selects lowest values first
    # For 'higher' direction (e.g., activity scores), ascending=False selects highest values first
    ascending_order = (score_direction == 'lower')
    sorted_compounds = input_compounds.sort_values(by='estimation', ascending=ascending_order)
    
    # Select top compounds based on batch size
    selected_compounds = sorted_compounds.head(batch_size)
    
    return selected_compounds


def random_query_function(compound_data, batch_size, seed, score_direction='higher'):
    """
    Random query function that randomly selects compounds from the available pool.
    
    Implements exploration strategy for active learning by randomly sampling compounds,
    ensuring diversity in compound selection across iterations.
    
    Args:
        compound_data (pd.DataFrame): Input DataFrame containing compounds with columns:
                                    - 'ID': Compound identifier  
                                    - 'SMILES': Molecular structure representation
        batch_size (int): Number of compounds to randomly select
        seed (int): Random seed for reproducible random sampling
        score_direction (str): Score direction (not used in random selection, kept for consistency)
        
    Returns:
        pd.DataFrame: DataFrame containing `batch_size` randomly selected compounds
    """
    # Shuffle all compounds using the provided seed for reproducibility
    shuffled_compounds = compound_data.sample(frac=1, random_state=seed)
    
    # Select the first batch_size compounds from shuffled data
    selected_compounds = shuffled_compounds.head(batch_size)
    
    return selected_compounds


def cluster_query_function(compound_data, estimation, batch_size):
    """
    Cluster-based query function for diversity-oriented compound selection.
    
    Uses molecular fingerprints and Butina clustering to select diverse compounds,
    ensuring chemical diversity in the selected batch. Includes memory management
    for large compound libraries.
    
    Note: This function is currently unused in the active learning pipeline but
    maintained for potential future diversity-based selection strategies.
    
    Args:
        compound_data (pd.DataFrame): Input DataFrame containing compounds with columns:
                                    - 'ID': Compound identifier
                                    - 'SMILES': Molecular structure representation
        estimation: Estimated values (currently unused in clustering logic)
        batch_size (int): Number of compounds to select through clustering
        
    Returns:
        list: List of compound indices selected through clustering
    """
    input_compounds = compound_data.copy()
    
    # Apply memory constraint for large compound libraries
    num_compounds = len(input_compounds["SMILES"].values)
    if num_compounds > MAX_COMPOUNDS_FOR_CLUSTERING:
        sampling_fraction = MAX_COMPOUNDS_FOR_CLUSTERING / num_compounds
        input_compounds = input_compounds.sample(
            frac=sampling_fraction, 
            random_state=DEFAULT_CLUSTERING_SEED
        )
    
    print(f"Processing {len(input_compounds['SMILES'].values)} compounds for clustering")
    
    # Generate RDKit fingerprints for molecular similarity calculation
    fingerprint_generator = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = []
    
    for smiles in input_compounds['SMILES']:
        molecule = Chem.MolFromSmiles(smiles)
        molecular_fingerprint = fingerprint_generator.GetFingerprint(molecule)
        fingerprints.append(molecular_fingerprint)
    
    # Perform Butina clustering on molecular fingerprints
    compound_clusters = cluster_fingerprints(fingerprints)
    
    # Sort clusters by size (largest first) for systematic selection
    sorted_clusters = sorted(compound_clusters, key=len, reverse=True)
    
    # Initialize selection with cluster centers (first molecule from each cluster)
    cluster_centers = [[cluster[0]] for cluster in sorted_clusters]
    selected_molecules = cluster_centers.copy()
    
    # If we have more clusters than requested batch size, return top cluster centers
    if batch_size < len(selected_molecules):
        selected_molecules = selected_molecules[:batch_size]
        return selected_molecules
    
    # Fill remaining slots by sampling from clusters
    remaining_slots = batch_size - len(selected_molecules)
    cluster_index = 0
    
    while remaining_slots > 0 and cluster_index < len(sorted_clusters):
        current_cluster = sorted_clusters[cluster_index]
        
        # Determine number of compounds to sample from current cluster
        if len(sorted_clusters[cluster_index]) > MAX_COMPOUNDS_PER_CLUSTER:
            compounds_to_sample = MAX_COMPOUNDS_PER_CLUSTER
        else:
            compounds_to_sample = int(CLUSTER_SAMPLING_FRACTION * len(current_cluster)) + 1
            
        # Don't exceed remaining slots
        if compounds_to_sample > remaining_slots:
            compounds_to_sample = remaining_slots
            
        # Add selected compounds from current cluster
        selected_molecules += [i for i in current_cluster[:compounds_to_sample]]
        
        cluster_index += 1
        remaining_slots = batch_size - len(selected_molecules)
    
    return selected_molecules


def _tanimoto_distance_matrix(fingerprint_list):
    """
    Calculate Tanimoto distance matrix for molecular fingerprint clustering.
    
    Private helper function that computes pairwise Tanimoto distances between
    molecular fingerprints for use in Butina clustering algorithm.
    
    Args:
        fingerprint_list (list): List of RDKit molecular fingerprints
        
    Returns:
        list: Flattened distance matrix for clustering algorithm
    """
    dissimilarity_matrix = []
    
    # Calculate pairwise similarities (skip diagonal and upper triangle)
    for i in range(1, len(fingerprint_list)):
        # Compare current fingerprint against all previous ones
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint_list[i], 
            fingerprint_list[:i]
        )
        # Convert similarities to distances (1 - similarity)
        distances = [1 - similarity for similarity in similarities]
        dissimilarity_matrix.extend(distances)
        
    return dissimilarity_matrix


def cluster_fingerprints(fingerprints, similarity_cutoff=TANIMOTO_SIMILARITY_CUTOFF):
    """
    Cluster molecular fingerprints using Butina clustering algorithm.
    
    Groups molecules based on their structural similarity using Tanimoto distance
    and the Butina clustering method, which ensures cluster centers are maximally
    distant from each other.
    
    Args:
        fingerprints (list): List of RDKit molecular fingerprints to cluster
        similarity_cutoff (float): Tanimoto similarity threshold for clustering.
                                  Lower values create more clusters with higher similarity
                                  
    Returns:
        list: List of clusters, where each cluster is a list of compound indices
              sorted by cluster size (largest first)
    """
    # Calculate Tanimoto distance matrix for all fingerprint pairs
    distance_matrix = _tanimoto_distance_matrix(fingerprints)
    
    # Apply Butina clustering algorithm
    compound_clusters = Butina.ClusterData(
        distance_matrix, 
        len(fingerprints), 
        similarity_cutoff, 
        isDistData=True
    )
    
    # Sort clusters by size (largest first) for consistent selection
    sorted_clusters = sorted(compound_clusters, key=len, reverse=True)
    
    return sorted_clusters