import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
import multiprocessing as mp
from itertools import combinations
import numpy as np
from functools import partial

def greedy_query_function(x_i, batch_size, seed):
    x_input = x_i.copy()

    sorted_df = x_input.sort_values(by='estimation', ascending=False)
    #print(sorted_df)
    queried_df = sorted_df.head(batch_size)
    return queried_df


def random_query_function(x_input, batch_size, seed):
    dataset = x_input.sample(frac=1, random_state=42)#!CHANGE ME  BACK

    return dataset.head(batch_size)

#def cluster_query_function(x_input,estimation, batch_size):


def cluster_query_function(x_input,estimation, batch_size):

    ##!RAM CONSTRAINT
    _len = len(x_input["SMILES"].values)
    if _len > 35000:
        frac = 35000/_len
        x_input = x_input.sample(frac=frac, random_state=42)
    ##!END
    print("len x_input", len(x_input["SMILES"].values))


    # Create a Morgan fingerprint generator
    #morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)#!maybe dont use this because different similarity meassure bc of length
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fps = []

    for smiles in x_input['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        #fps.append(morgan_gen.GetFingerprint(mol))
        fps.append(rdkit_gen.GetFingerprint(mol))

    #print("done calculating fingerprints")
    clusters = cluster_fingerprints(fps)
    # Give a short report about the numbers of clusters and their sizes
    #num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    #num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    #num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    #num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

    #print("total # clusters: ", len(clusters))
    #print("# clusters with only 1 compound: ", num_clust_g1)
    #print("# clusters with >5 compounds: ", num_clust_g5)
    #print("# clusters with >25 compounds: ", num_clust_g25)
    #print("# clusters with >100 compounds: ", num_clust_g100)


    #sort clusters
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    # Get the cluster center of each cluster (first molecule in each cluster)
    #print("len clusters beg", len(sorted_clusters[0]))
    #print("len cluster end", len(sorted_clusters[-1]))
    cluster_centers = [[c[0]] for c in sorted_clusters]
    # How many cluster centers/clusters do we have?
    #print("Number of cluster centers:", len(cluster_centers))
    # NBVAL_CHECK_OUTPUT
    selected_molecules = cluster_centers.copy()
    index = 0
    if batch_size < len(selected_molecules):
        selected_molecules = selected_molecules[:batch_size]
        #print("since the number of clustes was higher than te requested batch sizes, one of each of the sorted cluster centers is returned")
        return selected_molecules

    pending = batch_size - len(selected_molecules)
    while pending > 0 and index < len(sorted_clusters):
        # Take indices of sorted clusters
        tmp_cluster = sorted_clusters[index]
        # If the first cluster is > 10 big then take exactly 10 compounds
        if len(sorted_clusters[index]) > 10:
            num_compounds = 10
        # If smaller, take half of the molecules
        else:
            num_compounds = int(0.5 * len(tmp_cluster)) + 1
        if num_compounds > pending:
            num_compounds = pending
        # Write picked molecules and their structures into list of lists called picked_fps
        selected_molecules += [i for i in tmp_cluster[:num_compounds]]
        index += 1
        pending = batch_size - len(selected_molecules) #!why
    #print("# Selected molecules:", len(selected_molecules))
    return selected_molecules

def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def cluster_fingerprints(fingerprints, cutoff=0.4):#TODO cutoff has high impact on cluster variance
    """Cluster fingerprints
    Parameters:
        fingerprints
        cutoff: threshold for the clustering
    """
    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    #print("done calculating distance matrix")
    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters