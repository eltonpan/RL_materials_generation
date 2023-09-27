import pandas as pd
import numpy as np
from tqdm import tqdm
from ElM2D import ElM2D

def similarity_to_nearest_neighbor(unique_mats, metric='mod_petti'):
    '''
    Computes average Element Mover's distance between any two materials in a set of generated materials.
    
    Args:
    unique_mats: a list of valid unique generated materials. shape = (num_unique,)
    NOTE: unique_mats should be standardized before passing into this function,
    but we check this anyways here
    Returns:
    mean_distnace: mean Element Mover's distance between any pair of unique compounds in unique_mats
    std_distance: std Element Mover's distance between any pair of unique compounds in unique_mats
    '''
    unique_mats = list(set(unique_mats))
    mapper = ElM2D(metric)
    mapper.fit(unique_mats)
    distances = []
    for i in tqdm(range(len(unique_mats))):
        for j in range(i+1, len(unique_mats)):
            distances.append(mapper.dm[i][j])
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    print(f"Average Element Mover's distance: {round(mean_distance, 4)}")
    print(f"Std Element Mover's distance: {round(std_distance, 4)}")
    return mean_distance, std_distance