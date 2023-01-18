import math

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import entropy
from sklearn.decomposition import PCA
from .utils import getRandomArray

def _computeKLConv(epmf_input, epmf_rand):
    """Compute the KL divergence between two given EPMF
    """
    return entropy(epmf_input, epmf_rand, base=2)

def _getEPMF(arr, bins=20, smoothing=False):
    '''Compute an empirical probability mass function for given array of point
        input: arr: an numpy array of d dimension points
               bins: number of bins for computing the EPMF 
        return: an numpy array of EPMF values in the cells binned along arr's dimensions
    '''
    dims = arr.shape[1]

    # If smoothing is needed, initialize all counts with 1. 
    ans = np.zeros(int(math.pow(bins, dims)))
    if smoothing == True:
        ans = np.ones(int(math.pow(bins, dims)))
    
    # cut each dimension into bins with labels of bin indexes
    cats = np.zeros(arr.shape)
    for i in range(dims):
        cats[:, i] = pd.cut(arr[:, i], bins=bins, labels=range(0, bins))
    
    # Compute the index of the EPMF array using the 
    # category numbers of each point in the input array
    for i in range(arr.shape[0]):
        idx = 0
        for j in range(dims):
            pow = dims - 1 - j
            idx = idx + cats[i, j] * math.pow(bins, pow)
        ans[int(idx)] = ans[int(idx)] + 1 # update the counts at the cell indexed by idx


    return ans / sum(ans)




def calculate_spation_histogram(arr, bins=20, n = 500):
    '''Spatial Histogram for Clustering Tendency
        advice: try bigger bins:  10, 20, 30, 50, 100

        input: arr: an numpy array of input data in d dimension
               bins: the number of bins for computing Estimated Probability Mass Function along dimensions
               n: number of random instances for comparison
        return:  an numpy array of n KL divergence numbers between the EPMF of the 
                 input arr and the EPMFs of n randomly generated arrays 
    '''
    
    ans = np.zeros(n)

    # Compute the Estimated Probability Mass Function for the input array along all its dimensions
    # the second paramter number is for the number of binning on a dimension
    epmf_input = _getEPMF(arr, bins)

    for i in tqdm.tqdm(range(n)):
        aRandArr = getRandomArray(arr, arr.shape[0])
        epmf_rand = _getEPMF(aRandArr, bins)

        kl_conv = _computeKLConv(epmf_input, epmf_rand)
        ans[i] = kl_conv

    # Replace the 'inf' KL divergence value with the mean  
    kls_vals = np.where(np.isinf(ans), np.mean(ans[np.isfinite(ans)]), ans)

    return kls_vals


def apply_spatial_historgram(_embs, _bins=20):
    """
    Applied the spatial histogram clusterability metric.
    :param _embs: A set of (2D) PCA projected embeddings.
    :return: Spatial histogram information.
    """
    kls_embs = calculate_spation_histogram(_embs, bins=_bins, n=50)
    mu_kls = kls_embs.mean()
    std_kls = kls_embs.std()
    print('Spatial histogram:  Mu: {m}, sigma: {s}'.format(m=mu_kls, s=std_kls))
    return kls_embs, mu_kls, std_kls


def measure_spatial_histogram(embeddings, name=None):
    pca_projected = PCA(n_components=2).fit_transform(embeddings)
    kls, mu, std = apply_spatial_historgram(pca_projected)
    #plot_histogram(kls, 'sentence_embedding_clusterability', 'Sentence Embedding Space Spatial Histogram')
    return kls, mu, std
