import numpy as np
import pandas as pd
import scipy as sp
import tqdm
from sklearn.neighbors import BallTree

from .utils import getRandomArray

"""# Hopkins Statistic

The Hopkins statistic is a sparse test for spatial randomness. Given a dataset $\mathbf{D}$ comprising $n$ points, we 
generate $t$ random susamples $\mathbf{R}_{i}$ of $m$ points each, where $m<<n$. These samples are drawn from the same 
data space as $\mathbf{D}$, generated uniformly at random along each dimension. Further, we also generate $t$ subsamples 
of $m$ points directly from $\mathbf{D}$, using sampling without replacement.

Let $\mathbf{D}_{i}$ denote the $i$th direct subsample. Next, we compute the minimum distance between each points 
$\mathbf{x}_{j}\in \mathbf{D}_{i}$ and points in $\mathbf{D}$

$$\delta_{min}(\mathbf{x}_{j})=\min_{\mathbf{x}_{i}\in D, \mathbf{x}_{i}\neq \mathbf{x}_{j}}\left\{ \left\Vert \mathbf{x}_{j}-\mathbf{x}_{i} \right\Vert \right\}$$
Likewise, we compute the minimum distance $\delta_{min}(\mathbf{y}_{j})$ between a point 
$\mathbf{y}_{j}\in \mathbf{R}_{i}$ and points in $\mathbf{D}$.

The Hopkins statistic (in $d$ dimensions) for the $i$th pair of samples $\mathbf{R}_{i}$ and $\mathbf{D}_{i}$ is then defined as 
$$ HS_{i}=\frac{\Sigma_{\mathbf{y}_{j}\in \mathbf{R}_{i}} (\delta_{min}(\mathbf{y}_{j}))^d}{\Sigma_{\mathbf{y}_{j}\in \mathbf{R}_{i}}(\delta_{min}(\mathbf{y}_{j}))^d + \Sigma_{\mathbf{x}_{j}\in \mathbf{D}_{i}}(\delta_{min}(\mathbf{x}_{j}))^d} $$

This statistic compares the nearest-neighbors distribution of randomly generated points to the same distribution for 
random subsets of points from $\mathbf{D}$. If the data is well clustered we expect $\delta_{min}(\mathbf{x}_{j})$ 
values to be smaller compared to the $\delta_{min}(\mathbf{y}_{j})$  values, and in this case $HS_{i}$ tends to be 1. 
If both nearest-neighbor distances are similar, then $HS_{i}$ takes on values close to 0.5, which indicates that the 
data is essentially random, and there is no apparent clustering. Finally, if $\delta_{min}(\mathbf{x}_{j})$ values 
are larger compared to $\delta_{min}(\mathbf{y}_{j})$ values, then $HS_{i}$ tends to 0, and it indicates the point 
repulsion, with no clustering. From the $t$ different values of $HS_{i}$ we may then compute the mean and 
variance of the statistic to determne whether $\mathbf{D}$ is clusterable or not.
"""


def hopkins(data_frame, sampling_size):
    """Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express
    no clusterability and a score tending to 0 express a high cluster tendency.

    Parameters
    ----------
    data_frame : numpy array
        The input dataset
    sampling_size : int
        The sampling size which is used to evaluate the number of DataFrame.

    Returns
    ---------------------
    score : float
        The hopkins score of the dataset (between 0 and 1)

    Examples
    --------
    >>> from sklearn import datasets
    >>> from pyclustertend import hopkins
    >>> X = datasets.load_iris().data
    >>> hopkins(X,150)
    0.16
    """

    if type(data_frame) == np.ndarray:
        data_frame = pd.DataFrame(data_frame)

    # Sample n observations from D : P

    if sampling_size > data_frame.shape[0]:
        raise Exception(
            'The number of sample of sample is bigger than the shape of D')

    data_frame_sample = data_frame.sample(n=sampling_size)

    # Get the distance to their neirest neighbors in D : X

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(data_frame_sample, k=2)
    data_frame_sample_distances_to_nearest_neighbours = dist[:, 1]

    # Randomly simulate n points with the same variation as in D : Q.

    max_data_frame = data_frame.max()
    min_data_frame = data_frame.min()

    uniformly_selected_values_0 = np.random.uniform(min_data_frame[0], max_data_frame[0], sampling_size)
    uniformly_selected_values_1 = np.random.uniform(min_data_frame[1], max_data_frame[1], sampling_size)

    uniformly_selected_observations = np.column_stack((uniformly_selected_values_0, uniformly_selected_values_1))
    if len(max_data_frame) >= 2:
        for i in range(2, len(max_data_frame)):
            uniformly_selected_values_i = np.random.uniform(min_data_frame[i], max_data_frame[i], sampling_size)
            to_stack = (uniformly_selected_observations, uniformly_selected_values_i)
            uniformly_selected_observations = np.column_stack(to_stack)

    uniformly_selected_observations_df = pd.DataFrame(uniformly_selected_observations)

    # Get the distance to their neirest neighbors in D : Y

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(uniformly_selected_observations_df, k=1)
    uniformly_df_distances_to_nearest_neighbours = dist

    # return the hopkins score

    x = sum(data_frame_sample_distances_to_nearest_neighbours)
    y = sum(uniformly_df_distances_to_nearest_neighbours)

    if x + y == 0:
        raise Exception('The denominator of the hopkins statistics is null')

    return x / (x + y)[0]


def calculate_hopkins(_embs, _hbins):
    embs_hopkins = []
    for i in tqdm.tqdm(range(50)):
        embs_hopkins.append(hopkins(_embs, _hbins))
    return embs_hopkins, np.mean(embs_hopkins), np.std(embs_hopkins)


# Generate a subsample of m points directly from the given data set
def generateDirectSample(arr, m):
    '''
        input: arr is an numpy array of data points
               m: the size of direct sample without replaclement
        return: arr[idxs]: a direct sample of size m from the input numpy array
                idxs: the set of random indexes
    '''
    # number of input data points
    n_points = arr.shape[0]
    if m > n_points:
        raise Exception("The required sample size is too large.")
    
    idxs = np.random.choice(range(0, n_points), size=m, replace=False)

    return arr[idxs], idxs

# Compute the mininum distance from every point in arrA to arrB
def computeMinDistances(arrA, arrB, idxs=None):
    '''
        input: arrA a set of points in dimension d, typically shorter or equal to arrB
               arrB a set of points in dimension d, typically longer than arrA
               idxs: a set of indices in arrB which should not be included for computing minimum
        return: an array of minimum distances from each point in arrA to arrB
    '''
    dists = sp.spatial.distance.cdist(arrA, arrB)

    if idxs is not None:
        n_points = arrA.shape[0]  
        dists_ma = np.ma.array(dists, mask=False)
        for i in range(n_points):
            dists_ma[i, idxs[i]] = True

        return np.min(dists_ma, axis=1).data
    else:
        # return the minimum value of each row (the minimum distance from a point to arrB)
        return np.min(dists, axis=1)

# Compute Hopkins Statistics for a set of points
def calculate_hopkins_v2(arr, m):
    '''
        input: arr: a set of points in an numpy arrary in dimention d
               m: the size of sample for computing Hopkins Statistics
        return: Hopkins Statistics in (0, 1)
    '''
    Di, idxs = generateDirectSample(arr, m)
    Ri = getRandomArray(arr, m)

    dists_Di = computeMinDistances(Di, arr, idxs=idxs)
    dists_Ri = computeMinDistances(Ri,arr)

    dim = arr.shape[1]

    Ri_d_norm = np.sum(np.power(dists_Ri, dim))
    Di_d_norm = np.sum(np.power(dists_Di, dim))

    out = Ri_d_norm / (Ri_d_norm + Di_d_norm)
    print(out)
    return out


def measure_hopkins(embeddings, name=None):
    batch = int(.001 * len(embeddings)) #?
    return calculate_hopkins_v2(embeddings, batch)
    #eh, mu, st = calculate_hopkins(embeddings, batch)
    #plot_histogram(eh, 'sentence_embedding_hopkins', 'Sentence Embedding Space Hopkins Statistic')
    #return eh, mu, st
