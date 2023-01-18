import numpy as np
import seaborn as sns
#from skdim.id import *
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal
# from scipy.stats import uniform
from sklearn.decomposition import PCA


def plot_histogram(data, name, title,  bins=20):
    """
    Plots рistograms for a set of  embeddings.
    :param name: Graphic name.
    :param kls: data to plot.
    :return: None, plots.
    """
    plt.figure(figsize=(12, 8))
    sns.distplot(data, hist=False, label=str(name))
    plt.legend()
    
    plt.title(title)
    
    plt.savefig(f"{title}-{name}-{bins}.png" 'sentence_embedding_clusterability_{b}.png'.format(b=bins))

def sample_gaussian(num_points, dim, mean, cov, uniform_mean=True, uniform_cov=True): 
    """INPUTS:
       
       num_points: number of points in the sample. 
       dim: dimensionality of gaussian we are sampling from.
       mean: if uniform_mean==True then a scalar. Otherwise a dim dimensional vector of the mean.
       cov: if uniform_cov == True then a scalar to produce scaled identity for covariance. Otherwise custom dim x dim covariance matrix.
       
       RETURNS: 
       matrix of dim x num_points sampled from a multivariate_gaussian."""	
           
    # Create a unform mean and covariance:
    if uniform_mean == True:	
        mean = np.full((dim), mean)
    if uniform_cov == True:	
        cov = np.eye(dim)*cov
        
    # Use customized mean and covariance:	
    if uniform_mean == False:
        mean = mean	
    if uniform_cov == True:
        cov = cov	
    samples = multivariate_normal(mean=mean, cov=cov, size=num_points)	
    return samples.T 	 

def pca_normalization(points):
    """Projects points onto the directions of maximum variance.""" 
    points = np.transpose(points)	
    pca = PCA(n_components=len(np.transpose(points)))
    points = pca.fit_transform(points)	
    return np.transpose(points)
    
def skewered_meatball(dim, num_gauss, num_line):
    """Intersect points sampled from a multivariate Gaussian and a line in n dimensional space.""" 
    gauss = sample_gaussian(num_points=num_gauss, dim=dim, cov=1, mean=0)
    cov = np.full((dim,dim),1)
    line = sample_gaussian(num_points=num_line, dim=dim, mean=0, cov=cov, uniform_cov=False)
    points = np.hstack((line,gauss))
    return points

# computes closed-form expression for IsoScore of I_n^{(k)}
def map_k_to_Iso_Score(n, k):
    return 1-np.sqrt(n-np.sqrt(n*k)) / np.sqrt(n-np.sqrt(n))

# computes closed form expression for fxn which maps IsoScore to number of dimensions utilized
def map_Iso_Score_to_k(iota, n):
    return iota*(n-1) + 1