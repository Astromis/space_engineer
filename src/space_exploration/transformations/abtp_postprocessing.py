import numpy as np
from sklearn.decomposition import PCA

def all_but_the_top(v, D):
      """
      Arguments:
          :v: word vectors of shape (n_words, n_dimensions)
          :D: number of principal components to subtract
      """
      # 1. Subtract mean vector
      v_tilde = v - np.mean(v, axis=0)
      # 2. Compute the first `D` principal components
      #    on centered embedding vectors
      u = PCA(n_components=D).fit(v_tilde).components_  # [D, emb_size]
      # Subtract first `D` principal components
      # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
      return v_tilde - (v @ u.T @ u)  