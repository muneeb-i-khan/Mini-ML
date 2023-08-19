import numpy as np

def PCA(x,num_comp):
    mean = np.mean(x, axis = 0)
    x_mean = x - mean

    cov_matrix = np.cov(x_mean, rowvar=False)

    eigen_val, eigen_vec = np.linalg.eigh(cov_matrix)

    sort_idx = np.argsort(eigen_val)[::-1]
    sort_eigenval = eigen_val[sort_idx]
    sort_eigenvec = eigen_vec[:,sort_idx]

    final_set = sort_eigenvec[:,0:num_comp]
    x_final = np.dot(final_set.transpose(),x_mean.transpose()).transpose()

    return x_final


