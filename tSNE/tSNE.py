import numpy as np

def distance(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]
    return distances

def cond_prob(distances, perplexity=30.0):
    n = distances.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        tol = 1e-5
        target_entropy = np.log(perplexity)
        
        distances_i = distances[i, np.arange(n) != i]
        pij = np.exp(-beta * distances_i)
        sum_pij = np.sum(pij)
        pij /= sum_pij
        
        while True:
            entropy = -np.sum(pij * np.log2(pij + 1e-7))  
            entropy_diff = entropy - target_entropy
            
            if np.abs(entropy_diff) < tol:
                break
            
            if entropy_diff > 0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2
            
            pij = np.exp(-beta * distances_i)
            sum_pij = np.sum(pij)
            pij /= sum_pij
        
        P[i, np.arange(n) != i] = pij
    
    return P

def KL_divergence(P, Q):

    kl_div = np.sum(P * np.log2(P / (Q + 1e-7) + 1e-7))
    return kl_div

def tSNE(X, num_dimensions=2, perplexity=30.0, num_iterations=1000, learning_rate=200.0, tolerance=1e-5):
    n, d = X.shape
    distances = distance(X)
    P = cond_prob(distances, perplexity)
    
    
    Y = np.random.randn(n, num_dimensions)
    
    for iteration in range(num_iterations):
        distances_y = distance(Y)
        Q = cond_prob(distances_y, perplexity)
        
        gradient = np.zeros((n, num_dimensions))
        for i in range(n):
            diff = P[i] - Q[i]
            gradient[i] = 4 * np.dot(diff, (Y[i] - Y)) * (1 + np.sum((Y[i] - Y) ** 2)) ** -1
        
        Y -= learning_rate * gradient
        
        if iteration % 100 == 0:
            kl_div = KL_divergence(P, Q)
            print(f"Iteration {iteration}, KL Divergence: {kl_div}")
            
            if kl_div < tolerance:
                break
    
    return Y