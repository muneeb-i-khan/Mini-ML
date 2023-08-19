import pca as PCA
import numpy as np

np.random.seed(0)
num_samples = 1000
num_features = 5
custom_data = np.random.randn(num_samples, num_features) + 2  

num_components = 3
transformed_data = PCA.PCA(custom_data, num_components)

print("Original data shape:", custom_data.shape)
print("Transformed data shape:", transformed_data.shape)