import tSNE as tsne
import numpy as np

data = np.random.randn(100, 50)  
embedded_data = tsne.tSNE(data)
print(embedded_data.shape)  