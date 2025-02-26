import numpy as np

# Load the .npy files
data1 = np.load("data/embeddings/Ha.npy")
data2 = np.load("data/embeddings/Nam.npy")

# Check if they are identical
if np.array_equal(data1, data2):
    print("The files are identical.")
else:
    print("The files are different.")

