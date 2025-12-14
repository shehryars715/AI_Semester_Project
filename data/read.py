import numpy as np

# Replace 'your_file.npy' with the path to your .npy file
data = np.load('data/training_1a.npy', allow_pickle=True)

# To check the contents of the loaded data
print(data)
