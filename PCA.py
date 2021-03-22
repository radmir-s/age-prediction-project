import numpy as np

# importing the data
encodings = []
for i in range(1, 101):
    enc = np.load(f'extracted_encodings/train/X{i}.npy')
    enc_i = np.concatenate((enc, i * np.ones((enc.shape[0], 1))), axis=1)
    encodings.append(enc_i)
encodings = np.concatenate(encodings)

# measurements correlated to age
corcoeff = .35
meas_age_indices = np.where(abs(np.corrcoef(encodings.T)[-1, :]) > corcoeff)[0]

# finding PCA
encodings_avg = np.mean(encodings, axis=0)
n = encodings_avg.shape[0]
X = encodings - encodings_avg
U, S, VT = np.linalg.svd(X / np.sqrt(n), full_matrices=False)
cdS = np.cumsum(S) / np.sum(S)  # Cumulative energy
r90 = np.min(np.where(cdS > 0.9))
