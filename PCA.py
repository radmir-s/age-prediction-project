import numpy as np

Xlist = []
for i in range(1,101):
    Xlist.append(np.load(f'extracted_encodings/train/X{i}.npy'))
X = np.concatenate(Xlist)

