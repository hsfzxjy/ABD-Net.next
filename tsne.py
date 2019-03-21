from MulticoreTSNE import MulticoreTSNE as TSNE
import scipy.io as io
import numpy as np
from numpy.linalg import norm

mat = io.loadmat('fuck.mat')
t = mat['gt']
tsne = TSNE(n_jobs=32)

features = mat['g']
features = features / (norm(features, axis=1, keepdims=True))

a = tsne.fit_transform(mat['g'])

io.savemat('aa.mat', {'a': a, 't': t})