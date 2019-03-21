from MulticoreTSNE import MulticoreTSNE as TSNE
import scipy.io as io


mat = io.loadmat('fuck.mat')
t = mat['gt']
tsne = TSNE(n_jobs=32)
a = tsne.fit_transform(mat['g'])

io.savemat('aa.mat', {'a': a, 't': t})
