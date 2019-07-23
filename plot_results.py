import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt

path = "/home/jfelip/workspace/prob-comp-code/sampling_experiments/"
# filenames = ["res_BSPT_gmm.txt"]
filenames = fnmatch.filter(os.listdir(path), 'res_*rosenbrock*.txt')
series = []

for file in filenames:
    series.append(np.loadtxt(path+file))


fig = plt.figure()
for serie, name in zip(series, filenames):
    plt.subplot(211)
    plt.plot(serie[:, 0], serie[:, 1])
    plt.ylabel("KL Divergence")
    plt.xlabel("# samples")
    plt.legend(filenames)

    plt.subplot(212)
    plt.plot(serie[:, 0], serie[:, 2])
    plt.ylabel("bhattacharyya Distance")
    plt.xlabel("# samples")
    plt.legend(filenames)

plt.show(True)
