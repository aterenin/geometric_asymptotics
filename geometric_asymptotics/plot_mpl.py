import numpy as np
import matplotlib.pyplot as plt

def plot_kernel(K):
    (fig, ax) = plt.subplots()
    im = ax.imshow(K)
    fig.colorbar(im)

def plot_data(x,y, lims=(-1,1), colorbar=True, highlight = None, file_name = None):
    (fig, ax) = plt.subplots()
    ax.set_axis_off()
    sc = ax.scatter(x[:,0],x[:,1], c = y.squeeze(), vmin=lims[0], vmax=lims[1])
    if colorbar:
        fig.colorbar(sc) 
    if highlight is not None:
        ax.scatter(highlight[0], highlight[1], c="black", s=128)
    if file_name is not None:
        fig.savefig(file_name, transparent=True, bbox_inches='tight')

def plot_errors(idx,e1,e2,e3):
    (fig, ax) = plt.subplots()
    ax.plot(idx, np.log(e1), label="intrinsic")
    ax.plot(idx, np.log(e2), label="extrinsic")
    ax.plot(idx, np.log(e3), alpha=0.5, label="intrinsic approx")
    ax.legend()

