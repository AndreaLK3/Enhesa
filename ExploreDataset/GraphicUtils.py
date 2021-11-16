import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap


def create_gyr_colormap():
    viridisBig = cm.get_cmap('hsv', 1024)
    newcmp = ListedColormap(viridisBig(np.linspace(0.38, 0.08, 1024)))

    return newcmp


def save_figure(fig, out_fpath):
    fig.tight_layout()
    fig.savefig(out_fpath)