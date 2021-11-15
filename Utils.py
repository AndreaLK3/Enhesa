import pandas as pd
from enum import Enum
import Filepaths as F
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

class Column(Enum):
    CLASS = "class"
    ARTICLE = "article"

# Load train.csv and/or test.csv as Pandas dataframes
def load_split(split_enum):
    if split_enum == Split.TRAINING:
        df = pd.read_csv(F.training_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value])
    elif split_enum == Split.VALIDATION:
        df = None
    elif split_enum == Split.TEST:
        df = pd.read_csv(F.test_file, sep=";", names=[Column.CLASS.value, Column.ARTICLE.value], index_col=False)
    return df

# Green, through yellow, to red
def create_gyr_colormap():
    viridisBig = cm.get_cmap('hsv', 1024)
    newcmp = ListedColormap(viridisBig(np.linspace(0.38, 0.08, 1024)))

    return newcmp

def save_figure(fig, out_fpath):
    fig.tight_layout()
    fig.savefig(out_fpath)


