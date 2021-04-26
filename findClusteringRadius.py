import mrcfile as mrc
import numpy as np
import warnings
from math import ceil, floor
import statistics
from pathlib import Path
from pycm import ConfusionMatrix
from collections import OrderedDict
from scipy.spatial import distance
from contextlib import redirect_stdout
import argparse
import json
warnings.simplefilter('ignore') # for mrcfile

import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos
import os
model_dir = '../shrec2021/misc/particles/'
proteins_dir = os.listdir(model_dir)
proteins_dir_split = [proteins_dir[i].split('_') for i in range(len(proteins_dir))]
protein_names = [proteins_dir_split[i][0] for i in range(len(proteins_dir))]

protein_bb_size = []
volumes = []
for i in range(len(proteins_dir)):
    with mrcfile.open(model_dir + proteins_dir[i], permissive=True) as particle_model:

        # the tomo data is now accessible via .data, in following order: Z Y X
        particle_reconstruction = particle_model.data
        actual_vol = np.count_nonzero(particle_reconstruction)
        print('Protein ', protein_names[i], 'with shape ', particle_reconstruction.shape, 'and vol ', actual_vol)
        # max_x, min_x = np.max(particle_reconstruction, axis = 0), np.min(particle_reconstruction, axis = 0)
        # diff_x = max_x - min_x
        # max_y, min_y = np.max(particle_reconstruction, axis = 1), np.min(particle_reconstruction, axis = 1)
        # diff_y = max_y - min_y
        # max_z, min_z = np.max(particle_reconstruction, axis = 2), np.min(particle_reconstruction, axis = 2)
        # diff_z = max_z - min_z
        protein_bb_size.append(particle_reconstruction.shape[0])
        volumes.append(actual_vol)
# clustering_radius = ceil(max(protein_bb_size)/2)
clustering_radius = ceil(statistics.mean(protein_bb_size)/2)
clust_vol_min = min(volumes)
clust_vol_max = max(volumes)
print('Clustering Radius is ', clustering_radius)
print('Minimum Protein Volume is ', clust_vol_min)
print('Maximum Protein Volume is ', clust_vol_max)