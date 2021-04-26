import mrcfile as mrc
import numpy as np
import warnings
from pathlib import Path
from pycm import ConfusionMatrix
from collections import OrderedDict
from scipy.spatial import distance
from contextlib import redirect_stdout
import argparse
import json
warnings.simplefilter('ignore') # for mrcfile
# This is an example of loading a tomogram as a numpy array.
#
# The example uses "mrcfile" package, which is installable
# 	by PIP: pip install mrcfile
# 	by Conda: conda install --channel conda-forge mrcfile
#
# mrcfile documentation can be found here: https://mrcfile.readthedocs.io/
#
# Contact us if you have questions at i.gubins@uu.nl


import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos

model_id = 0
year = '2021'
out_dir = '../U_Net_Cascade/Input/Dataset' + year + '/'
reconstruction = []
bounding_box = []
mmask = []
grandmodels = []
grandmodels_noisefree = []
occupancy_bounding_box = []
occupancy_mmask = []
projection = []
projection_noisefree = []
for model_id in range(9):
	model_dir = '../shrec' +year + '/model_' + str(model_id)
	## read input volumes (noisy)
	with mrcfile.open(model_dir+ '/reconstruction.mrc', permissive=True) as tomo0:

		# the tomo data is now accessible via .data, in following order: Z Y X
		tomo0_reconstruction = tomo0.data
		center_crop = tomo0_reconstruction[166:-166, :, :]
		reconstruction.append(center_crop)
		print('Shape of the tomogram:', center_crop.shape)
		print('Size of the tomogram:', center_crop.nbytes / float(1000**3), 'GB')
    ## read ground truth class masks
	with mrcfile.open(model_dir + '/class_mask.mrc', permissive=True) as mask:
		tomo0_mmask = mask.data
		center_crop = tomo0_mmask[166:-166, :, :]
		mmask.append(center_crop)
		print('Masks',np.min(mmask[model_id]), np.max(mmask[model_id]))
		print('Shape of the mask:', center_crop.shape)
	## read denoised volumes ground truth
	with mrcfile.open(model_dir + '/grandmodel.mrc' , permissive=True) as grandmodel:
		tomo0_grandmodel = grandmodel.data
		center_crop = tomo0_grandmodel[166:-166, :, :]
		grandmodels.append(center_crop)
		print('Shape of the grandmodel:', tomo0_grandmodel.shape)

reconstruction = np.asarray(reconstruction)
mmask = np.asarray(mmask)
grandmodels = np.asarray(grandmodels)

## save data into np compressed format in directory Input/Dataset2021
with open(out_dir + 'reconstruction.npy', 'wb') as f:
	np.save(f, reconstruction)
with open(out_dir + 'grandmodels.npy', 'wb') as f:
	np.save(f, grandmodels)
with open(out_dir + 'segmentation_masks.npy', 'wb') as f:
	np.save(f, mmask)

