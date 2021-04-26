import os
import numpy as np
from utils.utils import generate_dataset

def dataset_gen(mode, year, valid_id):
    datadir = os.getcwd()
    random_seed = 42


    inputs = datadir + '/Input/Dataset' + year + '/reconstruction.npy'
    denoising_target= datadir + '/Input/Dataset' + year + '/grandmodels.npy'
    segmentation_target = datadir + '/Input/Dataset'+ year + '/segmentation_masks.npy'

    train_size = 8/9  # 8 out of 9 available tomograms used for training and one for validation


    overlap = 32
    # dataset training
    if(mode == 'In'): # generate input training volumes of 64 X 64 X 64
        training_set = np.load(inputs, allow_pickle=False)
        x1,y = training_set[:valid_id, :, :, :], training_set[valid_id, :, :, :]
        x2 = training_set[valid_id + 1:, :, :, :]
        x = np.concatenate((x1,x2), axis=0)
        y = np.expand_dims(y,axis=0)
        generate_dataset(x, tomograms_num=8, dim=64, overlap=overlap,train_or_valid='Training')
        generate_dataset(y, tomograms_num=1, dim=64, overlap=overlap, train_or_valid='Validation')


    elif (mode == 'Den'): # generate denoised target volumes of 64 X 64 X 64
        denoising_gt = np.load(denoising_target, allow_pickle=False)
        x1,y = denoising_gt[:valid_id, :, :, :], denoising_gt[valid_id, :, :, :]
        x2 = denoising_gt[valid_id + 1:, :, :, :]
        x = np.concatenate((x1,x2), axis=0)
        y = np.expand_dims(y,axis=0)
        generate_dataset(x, tomograms_num=8, dim=64, overlap=overlap, mode=mode,train_or_valid='Training')
        generate_dataset(y, tomograms_num=1, dim=64, overlap=overlap, mode = mode, train_or_valid= 'Validation')

    else: # generate segmentation target masks of 64 X 64 X 64
        seg_gt = np.load(segmentation_target, allow_pickle= False)
        x1,y = seg_gt[:valid_id, :, :, :], seg_gt[valid_id, :, :, :]
        x2 = seg_gt[valid_id + 1:, :, :, :]
        x = np.concatenate((x1,x2), axis=0)
        y = np.expand_dims(y,axis=0)
        generate_dataset(x, tomograms_num=8, dim=64, overlap=overlap, mode=mode, train_or_valid='Training')
        generate_dataset(y, tomograms_num=1, dim=64, overlap=overlap, mode = mode, train_or_valid= 'Validation')

modes = ['In', 'Den', 'Seg']
valid_id = 5 # valid id defines which tomogram is used for validation during training
for mode in modes:
    dataset_gen(mode, '2021', valid_id)


