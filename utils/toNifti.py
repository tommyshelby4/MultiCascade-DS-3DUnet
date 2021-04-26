import numpy as np
import nibabel as nib
import os

## generate the 3D segmentation mask in .nii format (postprocessing purposes) of the produced segmentation map
def inference_to_Nifti(center_crop_mask, model_name):
    curr_dir = os.getcwd() + '/Output/SegMapSlices/'
    ## uncomment the commented lines to generate 2D slices of the segmentation map for visualization purposes
    # os.makedirs(curr_dir + model_name.split('.')[0]+ '/')
    center_crop_mask = center_crop_mask.astype('float32')
    # for i in range(center_crop_mask.shape[0]):
    #     slice_mask = center_crop_mask[i, :, :]
    #     slice_mask = nib.Nifti1Image(slice_mask, affine=np.eye(4))
    #     nib.save(slice_mask, curr_dir + model_name.split('.')[0]+ '/''seg' + str(i+1)+ '.nii')
    center_crop_mask = nib.Nifti1Image(center_crop_mask, affine=np.eye(4))
    nib.save(center_crop_mask, curr_dir + model_name.split('.')[0] + '.nii')
