import torch
from utils.utils import preprocess, postprocess
from models.model_transpose import UNet # pick the architecture of your preference
# for inference from the ones available in models/ directory
import os
import numpy as np
import mrcfile
import warnings
import time
from utils.toNifti import inference_to_Nifti
warnings.simplefilter('ignore') # for mrcfile

def predict(tomogram,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    tomogram = preprocess(tomogram)  # preprocess image
    x = torch.from_numpy(tomogram)  # convert to torch tensor
    x = x.to(device) # send tensor to device (GPU normally)
    with torch.no_grad():
        out = model(x)  # send through model/network

    seg_map = out[1] # obtain the segmentation map as output
    # den_map = out[0] # to obtain denoised volume as output uncomment this line
    out_softmax = torch.softmax(seg_map, dim=1)  # perform softmax on segmentation output to extract probabilities for each class
    result = postprocess(out_softmax)  # postprocess outputs

    return result

def diceLoss(y_true, y_pred, epsilon=1e-6): # calculate Dice Loss for the produced segmentation map
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch


start = time.time()
datadir= os.getcwd() + '/Output/TestTomogram/InputTomos/'

test_tom_id = 5 # pick a test tomogram of your choice from 0 to 9 (unseen during training)
gt_dir = '../shrec2021/model_' + str(test_tom_id)

# read reconstruction input volume
with mrcfile.open(gt_dir + '/reconstruction.mrc', permissive=True) as tomo0:

    # the tomo data is now accessible via .data, in following order: Z Y X
    tomo0_reconstruction = tomo0.data
    tom = tomo0_reconstruction[160:-160, :, :] ## we keep the middle 192 slices to pass to the network as input for the forward pass

# read ground truth segmentation map
with mrcfile.open(gt_dir+ '/class_mask.mrc', permissive=True) as tomo_mask:

    # the tomo data is now accessible via .data, in following order: Z Y X
    tomo0_mask = tomo_mask.data
    tomm = tomo0_mask[166:-166, :, :] ## we keep the middle 180 slices to use for the calculation of the inference error
seg_gt = np.asarray(tomm)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

# set model parameters
model_name = 'model_TT_final25.pt' ## set the model name to one of the pretrained models available in models directory
model = UNet(in_channels=1,
             out_channels_denoise=1,
             out_channels_segment=16,
             n_blocks=5,
             start_filters=16,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3, name = model_name).to(device)

model_weights = torch.load(os.getcwd() + '/Output/models/' + model_name)
model.load_state_dict(model_weights)

dim_total = 512 ## x-y dimensions of the tomogram
overlap_mask = np.zeros((model.out_channels_segment,tom.shape[0], dim_total, dim_total)) ## initialize overlap mask
seg_map = torch.zeros((model.out_channels_segment,tom.shape[0], dim_total, dim_total)) ## initialize output segmentation map

## We segment the input to patches of size 192x128x128 due to GPU memory. We then iteratively pass these patches to our network
## to get their segmentation maps.
overlap_mode = True ## If overlap is True the patches are overlapping, if False they are not. (True is the deafult choice
## because of better results).
patch_size = 128 # size of each patch in x-y dimensions
overlap = 32 ## 25% in each of x-y dimension, (default to better results).
step=patch_size-overlap
start_x=0

if (overlap_mode == True):
    num_patches_x = int(np.floor(dim_total/step)) ## number of patches in x and y dimension are equal
    for x in range(0,num_patches_x):
        start_y = 0
        for y in range(0, num_patches_x ):
            patch = tom[:, start_x: start_x + patch_size, start_y : start_y + patch_size] # define a 192x128x128 patch of input
            seg_patch = predict(patch, model, preprocess, postprocess, device)
            seg_patch = seg_patch.to('cpu')
            seg_map[:, :, start_x: start_x + patch_size, start_y : start_y + patch_size] += seg_patch
            overlap_mask[:, :, start_x: start_x + patch_size, start_y : start_y + patch_size] += 1
            start_y += step
        start_x += step
    seg_map = seg_map.numpy()
    seg_map = np.divide(seg_map, overlap_mask)
else:
    num_patches_x = int(np.floor(dim_total/patch_size)) ## number of patches in x and y dimension are equal
    for x in range(0,num_patches_x):
        start_y = 0
        for y in range(0, num_patches_x):
            patch = tom[:, start_x: start_x + patch_size, start_y : start_y + patch_size]
            seg_patch = predict(patch, model, preprocess, postprocess, device)
            seg_patch = seg_patch.to('cpu')
            seg_map[:, :, start_x: start_x + patch_size, start_y : start_y + patch_size] = seg_patch
            start_y += patch_size
        start_x += patch_size
    seg_map = seg_map.numpy()

seg_map = np.argmax(seg_map, axis=0) # extract class labels depending on the probability score of each voxel
seg_map = seg_map[6:186,:, :] # from 192 to 180 intermediate slices (in Z dimension)
loss = diceLoss(seg_gt, seg_map) # compute the inference error
print('Dice loss for tomogram ', str(test_tom_id), 'is', loss)
end = time.time()
print('Execution time: ', np.floor((end - start)/60), ' minutes ', (end-start)%60, ' seconds')
inference_to_Nifti(seg_map, model_name) # save output segmentation map to Nifti file format for visualization purposes -- slices are saved
## individually.
