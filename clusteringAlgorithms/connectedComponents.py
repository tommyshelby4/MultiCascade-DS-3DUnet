import numpy as np
import time
import cc3d
from clusteringAlgorithms.utils_clust import add_obj
import os
import nibabel as nib

# datadir = os.getcwd() + '/segmentation_masks.npy'
def ConnectedComponents(datadir, tomograph_id):

    seg_gt = nib.load(datadir)
    seg_gt = np.asarray(seg_gt.dataobj)
    seg_gt = seg_gt.astype(int)
    seg_gt = np.moveaxis(seg_gt,-1,0)
    seg_gt = seg_gt[6:186, :, :]
    connectivity = 26
    start = time.time()
    seg_out,N = cc3d.connected_components(seg_gt, connectivity=connectivity, return_N=True)
    proteinVoxelsList = []
    centroid_and_IDs = []
    for segid in range(1, N + 1):
        extracted_protein = np.argwhere(seg_out == segid)
        clustSize = extracted_protein.shape[0]
        sum_x = np.sum(extracted_protein[:, 2])
        sum_y = np.sum(extracted_protein[:, 1])
        sum_z = np.sum(extracted_protein[:, 0])

        voxelsValues = seg_gt[extracted_protein[:, 0], extracted_protein[:, 1], extracted_protein[:,2]]
        values = np.unique(voxelsValues)
        if((values.shape[0]!=1) or (values[0] > 16)):
            print('error', segid)
        if(values[0] == 0 or values[0] == 14): ## ignore background class 0
            continue
        else:
            winningLabel = values[0]
            centroid = np.array([int(round(sum_x/clustSize)), int(round(sum_y/clustSize)), int(round(sum_z/clustSize))+166])
            centroid_and_ID = np.array([int(round(sum_x/clustSize)), int(round(sum_y/clustSize)), int(round(sum_z/clustSize)), winningLabel])
            centroid_and_IDs.append(centroid_and_ID)
        proteinVoxelsList = add_obj(proteinVoxelsList, label=winningLabel, centroid=centroid, cluster_size=clustSize)

    end = time.time()
    centroid_and_IDs = np.asarray(centroid_and_IDs)
    # np.save(os.getcwd()+ '/Output/TestTomogram/centroids'+ str(tomograph_id) + '.npy', centroid_and_IDs, allow_pickle=False) --> uncomment to write the array of xyz centroids
    print(np.floor((end - start)/60), ' minutes ', (end-start)%60, ' seconds')

    return proteinVoxelsList