import numpy as np
import time
import cc3d
from clusteringAlgorithms.utils_clust import add_obj
from sklearn.cluster import MeanShift

def hybridClustering(datadir, tomograph_id):

    seg_gt = np.load(datadir, allow_pickle= False)
    seg_gt = seg_gt.astype(int)
    seg_gt = seg_gt[tomograph_id, :, :, :]
    connectivity = 26
    start = time.time()
    seg_out,N = cc3d.connected_components(seg_gt, connectivity=connectivity, return_N=True)
    # end = time.time()
    counter_fake = 0

    centroids = []
    # values = np.unique(seg_out)
    for segid in range(1, N + 1):
        extracted_protein = np.argwhere(seg_out == segid)
        clustSize = extracted_protein.shape[0]
        # if (clustSize < 200):
        #     continue
        sum_x = np.sum(extracted_protein[:, 2])
        sum_y = np.sum(extracted_protein[:, 1])
        sum_z = np.sum(extracted_protein[:, 0])

        # voxelsValues = seg_gt[extracted_protein[:, 0], extracted_protein[:, 1], extracted_protein[:,2]]
        # values = np.unique(voxelsValues)
        # # assert((values.shape[0]==1) and (values[0] < 16))
        # if((values.shape[0]!=1) or (values[0] > 16)):
        #     print('error', segid)
        # if(values[0] == 0): ## ignore background class 0
        #     continue
        # else:
        #     winningLabel = values[0]
        centroid = np.array([float(round(sum_x/clustSize)), float(round(sum_y/clustSize)), float(round(sum_z/clustSize))])
        centroids.append(centroid)
    centroids = np.asarray(centroids)
    seg_nonzero = np.nonzero(seg_gt>0)
    seg_nonzero = np.array(seg_nonzero).T

    clustering = MeanShift(bandwidth=14, cluster_all=True, seeds=centroids)
    clustering.fit(seg_nonzero)
    cluster_centers = clustering.cluster_centers_
    #
    num_clusters = cluster_centers.shape[0]
    proteinVoxelsList = []
    # instancesPerClass = np.zeros((num_classes,))
        # proteinVoxelsList = add_obj(proteinVoxelsList, label=winningLabel, centroid=centroid, cluster_size=clustSize)
    # assert()
    # print('')
    # protein = np.argwhere(seg_out == segid)
    # if (protein.shape[0]<200):
    #     counter_fake +=1
    # print('')
    # process(extracted_image)
    # for label, image in cc3d.each(seg_out, binary=False, in_place=True):
    #     process(image)
    # graph = cc3d.voxel_connectivity_graph(seg_out, connectivity=connectivity)
    # print(N-counter_fake)
    end = time.time()
    print(np.floor((end - start)/60), ' minutes ', (end-start)%60, ' seconds')

    return