import numpy as np
from sklearn.cluster import MeanShift
from clusteringAlgorithms.utils_clust import add_obj
import time
import warnings
warnings.simplefilter('ignore')

def MeanShiftClust(datadir, tomograph_id):

    seg_gt = np.load(datadir, allow_pickle= False)
    seg_gt = seg_gt[tomograph_id, :, :, :]
    # seg_gt = np.reshape(seg_gt, (seg_gt.shape[0]* seg_gt.shape[1], seg_gt.shape[2]))

    ## discard non-protein voxels and keep voxels contained in protein macromolecules
    ## classes 1-12
    num_classes = len(np.unique(seg_gt)) - 1
    seg_nonzero = np.nonzero(seg_gt>0)
    seg_nonzero = np.array(seg_nonzero).T

    ## define clustering algorithm
    start = time.time()
    clustering = MeanShift(bandwidth=14, bin_seeding=True, cluster_all=True)
    clustering.fit(seg_nonzero)
    end = time.time()
    print(np.floor((end - start)/60), ' minutes ', (end-start)%60, ' seconds')
    ## define clustering
    # clustering.fit(seg_gt)
    # end = time.time()
    cluster_centers = clustering.cluster_centers_
    #
    num_clusters = cluster_centers.shape[0]
    proteinVoxelsList = []
    instancesPerClass = np.zeros((num_classes,))

    numClusters = num_clusters
    for cluster in range(num_clusters):
        cluster_id_member = np.nonzero(clustering.labels_ == cluster)
        ## find cluster size and position
        clusterSize = np.size(cluster_id_member)
        if(clusterSize < 500):
            numClusters = numClusters-1
            continue

        centroid = clustering.cluster_centers_[cluster]

        clustMember = []
        for member in range(clusterSize):  # get labels of cluster members
            clustMemberCoords = seg_nonzero[cluster_id_member[0][member], :]
            clustMember.append(seg_gt[clustMemberCoords[0], clustMemberCoords[1], clustMemberCoords[2]])

        for label in range(num_classes):  # get most present label in cluster

            instancesPerClass[label] = np.size(np.nonzero(np.array(clustMember) == label + 1))
        winninglabel = np.argmax(instancesPerClass) + 1
        proteinVoxelsList = add_obj(proteinVoxelsList, label=winninglabel, centroid=centroid, cluster_size=clusterSize)

    return proteinVoxelsList

