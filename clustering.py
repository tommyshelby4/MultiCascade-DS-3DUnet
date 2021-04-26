import os
from clusteringAlgorithms.meanShiftClustering import MeanShiftClust
from clusteringAlgorithms.connectedComponents import ConnectedComponents
from clusteringAlgorithms.hybridClustering import hybridClustering
from clusteringAlgorithms.utils_clust import write_txt

proteinDict = {       ## dict with segmentation classes (protein IDs + background)
    "0": "background",
    "1": "4V94",
    "2": "4CR2",
    "3": "1QVR",
    "4": "1BXN",
    "5": "3CF3",
    "6": "1U6G",
    "7": "3D2F",
    "8": "2CG9",
    "9": "3H84",
    "10": "3GL1",
    "11": "3QM1",
    "12": "1S3X",
    "13": "5MRC",
    "14": "vesicle",
    "15": "fiducial"
}

tom_id = 5 ## pick a tomogram for performing clustering and subsequently localizing protein molecules
test_tom_output = 'seg_model_TT_final25_post' ## define the post-processed segmentation map (in Nifti format) to perform clustering on
datadir = os.getcwd() + '/Output/TestTomogram/' + test_tom_output + '.nii'
idx=1 ## pick clustering algorithm
# 0: Mean Shift
# 1: Connected Components --> recommended
mode = ['MeanShift', 'ConnectedComponents', 'Hybrid']
pick_mode = mode[idx]
model_name = 'SegAT' # define name of trained model employed for the inference to avoid confusion between different runs
if(pick_mode == 'MeanShift'):
    proteinsDetected = MeanShiftClust(datadir, tomograph_id=tom_id)
    write_txt(proteinsDetected, filename=os.getcwd() + '/Output/ClusteringResults/' + model_name + '_tom' +str(tom_id) + '_' + pick_mode, classID2classname=proteinDict)
if(pick_mode == 'ConnectedComponents'):
    proteinsDetected = ConnectedComponents(datadir, tomograph_id= tom_id)
    write_txt(proteinsDetected, filename=os.getcwd() + '/Output/ClusteringResults/' + model_name + '_tom' +str(tom_id) + '_' + pick_mode, classID2classname=proteinDict)

