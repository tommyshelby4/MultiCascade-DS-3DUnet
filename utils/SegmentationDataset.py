import torch
from torch.utils import data
import numpy as np

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets_denoising: list,
                 targets_seg:list,
                 transform=None,
                 use_cache= False,
                 pre_transform = None
                 ):
        self.inputs = inputs
        self.targets_1 = targets_denoising
        self.targets_2 = targets_seg
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.p = 0.35 ## change p value to adjust the percentage of total examples used for augmentation purposes

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_voxel_grid, zip(inputs, targets_denoising, targets_seg, repeat(self.pre_transform)))




    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        if self.use_cache:
            x, y, z = self.cached_data[index]
        else:
            input_ID = np.load(self.inputs[index], allow_pickle= False)
            target_ID_1 = np.load(self.targets_1[index], allow_pickle=False)
            target_ID_2 = np.load(self.targets_2[index], allow_pickle=False)
            number_of_flips = np.random.randint(2, size=1) + 1
            ## data augmentation --> flipped volumes by 90 or 180 degrees randomly with equal probability
            if (np.random.uniform() < self.p):
                input_ID = np.rot90(input_ID, k = number_of_flips, axes=(0,2))
                target_ID_1 = np.rot90(target_ID_1, k = number_of_flips, axes=(0,2))
                target_ID_2= np.rot90(target_ID_2, k = number_of_flips, axes=(0,2))
            # Load input and target
            x, y, z = input_ID, target_ID_1, target_ID_2

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y, z = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)
        x, y, z = torch.from_numpy(x.copy()).type(self.inputs_dtype), torch.from_numpy(y.copy()).type(self.targets_dtype), torch.from_numpy(z.copy()).type(self.targets_dtype)
        return x, y, z

    @staticmethod
    def read_voxel_grid(inp, tar1, tar2, pre_transform):
        inp_final, tar1_final, tar2_final = inp, tar1, tar2
        if pre_transform:
            inp, tar = pre_transform(inp, tar1)
        return inp_final, tar1_final, tar2_final