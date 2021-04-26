import numpy as np
import torch
from utils.Transforms import normalize, normalize_01

def generate_dataset(data_tensor, tomograms_num, dim, overlap, mode = 'In', train_or_valid = ''):

    # in z axis, we know the protein information is included in the intermediate 180 slices so we only crop training volumes from this part

    # up_limit_x = 113
    up_limit_x = data_tensor.shape[1] - dim
    examples = [[data_tensor[id, i: i+dim, j: j + dim, k: k + dim]
                 for i in range(0,up_limit_x, overlap)
                 for j in range(0,512 - dim + 1, overlap)
                 for k in range(0,512 - dim + 1, overlap)] for id in range(tomograms_num)]

    examples = np.stack(examples, axis = 0)
    examples = np.reshape(examples,(examples.shape[0]* examples.shape[1], examples.shape[2], examples.shape[3], examples.shape[4]))
    counter = 0
    for example in examples:
        counter = counter + 1
        np.save('Input/3DImages/' + str(mode) + '/' + str(train_or_valid) + '/' + str(counter)+ '.npy', example, allow_pickle=False)
    print(examples.shape) ## tensor shape for training and validation respectively - first dimension corresponds to training
    ## and validation examples' size



def _to_one_hot(y, num_classes, device):

    y = torch.squeeze(y)
    y = torch.tensor(y, dtype=torch.int64)
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    y_tensor = y_tensor.to(device)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    zeros = zeros.to(device)

    return zeros.scatter(scatter_dim, y_tensor, 1)

def preprocess(img: np.ndarray):

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = normalize(img)
    img = img.astype(np.float32)  # typecasting to float32

    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    return img

def augment(input, tar1, tar2, p):
    input_aug = []
    tar1_aug = []
    tar2_aug = []
    for i in range(input.shape[0]):
        number_of_flips = np.random.randint(2, size=1) + 1
        if (np.random.uniform() < p):
            input_aug.append(np.rot90(input[i, :, : ,:], k = number_of_flips, axes=(0,2)))
            tar1_aug.append(np.rot90(tar1[i, :, : ,:], k = number_of_flips, axes=(0,2)))
            tar2_aug.append(np.rot90(tar2[i, :, : ,:], k = number_of_flips, axes=(0,2)))
    input_aug = np.asarray(input_aug)
    tar1_aug = np.asarray(tar1_aug)
    tar2_aug = np.asarray(tar2_aug)
    input = np.concatenate((input, input_aug), axis=0)
    tar1 = np.concatenate((tar1,tar1_aug), axis=0)
    tar2 = np.concatenate((tar2,tar2_aug), axis=0)

    return input,tar1,tar2

