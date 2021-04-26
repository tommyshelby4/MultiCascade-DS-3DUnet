from utils.SegmentationDataset import SegmentationDataSet
from utils.Transforms import Normalize01, Normalize, Compose
import os
import torch
from models.model_upsampling import UNet ## pick type of model, model_transpose, model_ATTENTION_upsampling and model_upsampling available
from utils.trainer import Trainer, plot_training
from torch.utils.data import DataLoader
from utils.losses import TverskyLoss, SoftDiceLoss, FocalTverskyLoss
from torchsummary import summary
import pathlib


# load 3D Images for Training and Validation respectively
root = pathlib.Path(os.getcwd() + '/Input/3DImages/')
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames
# input and target files

inputs_train = get_filenames_of_path(root / 'In/Training/')
targets_1_train = get_filenames_of_path(root / 'Den/Training/')
targets_2_train = get_filenames_of_path(root / 'Seg/Training/')
inputs_valid = get_filenames_of_path(root / 'In/Validation/')
targets_1_valid = get_filenames_of_path(root / 'Den/Validation/')
targets_2_valid = get_filenames_of_path(root / 'Seg/Validation/')

year = '2021'

# training transformations and augmentations
transforms_training = Compose([
   # DenseTarget(),
    Normalize()
])
# validation transformations
transforms_validation = Compose([
    # DenseTarget(),
    Normalize()
])

# generate training datasets for denoising and segmentation for both training and validation
# augmentation techniques are applied and can be edited through the Segmentation Dataset class
#dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets_denoising=targets_1_train,
                                    targets_seg= targets_2_train,
                                    transform=transforms_training,
                                    use_cache=False,
                                    pre_transform=None)

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets_denoising=targets_1_valid,
                                    targets_seg= targets_2_valid,
                                    transform=transforms_validation,
                                    use_cache=False,
                                    pre_transform=None)
# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=4,
                                 shuffle=True)
# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=4,
                                   shuffle=True)

# set device --> current configuration for GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

# define model and parameters
epochs = 1 # change value to adjust number of epochs for training
model_name = 'model_TT_final' + str(epochs)
model = UNet(in_channels=1,
             out_channels_denoise=1,
             out_channels_segment=16, # 16 classes of proteins in segmentation maps
             n_blocks=5,       # number of convolutional blocks
             start_filters=16, # number of feature maps in the first encoding layer
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3, name = model_name).to(device)

# summary = summary(model, (1, 64, 64, 64))

# # criterion
criterion_denoise = torch.nn.MSELoss() ## denoising loss
criterion_seg = TverskyLoss() ## segmentation Loss
# criterion_seg = SoftDiceLoss() , for SoftDice Loss comment the previous line and uncomment this one
# criterion_seg = FocalTverskyLoss() , for FocalTverskyLoss comment the previous line and uncomment this one

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 0
resume = False ## to resume your training after a certain checkpoint set resume to True and epoch to the respective starting epoch
## and desired model checkpoint
## default value for resume set to False to start training from scratch
model_checkpoint = 'model_TT_final25_1.pt'
if(resume):
    checkpoint = torch.load(os.getcwd() + '/Output/checkpoints/' + model_checkpoint + '/',map_location=lambda storage,
                                                                                    loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion_denoise= criterion_denoise,
                  criterion_segment=criterion_seg,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=epochs,
                  epoch=epoch,
                  notebook=False, model_name=model_name)
# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()
fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10,4))
torch.save(model.state_dict(), os.getcwd() + '/Output/' + model.name)
fig.savefig(os.getcwd() + '/LossesFigs/' + model_name +'.jpg')
