import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import pandas as pd
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm, trange
import math
from utils.utils import _to_one_hot
import os

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion_denoise: torch.nn.Module,
                 criterion_segment: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset,
                 lr_scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 model_name: str = ''
                 ):

        self.model = model
        self.criterion_denoise = criterion_denoise
        self.criterion_segment = criterion_segment
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.model_name = model_name
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epoch,self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            save_checkpoint({
                'epoch': self.epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, True, filename=os.getcwd() + '/Output/checkpoints/' + self.model_name + '_' +str(self.epoch) + '.pt')
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        torch.save(self.model.state_dict(), os.getcwd() + '/Output/models/' + self.model_name) ## save final model after training

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y_denoise, y_segment) in batch_iter:
            input, target_denoise, target_segment = x.to(self.device), y_denoise.to(self.device), y_segment.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out_denoise, out_segment = self.model(input)  # one forward pass
            loss_den = self.criterion_denoise(out_denoise, target_denoise)
            device = self.device
            target_segment = _to_one_hot(target_segment, 16, device)
            out_segment = out_segment.permute(0, 2, 3, 4, 1)
            loss_seg = self.criterion_segment(out_segment, target_segment)  # calculate loss
            loss = loss_den + loss_seg
            loss_value_den = loss_den.item()
            loss_value_seg = loss_seg.item()
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f}) (loss_den {loss_value_den:.4f}) (loss_seg {loss_value_seg:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y_denoise, y_segment) in batch_iter:
            input, target_denoise, target_segment = x.to(self.device), y_denoise.to(self.device), y_segment.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out_denoise, out_segment = self.model(input)
                loss_den = self.criterion_denoise(out_denoise, target_denoise)
                device = self.device
                target_segment = _to_one_hot(target_segment, 16
                                             , device)
                out_segment = out_segment.permute(0, 2, 3, 4, 1)

                loss_seg = self.criterion_segment(out_segment, target_segment)  # c
                loss = loss_den + loss_seg
                loss_value_den = loss_den.item()
                loss_value_seg = loss_seg.item()
                loss_value = loss.item()
                valid_losses.append(loss_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}) (loss_den {loss_value_den:.4f}) (loss_seg {loss_value_seg:.4f})')
        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 device
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        self.device = device

    def fit(self,
            data_loader: torch.utils.data.DataLoader,
            steps=100,
            min_lr=1e-7,
            max_lr=1,
            constant_increment=False
            ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)

            for i, (x, y) in batch_iter:
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()

                steps_counter += 1
                if steps_counter > steps:
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def reset(self):
        """
        Resets the model and optimizer to its initial state
        """
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')

def plot_training(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('LR')

    return fig

def save_checkpoint(state, filename= ''):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a checkpoint")
    torch.save(state, filename)  # save checkpoint
