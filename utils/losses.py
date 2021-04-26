import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class SoftDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    def forward(self, y_pred, y_true, epsilon=1e-6):

        axes = tuple(range(1, len(y_pred.shape)-1))
        numerator = 2. * torch.sum(y_pred * y_true, axes)
        denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)

        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch

ALPHA = 0.7
BETA = 0.3


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, logits, true, smooth=1, alpha=ALPHA, beta=BETA, eps = 1e-7):

    #         Args:
    #         true: a tensor of shape [B, H, W] or [B, 1, H, W].
    #         logits: a tensor of shape [B, C, H, W]. Corresponds to
    #         the raw output or logits of the model.
    #     alpha: controls the penalty for false positives.
    #         beta: controls the penalty for false negatives.
    #         eps: added to the denominator for numerical stability.
    # Returns:
    # tversky_loss: the Tversky loss.
    # Notes:
    # alpha = beta = 0.5 => dice coeff
    # alpha = beta = 1 => tanimoto coeff
    # alpha + beta = 1 => F beta coeff
    # References:
    # [1]: https://arxiv.org/abs/1706.05721
        true_1_hot = true.permute(0,4, 1, 2, 3).float()
        logits = logits.permute(0,4, 1, 2, 3).float()
        probas = F.softmax(logits.float(), dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()

        return (1 - tversky_loss)

ALPHA = 0.7
BETA = 0.3
GAMMA = 0.8

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()



    def forward(self, logits, true, smooth=1, alpha=ALPHA, beta=BETA, gamma = GAMMA, eps = 1e-7):
        true_1_hot = true.permute(0,4, 1, 2, 3).float()
        logits = logits.permute(0,4, 1, 2, 3).float()
        probas = F.softmax(logits.float(), dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        FocalTversky = (1 - tversky_loss)**gamma

        return FocalTversky