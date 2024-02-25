import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.utils import pytorch_after

def one_hot_encode(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

def dice_loss(input, target, smooth=1e-4):
    target = target.float()
    intersect = torch.sum(input * target)
    ground = torch.sum(target)
    pred = torch.sum(input)
    dice = (2 * intersect + smooth) / (ground + pred + smooth)
    loss = 1 - dice
    return loss, dice

class DiceLoss1class(nn.Module):
    def __init__(self, reduction='one', sigmoid=True, softmax=False):
        super(DiceLoss1class, self).__init__()
        self.reduction = reduction
        self.sigmoid = sigmoid

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        loss1, dice1 = dice_loss(inputs,targets)

        total_loss = loss1
        return total_loss, [0, dice1]

class DiceLoss(nn.Module):
    def __init__(self, reduction='sum', sigmoid=False, softmax=True, on_hot=True):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.on_hot = on_hot

    def forward(self, inputs, targets):
        if self.sigmoid:
            inputs = torch.sigmoid(inputs)
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        if self.on_hot:
            targets = one_hot_encode(targets, num_classes=3, dim=1)

        back_loss, back_dice = dice_loss(inputs[:,0],targets[:,0])
        loss1, dice1 = dice_loss(inputs[:,1],targets[:,1])
        loss2, dice2 = dice_loss(inputs[:,2],targets[:,2])

        if self.reduction == 'sum':
            #total_loss = back_loss + loss1 + loss2
            total_loss = back_loss + loss1 + loss2
        if self.reduction == 'mean':
            total_loss = (back_loss + loss1 + loss2 ) / 3
        return total_loss, [dice1, dice2]

class CELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.old_pt_ver = not pytorch_after(1, 10)
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(input, target)
        dice_loss, dice = self.dice_loss(input, target)
        return ce_loss, dice

class DiceLossDS(nn.Module):
    def __init__(self, reduction='sum', sigmoid=False, softmax=True, on_hot=True):
        super(DiceLossDS, self).__init__()
        self.diceloss = DiceLoss()
        self.ds_weight = [0.0625,0.125,0.25,0.5,1]

    def forward(self, final, ds_outs, targets):
        pred_loss, dice = self.diceloss(final, targets)
        total_loss = pred_loss.clone()

        for idx, ds in enumerate(ds_outs):
            total_loss += self.ds_weight[idx] * self.diceloss(ds, targets)[0]
        return total_loss, pred_loss, dice

class DSLoss(nn.Module):
    def __init__(self):
        super(DSLoss, self).__init__()
        self.diceloss = DiceLoss()
        self.ds_weight = [0.0625,0.125,0.25,0.5,0.75]

    def forward(self, ds_outs, targets):
        total_loss = self.ds_weight[0] * self.diceloss(ds_outs[0], targets)[0]

        for idx in range(1, len(ds_outs)):
            total_loss += self.ds_weight[idx] * self.diceloss(ds_outs[idx], targets)[0]
        return total_loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1, sigmoid=False, softmax=True):

        if sigmoid:
            inputs = torch.sigmoid(inputs)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        ce_targets =targets.clone()
        ce_targets = torch.squeeze(ce_targets,1).long()
        CE = self.ce(inputs, ce_targets)

        targets = one_hot_encode(targets, num_classes=3, dim=1)
        back_loss, back_dice = dice_loss(inputs[:,0],targets[:,0])
        liver_loss, liver_dice = dice_loss(inputs[:,1],targets[:,1])
        tumor_loss, tumor_dice = dice_loss(inputs[:,2],targets[:,2])
        dice_part = (back_loss + liver_loss + tumor_loss)/3

        total_loss = CE + dice_part

        return total_loss, CE, liver_dice, tumor_dice


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, alpha=0.3, beta=0.7):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])[0]
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])[0]
        return l