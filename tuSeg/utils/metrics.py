import torch
import torch.nn as nn

def threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def Dice(outputs, targets, smooth=1):
    outputs = threshold(outputs, 0.5)
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    intersection = torch.sum(outputs * targets)
    dice =  (2.*intersection + smooth)/(torch.sum(outputs) + torch.sum(targets) + smooth)  
    print('dice mose:',dice.item())
    return dice

def IoU(outputs, labels, smooth=1e-6):
    outputs = threshold(outputs, 0.5)
    outputs = outputs.view(-1)
    labels = labels.view(-1)

    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels) - intersection + smooth 
    return (intersection + smooth) / union

def F1_score(pr, gt, eps=1e-6, beta=1):
    pr = threshold(pr, 0.5)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score


def Accuracy(pr, gt):
    pr = threshold(pr, 0.5)
    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def Recall(pr, gt, eps=1e-6):
    pr = threshold(pr, 0.5)
    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp
    score = (tp + eps) / (tp + fn + eps)
    return score

def Precision(pr, gt, eps=1e-6):
    pr = threshold(pr, 0.5)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score