import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_metric(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return dice.mean()

def iou_metric(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def hausdorff_distance(pred, target):
    pred = pred.cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)
    distances = []
    for p, t in zip(pred, target):
        p = p.squeeze()
        t = t.squeeze()
        distances.append(max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0]))
    return np.mean(distances)