import torch

def pixel_accuracy(pred, gt):
    pred = pred > 0.5
    return (pred == gt).sum() / gt.numel()

def iou(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    return inter / (union + 1e-6)

def dice(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    inter = (pred & gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-6)

def precision_recall_f1(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5

    tp = (pred & gt).sum()
    fp = (pred & (~gt)).sum()
    fn = ((~pred) & gt).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1
