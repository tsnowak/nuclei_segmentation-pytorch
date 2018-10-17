import torch
import numpy as np

__all__ = ['iou_pytorch']

# taken from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
SMOOTH = 1e-6

def iou_pytorch(outputs, labels, thresh=.5):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs > thresh
    labels = labels.byte()
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)

    intersection = (outputs & labels).float().sum(2).sum(1)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(2).sum(1)         # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


# NOT IMPLEMENTED
def iou_numpy(outputs, labels):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


if __name__ == "__main__":

    x_t = torch.zeros((1,1,512,512))
    y_t = torch.ones((1,1,512,512))
    miou_t = iou_pytorch(x_t, y_t)
    assert miou_t == 0

    print("Passed Mean IOU all zeros/ones test.")

    x_t = torch.ones((1,1,512,512))
    y_t = torch.ones((1,1,512,512))
    miou_t = iou_pytorch(x_t, y_t)
    assert miou_t == 1

    print("Passed Mean IOU all ones test.")

    x_t = torch.zeros((1,1,512,512))
    y_t = torch.zeros((1,1,512,512))
    miou_t = iou_pytorch(x_t, y_t)
    assert miou_t == 1

    print("Passed Mean IOU all zeros test.")

    x_t = torch.ones((1,1,512,512))*.6
    y_t = torch.ones((1,1,512,512))
    miou_t = iou_pytorch(x_t, y_t)
    assert miou_t == 1

    print("Passed Mean IOU .5/ones test.")
