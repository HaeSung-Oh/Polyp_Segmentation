import os, cv2, torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

def imread_kor(path, mode=cv2.IMREAD_UNCHANGED):
    stream = open(path.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    return cv2.imdecode(np.asarray(bytes, dtype=np.uint8), mode)

class ImagesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size=352):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = imread_kor(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = imread_kor(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = self.to_tensor(img).float()
        mask = self.to_tensor(mask).float()

        img = F.interpolate(img.unsqueeze(0), (self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), (self.size, self.size),
                             mode="nearest").squeeze(0)
        mask[mask > 0] = 1

        return img, mask
