import random
from matplotlib.pyplot import axis
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
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

class LiTSDataset(Dataset):
    def __init__(self, base_dir, split='train', num=None, transform=False, one_hot=False):
        self._base_dir = base_dir + '/new'
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.one_hot = one_hot
        if self.split == 'train':
            with open(self._base_dir + '/train_list.txt', 'r') as f:
                 self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        elif self.split == 'val':
            with open(self._base_dir + '/val_list.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        elif self.split == 'test':
            with open(self._base_dir + '/test_list.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        self.ct_list = self.sample_list
        self.seg_list = list(map(lambda x: x.replace('ct', 'seg').replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))

    def __len__(self):
        return len(self.ct_list)

    def __getitem__(self, idx):
        ct_path = '/workspace'+self.ct_list[idx]
        seg_path = '/workspace'+self.seg_list[idx]

        ct = sitk.ReadImage(ct_path)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        image = sitk.GetArrayFromImage(ct)
        labels = sitk.GetArrayFromImage(seg)

        samples = {}
        samples['image'] = image
        samples['labels'] = labels

        if self.transform is not None:
            samples = self.transform(samples)
            image = samples['image']
            labels = samples['labels']

        image = np.expand_dims(image, axis=-1)
        image = torch.from_numpy(np.transpose(np.array(image), (3, 0, 1, 2))).type(torch.float32)
        labels = np.expand_dims(labels, axis=-1)
        labels = torch.from_numpy(np.transpose(np.array(labels), (3, 0, 1, 2))).type(torch.float32)

        if self.one_hot:
            labels = one_hot_encode(labels, num_classes=3, dim=0)

        return {'image': image, 'labels': labels}

if __name__ == '__main__':
    test = LiTSDataset('/data1/lsx/data/LiTS')