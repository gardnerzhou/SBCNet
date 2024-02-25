import logging

from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from tuSeg.data.datasets.lits import LiTSDataset
from tuSeg.data.datasets.litsEdge import LiTSEdgeDataset
from tuSeg.data.datasets.dataset_utils import RandomCrop, RandomRotFlip, ImageNormalize, CenterCrop

logger = logging.getLogger(__name__)

class SegDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str ="", data_dir: str = "", batch_size: int = 32, patch_size: list = [16, 128, 128], num_workers: int = 8, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        logger.info("Initialize Seg DataModule")

    def setup(self, stage=None):
        if self.dataset == 'lits':
            if stage == 'fit' or stage is None:
                self.seg_train = LiTSDataset(self.data_dir,
                                         split='train',
                                         transform = transforms.Compose([
                                         RandomCrop(self.patch_size),
                                         RandomRotFlip()
                                        ]))
                self.seg_val = LiTSDataset(self.data_dir,
                                       split='val',
                                       transform = transforms.Compose([
                                       CenterCrop([96,96,96])
                                       ]))
        elif self.dataset == 'litsEdge':
            if stage == 'fit' or stage is None:
                self.seg_train = LiTSEdgeDataset(self.data_dir,
                                         split='train',
                                         transform = transforms.Compose([
                                         RandomCrop(self.patch_size),
                                         RandomRotFlip()
                                        ]))
                self.seg_val = LiTSEdgeDataset(self.data_dir,
                                       split='val',
                                       transform = transforms.Compose([
                                       CenterCrop([128,256,256])
                                       ]))

    def train_dataloader(self):
        return DataLoader(self.seg_train, batch_size=self.batch_size, num_workers=self.num_workers,\
                            shuffle=True, drop_last=True, pin_memory=True)

    # Double workers for val and test loaders since there is no backward pass and GPU computation is faster
    def val_dataloader(self):
        return DataLoader(self.seg_val, batch_size=1, num_workers=self.num_workers, shuffle=False)
