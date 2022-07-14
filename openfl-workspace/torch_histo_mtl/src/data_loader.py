import sys
import socket

if socket.gethostname() == 'NCI-02218974-ML':
    sys.path.insert(0, '/Users/zhongz2/HistoVAE/MAE/')
else:
    sys.path.insert(0, '/data/zhongz2/HistoVAE/MAE/')
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import openslide
from PIL import Image

from collections import namedtuple
from sklearn.model_selection import train_test_split

from openfl.federated import PyTorchDataLoader

import histo_dataset_v5


class PyTorchHistoDataLoader(PyTorchDataLoader):
    """PyTorch data loader for Kvasir dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        Args = namedtuple('Args', ['masks_dir', 'num_patches', 'debug', 'cache_root', 'norm_type', 'image_size'])
        masks_dir = './data/all'
        csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/openfl_split_{}.csv'.format(data_path)
        cache_root = 'None'
        args = Args(masks_dir=masks_dir,
                    num_patches=32,
                    debug=False,
                    cache_root=cache_root,
                    norm_type='mean_std',
                    image_size=128)
        df = pd.read_csv(csv_filename)
        # # replace the DX_filename
        # DX_filenames_in_local = []
        # svs_root = './data/svs'
        # for dx_filename in df['DX_filename'].values:
        #     DX_filenames_in_local.append(os.path.join(svs_root, os.path.basename(dx_filename)))
        # df['DX_filename'] = DX_filenames_in_local
        # # done

        train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['HistoAnno'])
        self.train_dataset = histo_dataset_v5.HistoDataset(df=train_df, mask_root=masks_dir, args=args,
                                                           num_patches=args.num_patches, debug=args.debug,
                                                           prefix='train',
                                                           cache_root=args.cache_root,
                                                           use_cache_data=args.cache_root != 'None')
        self.valid_dataset = histo_dataset_v5.HistoDataset(df=val_df, mask_root=args.masks_dir, args=args,
                                                           num_patches=args.num_patches, debug=args.debug, prefix='val',
                                                           cache_root=args.cache_root,
                                                           use_cache_data=args.cache_root != 'None')
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size
        self.num_classes = 3

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          num_workers=4)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=4)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape  # the first batch, (img, label)


def main():
    Args = namedtuple('Args', ['masks_dir', 'num_patches', 'debug', 'cache_root', 'norm_type', 'image_size'])
    dataset_root = '/data/zhongz2/BigData/TCGA-BRCA/'
    masks_dir = '/data/zhongz2/tcga_brca_256/patches'
    csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/openfl_split_1.csv'
    args = Args(masks_dir=masks_dir,
                num_patches=32,
                debug=False,
                cache_root='None',
                norm_type='mean_std',
                image_size=128)
    df = pd.read_csv(csv_filename)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['HistoAnno'])
    train_dataset = histo_dataset_v5.HistoDataset(df=train_df, mask_root=masks_dir, args=args,
                                                  num_patches=args.num_patches, debug=args.debug, prefix='train',
                                                  cache_root=args.cache_root,
                                                  use_cache_data=args.cache_root != 'None')
    val_dataset = histo_dataset_v5.HistoDataset(df=val_df, mask_root=args.masks_dir, args=args,
                                                num_patches=args.num_patches, debug=args.debug, prefix='val',
                                                cache_root=args.cache_root,
                                                use_cache_data=args.cache_root != 'None')

    # compute loss weights
    num_classes = 3
    label_col = 'HistoAnno'
    N = float(len(train_dataset))
    slide_cls_ids = [[] for _ in range(num_classes)]
    print(label_col)
    for i in range(num_classes):
        slide_cls_ids[i] = np.where(train_dataset.slide_data[label_col] == i)[0]
        print(i, len(slide_cls_ids[i]))
    weight_per_class = [N / len(slide_cls_ids[c]) for c in range(len(slide_cls_ids))]
    class_weights = torch.FloatTensor(weight_per_class)
    print('class_weights for {}: {}'.format(label_col, class_weights))  # [6.2500, 1.6164, 4.5181]
