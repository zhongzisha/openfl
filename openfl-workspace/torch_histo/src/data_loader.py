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


class CachedDataset(Dataset):
    def __init__(self, cache_dir, slide_data, max_epochs):
        super(CachedDataset, self).__init__()
        self.cache_dir = cache_dir
        self.slide_data = slide_data
        self.max_epochs = max_epochs

    def __getitem__(self, epoch):
        data = {}
        for svs_filename in self.slide_data['DX_filename'].values:
            svs_prefix = os.path.basename(svs_filename).replace('.svs', '')
            data[svs_prefix] = torch.load(os.path.join(self.cache_dir, svs_prefix, str(epoch) + '.pth'))
        return data

    def __len__(self):
        return self.max_epochs


def collate_fn(batch):
    return batch[0]


class HistoDataset(Dataset):
    def __init__(self, df, mask_root=None, args=None, num_patches=64,
                 mode='omic', apply_sig=False, seed=7, print_info=True, n_bins=4,
                 patient_strat=False, label_col=None, eps=1e-6,
                 debug=False, prefix=None,
                 cache_dir=None, use_cache_data=False, max_epochs=100
                 ):
        super().__init__()
        self.debug = debug
        self.prefix = prefix
        self.df = df
        self.mask_root = mask_root
        self.num_patches = num_patches
        self.args = args

        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.mode = mode
        self.epoch = -1
        self.cache_dir = cache_dir
        self.use_cache_data = use_cache_data
        self.max_epochs = max_epochs

        slide_data = df  # pd.read_csv(csv_path, low_memory=False)
        # slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # if "IDC" in slide_data['oncotree_code'].values:  # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])

        slide_data = slide_data[new_cols]
        self.slide_data = slide_data

        # get the data_dicts for each HE image
        data_dicts = []
        total_count_of_patches = 0
        for svs_filename in self.slide_data['DX_filename'].values:
            mask_filename = os.path.join(self.mask_root, os.path.basename(svs_filename).replace('.svs', '.h5'))
            if os.path.exists(mask_filename):
                # with open(mask_filename, 'rb') as fp:
                #     data_dict = pickle.load(fp)
                #     total_count_of_patches += len(data_dict['locations'])
                #     data_dicts.append(data_dict)
                data_dict = {}
                with h5py.File(mask_filename, 'r') as h5file:  # the mask_root is the CLAM patches dir
                    data_dict['locations'] = h5file['coords'][()]
                    data_dict['patch_level'] = h5file['coords'].attrs['patch_level']
                    data_dict['patch_size'] = h5file['coords'].attrs['patch_size']
                total_count_of_patches += len(data_dict['locations'])
                data_dicts.append(data_dict)
            else:
                data_dicts.append(None)

        self.data_dicts = data_dicts
        self.total_count_of_patches = total_count_of_patches

        if 'path' in self.mode:
            self.metadata = None
        else:
            self.metadata = slide_data.columns[:11]

        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

        self.cached_data = None
        if use_cache_data:
            # queue related
            cached_dataset = CachedDataset(cache_dir=self.cache_dir, slide_data=self.slide_data,
                                           max_epochs=self.max_epochs)
            cache_loader = DataLoader(cached_dataset, num_workers=2, collate_fn=collate_fn)
            self.cache_loader_iter = iter(cache_loader)
        else:
            self.cache_loader_iter = None

        if args.norm_type == 'patch_mean_std':
            self.patch_mean_std_dict = torch.load('/data/zhongz2/tcga_brca_256/mean_std.pth')
        elif args.norm_type == 'mean_std':
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # rgb imagenet 120m patch-mean patch-std
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif args.norm_type == 'zero_one':
            self.mean = np.array([0, 0, 0], dtype=np.float32)
            self.std = np.array([1, 1, 1], dtype=np.float32)
        else:
            raise ValueError('wrong norm type')

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

        print('total count of cases:   {}'.format(len(self.data_dicts)))
        print('total count of patches: {}'.format(self.total_count_of_patches))

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def getlabel_by_colname(self, colname, ids):
        return self.slide_data[colname][ids]

    def set_epoch(self, epoch):
        self.epoch = epoch

        if self.use_cache_data:
            self.cached_data = next(self.cache_loader_iter)

    def __getitem__(self, index):
        if self.use_cache_data:
            return self.__getitem_cache__(index)
        else:
            return self.__getitem_no_cache__(index)

    def __getitem_cache__(self, index):
        case_id = self.slide_data['case_id'][index]
        label = self.slide_data['disc_label'][index]
        event_time = self.slide_data[self.label_col][index]
        c = self.slide_data['censorship'][index]
        slide_ids = self.patient_dict[case_id]

        item = self.slide_data.iloc[index]
        svs_filename = item['DX_filename']
        svs_prefix = os.path.basename(svs_filename).replace('.svs', '')
        # with open(os.path.join(self.mask_root, os.path.basename(svs_filename).replace('.svs', '.pkl')), 'rb') as fp:
        #     data_dict = pickle.load(fp)
        data_dict = self.data_dicts[index]
        locations = data_dict['locations']
        # patch_level = data_dict['patch_level']
        # patch_size = data_dict['patch_size']
        # slide = openslide.open_slide(svs_filename)
        # image = pyvips.Image.new_from_file(svs_filename)
        # width = image.get('width')
        # height = image.get('height')
        # patch_size = data_dict['patch_size']
        # image_size = (data_dict['image_size'], data_dict['image_size'])
        # print(index, item['CLID'], item['AppMag'], item['MPP'], len(locations), svs_filename)
        # indices = np.random.choice(len(locations), size=self.num_patches, replace=len(locations) <= self.num_patches)
        indices = np.arange(len(locations))

        # mean curves
        # total loss mean curves from different parameters
        # res18 + 0.2dropout
        # res50 + 0.8dropout
        # dropout on backbone

        patches_pil = self.cached_data[svs_prefix]
        patches = []
        if self.args.norm_type == 'patch_mean_std':
            mean = self.patch_mean_std_dict[svs_prefix]['mean'].astype(np.float32) / 255
            std = self.patch_mean_std_dict[svs_prefix]['std'].astype(np.float32) / 255
        else:
            mean = self.mean
            std = self.std

        if self.prefix == 'train':
            data_transform = T.Compose([
                T.RandomCrop((self.args.image_size, self.args.image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            data_transform = T.Compose([
                T.CenterCrop((self.args.image_size, self.args.image_size)),
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        for patch in patches_pil:
            patches.append(data_transform(patch))
            if len(patches) == self.num_patches:
                break

        patches = torch.stack(patches)
        results_dict = {
            'svs_filename': svs_filename,
            'patches': patches,
            'Stage': item['Stage'].astype(np.float32),
            'subtype': int(item['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype']),
            'IHC_HER2': int(item['IHC_HER2']),  # if using CELoss
            'HistoAnno': int(item['HistoAnno']),  # if using CELoss
            'MDSC': item['MDSC'].astype(np.float32),
            'CAF': item['CAF'].astype(np.float32),
            'M2': item['M2'].astype(np.float32),
            'Dysfunction': item['Dys'].astype(np.float32),
            'CTL': item['CTL'].astype(np.float32),
            'c': int(c),
            'event_time': event_time.astype(np.float32),
            'label': int(label)
        }

        if self.debug:
            results_dict['debug_top10_indices'] = indices[:10]
            results_dict['debug_num_indices'] = len(indices)

        # return results_dict
        return patches, int(item['HistoAnno'])

    def __getitem_no_cache__(self, index):

        case_id = self.slide_data['case_id'][index]
        label = self.slide_data['disc_label'][index]
        event_time = self.slide_data[self.label_col][index]
        c = self.slide_data['censorship'][index]
        slide_ids = self.patient_dict[case_id]

        item = self.slide_data.iloc[index]
        svs_filename = item['DX_filename']
        svs_prefix = os.path.basename(svs_filename).replace('.svs', '')
        # with open(os.path.join(self.mask_root, os.path.basename(svs_filename).replace('.svs', '.pkl')), 'rb') as fp:
        #     data_dict = pickle.load(fp)
        data_dict = self.data_dicts[index]
        locations = data_dict['locations']
        patch_level = data_dict['patch_level']
        patch_size = data_dict['patch_size']
        slide = openslide.open_slide(svs_filename)
        # image = pyvips.Image.new_from_file(svs_filename)
        # width = image.get('width')
        # height = image.get('height')
        # patch_size = data_dict['patch_size']
        # image_size = (data_dict['image_size'], data_dict['image_size'])
        # print(index, item['CLID'], item['AppMag'], item['MPP'], len(locations), svs_filename)
        # indices = np.random.choice(len(locations), size=self.num_patches, replace=len(locations) <= self.num_patches)
        indices = np.arange(len(locations))

        # mean curves
        # total loss mean curves from different parameters
        # res18 + 0.2dropout
        # res50 + 0.8dropout
        # dropout on backbone

        target_patch_size = (256, 256)
        patches = []
        if self.args.norm_type == 'patch_mean_std':
            mean = self.patch_mean_std_dict[svs_prefix]['mean'].astype(np.float32) / 255
            std = self.patch_mean_std_dict[svs_prefix]['std'].astype(np.float32) / 255
        else:
            mean = self.mean
            std = self.std

        if self.prefix == 'train':
            data_transform = T.Compose([
                T.RandomCrop((self.args.image_size, self.args.image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            data_transform = T.Compose([
                T.CenterCrop((self.args.image_size, self.args.image_size)),
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        while True:
            np.random.shuffle(indices)

            # if self.debug:
            #     print('prefix: {}'.format(self.prefix))
            #     print('svs_filename: {}'.format(svs_filename))
            #     print('length of indices: {}'.format(len(indices)))
            #     print('first ten indices: ', indices[:10])

            if len(patches) == self.num_patches:
                break

            for i in indices:
                # left, top = locations[i]
                # t = image.crop(left, top, patch_size, patch_size)  # left, top, width, height
                # t_np = vips2numpy(t)  # H x W x 4  (448 x 448 x 4)  4-th is alpha channel, first 3 channels are RGB
                # patch = Image.fromarray(t_np[:, :, :3])  # [:, :, ::-1] PIL format

                coord = locations[i]
                patch = slide.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')

                if target_patch_size is not None:
                    patch = patch.resize(target_patch_size, resample=Image.BILINEAR)

                patch1 = np.array(patch).astype(np.float32)  # h, w, c
                if len(np.unique(np.std(patch1, axis=2))) < 10:  # if all equal values
                    continue

                mean = np.mean(patch1, axis=(0, 1))
                std = np.std(patch1, axis=(0, 1))
                if np.any(std < 1e-6):
                    continue

                most_white_pixels = len(np.where(patch1[:, :, 0] > 240)[0])
                if most_white_pixels > 0.5 * np.prod(patch1.shape[:2]):
                    continue

                most_black_pixels = len(np.where(patch1[:, :, 0] < 15)[0])
                if most_black_pixels > 0.5 * np.prod(patch1.shape[:2]):
                    continue

                patches.append(data_transform(patch))

                if len(patches) == self.num_patches:
                    break

        patches = torch.stack(patches)
        results_dict = {
            'svs_filename': svs_filename,
            'patches': patches,
            'Stage': item['Stage'].astype(np.float32),
            'subtype': int(item['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype']),
            'IHC_HER2': int(item['IHC_HER2']),  # if using CELoss
            'HistoAnno': int(item['HistoAnno']),  # if using CELoss
            'MDSC': item['MDSC'].astype(np.float32),
            'CAF': item['CAF'].astype(np.float32),
            'M2': item['M2'].astype(np.float32),
            'Dysfunction': item['Dys'].astype(np.float32),
            'CTL': item['CTL'].astype(np.float32),
            'c': int(c),
            'event_time': event_time.astype(np.float32),
            'label': int(label)
        }

        if self.debug:
            results_dict['debug_top10_indices'] = indices[:10]
            results_dict['debug_num_indices'] = len(indices)

        # return results_dict
        return patches, int(item['HistoAnno'])


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
        masks_dir = '/data/zhongz2/tcga_brca_256/patches'
        csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/openfl_split_{}.csv'.format(data_path)
        args = Args(masks_dir=masks_dir,
                    num_patches=32,
                    debug=False,
                    cache_root='',
                    norm_type='mean_std',
                    image_size=128)
        df = pd.read_csv(csv_filename)
        # replace the DX_filename
        DX_filenames_in_local = []
        svs_root = './data/svs'
        for dx_filename in df['DX_filename'].values:
            DX_filenames_in_local.append(os.path.join(svs_root, os.path.basename(dx_filename)))
        df['DX_filename'] = DX_filenames_in_local
        # done

        train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['HistoAnno'])
        self.train_dataset = HistoDataset(df=train_df, mask_root=masks_dir, args=args,
                                          num_patches=args.num_patches, debug=args.debug, prefix='train',
                                          cache_dir=args.cache_root,
                                          use_cache_data=args.cache_root != '')
        self.valid_dataset = HistoDataset(df=val_df, mask_root=args.masks_dir, args=args,
                                          num_patches=args.num_patches, debug=args.debug, prefix='val',
                                          cache_dir=args.cache_root,
                                          use_cache_data=args.cache_root != '')
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size
        self.num_classes = 3

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape    # the first batch, (img, label)


def main():
    Args = namedtuple('Args', ['masks_dir', 'num_patches', 'debug', 'cache_root', 'norm_type', 'image_size'])
    dataset_root = '/data/zhongz2/BigData/TCGA-BRCA/'
    masks_dir = '/data/zhongz2/tcga_brca_256/patches'
    csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/openfl_split_1.csv'
    args = Args(masks_dir=masks_dir,
                num_patches=32,
                debug=False,
                cache_root='',
                norm_type='zero_one',
                image_size=128)
    df = pd.read_csv(csv_filename)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['HistoAnno'])
    train_dataset = HistoDataset(df=train_df, mask_root=masks_dir, args=args,
                                 num_patches=args.num_patches, debug=args.debug, prefix='train',
                                 cache_dir=args.cache_root,
                                 use_cache_data=args.cache_root != '')
    val_dataset = HistoDataset(df=val_df, mask_root=args.masks_dir, args=args,
                               num_patches=args.num_patches, debug=args.debug, prefix='val',
                               cache_dir=args.cache_root,
                               use_cache_data=args.cache_root != '')

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
    print('class_weights for {}: {}'.format(label_col, class_weights))  #[6.2500, 1.6164, 4.5181]

