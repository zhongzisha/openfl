# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from .tv_resnet_ae import resnet18
from .utils import Accuracy_Logger
from torch.hub import load_state_dict_from_url

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def cross_entropy(output, target):
    """Calculate Cross-entropy loss."""
    return F.cross_entropy(input=output, target=target,
                           weight=torch.FloatTensor([6.2500, 1.6164, 4.5181]),
                           ignore_index=2)


class LossFn(nn.Module):
    def __init__(self, device):
        super(LossFn, self).__init__()
        self.weight = torch.FloatTensor([6.2500, 1.6164, 4.5181]).to(device)
        self.__name__ = 'weighted_ce_loss'

    def forward(self, output, target):
        return F.cross_entropy(input=output['HistoAnno_logits'], target=target,
                               weight=self.weight,
                               ignore_index=2)


class PyTorchFederatedHistoCNN(PyTorchTaskRunner):
    """Simple CNN for classification."""

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional arguments to pass to the function
        """
        super().__init__(loss_fn=LossFn(device=device), **kwargs)

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.backbone = 'resnet18'
        self.num_classes = self.data_loader.num_classes
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer(lr=kwargs.get('lr'))
        self.initialize_tensorkeys_for_functions()

        print('kwargs', kwargs.keys())

    def _init_optimizer(self, lr):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=float(lr or 1e-4))

    def init_network(self,
                     device,
                     print_model=True,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            **kwargs: Additional arguments to pass to the function

        """
        channel = self.data_loader.get_feature_shape()[
            0]  # (channel, dim1, dim2)

        # self.feature_extractor = tv_resnet_ae.__dict__[self.backbone](pretrained=True)
        self.feature_extractor = resnet18(pretrained=True)
        dropout = 0.25
        # for param in self.feature_extractor.parameters():   # fix the feature extraction network
        #     param.requires_grad = False
        latent_dim = self.feature_extractor.latent_dim
        L = latent_dim
        D = latent_dim
        K = 1
        a1 = [nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout)]
        a2 = [nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout)]

        self.attention_V = torch.nn.Sequential(*a1)
        self.attention_U = torch.nn.Sequential(*a2)
        self.attention_weights = torch.nn.Linear(D, K)

        self.classifier = nn.Linear(L, self.num_classes)

        self.initialize_weights()

        state_dict = load_state_dict_from_url(model_urls[self.backbone], progress=False)
        self.feature_extractor.load_state_dict(state_dict, strict=False)

        if print_model:
            print(self)
        self.to(device)

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        batch_size, num_patches, num_channels, h, w = x.size()
        x = torch.reshape(x, (batch_size * num_patches, num_channels, h, w))
        feat = self.feature_extractor(x)  # features after ResNet-18 512
        feat = torch.reshape(feat, (batch_size, num_patches, -1))

        L = feat.shape[2]  # num_hidden
        P = feat.shape[1]  # num_patches
        B = feat.shape[0]  # batch_size
        A_V = self.attention_V(feat)  # A_V: B x P x D
        A_U = self.attention_U(feat)  # A_U: B x P x D
        A = self.attention_weights(A_V * A_U)  # A: B x P x 1
        A = F.softmax(A, dim=1)  # B x P
        A = A.repeat(1, 1, L)  # B x P x L
        h = torch.sum(A * feat, dim=1)  # B x P x L * B x P x L --> B x L   2 x 25
        logits_k = self.classifier(h)
        Y_hat_k = torch.topk(logits_k, 1, dim=1)[1]
        Y_prob_k = F.softmax(logits_k, dim=1)
        results_dict = {'HistoAnno_logits': logits_k,
                        'HistoAnno_Y_hat': Y_hat_k.squeeze(),
                        'HistoAnno_Y_prob': Y_prob_k}

        return results_dict

    def validate(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress
                                 bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        val_score = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='validate')

        classification_dict = {
            # 'stage': ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
            # 'subtype': ['LumA', 'LumB', 'Basal', 'HER2E', 'normal-like'],
            # 'IHC_HER2': ['Negative', 'Positive', 'Other'],
            'HistoAnno': ['Invasive lobular carcinoma', 'Invasive ductal carcinoma', 'Other'],
        }

        loggers_dict = {}
        for k, v in classification_dict.items():
            loggers_dict[k] = Accuracy_Logger(n_classes=len(v), task_name=k, label_names=v)

        epoch = round_num
        save_dir = './save/'
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = (data.to(self.device),
                                target.to(self.device))
                results_dict = self(data)
                # get the index of the max log-probability
                for k in classification_dict.keys():
                    loggers_dict[k].log(results_dict[k + '_Y_hat'], target, results_dict[k + '_Y_prob'])

        losses_dict = {}
        for name, labels in classification_dict.items():
            print(20 * '=')
            print('{} confusion matrix'.format(name))
            print(loggers_dict[name].get_confusion_matrix())

            for average in ['micro', 'macro', 'weighted']:
                score = loggers_dict[name].get_f1_score(average=average)
                losses_dict['{}_f1_{}'.format(name, average)] = score
                print('f1_score({}) = {}'.format(average, score))
                # if writer:
                #     writer.add_scalar('val/{}_f1_{}'.format(name, average), score, epoch)

            for average in ['macro', 'weighted']:
                auc = loggers_dict[name].get_auc_score(average=average)
                losses_dict['{}_auc_{}'.format(name, average)] = auc
                print('auc_score({}) = {}'.format(average, auc))
                # if writer:
                #     writer.add_scalar('val/{}_auc_{}'.format(name, average), auc, epoch)

            print('generated ROC curves ...')
            loggers_dict[name].get_roc_curve(os.path.join(save_dir, 'epoch_{:03}_{}_val_ROC.jpg'.format(epoch, name)))

            print('save data')
            loggers_dict[name].save_data(os.path.join(save_dir, 'epoch_{:03}_{}_val_data.txt'.format(epoch, name)))

            for j in range(len(labels)):
                acc, correct, count = loggers_dict[name].get_summary(j)
                print('task {}, class {}({}): acc {}, correct {}/{}'.format(name, j, classification_dict[name][j], acc,
                                                                            correct, count))

                losses_dict['{}_{}_acc'.format(name, classification_dict[name][j])] = acc
                losses_dict['{}_{}_correct'.format(name, classification_dict[name][j])] = correct
                losses_dict['{}_{}_count'.format(name, classification_dict[name][j])] = count
                # if writer:
                #     writer.add_scalar('val/{}_{}_{}_acc'.format(name, j, classification_dict[name][j]), acc, epoch)

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        # TODO figure out a better way to pass in metric for
        #  this pytorch validate function
        output_tensor_dict = {
            TensorKey('acc', origin, round_num, True, tags):
                # np.array(val_score / total_samples)
                losses_dict['HistoAnno_auc_weighted']
        }

        # empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer(lr=self.optimizer.defaults.get('lr'))
