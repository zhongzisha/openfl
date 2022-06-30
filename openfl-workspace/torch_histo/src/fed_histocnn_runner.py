# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from .tv_resnet_ae import resnet18
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
        self.weight = torch.FloatTensor([6.2500, 1.6164, 4.5181], device=device)

    def forward(self, output, target):
        return F.cross_entropy(input=output, target=target,
                               weight=self.weight,
                               ignore_index=2)


class PyTorchFederatedHistoCNN(PyTorchTaskRunner):
    """Simple CNN for classification."""

    def __init__(self, device='cuda:0', **kwargs):
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

    def _init_optimizer(self, lr):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=float(lr or 1e-3))

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
        x = self.classifier(h)
        return x

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

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = (torch.tensor(data).to(self.device),
                                torch.tensor(target).to(self.device))
                output = self(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                val_score += pred.eq(target).sum().cpu().numpy()

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
                np.array(val_score / total_samples)
        }

        # empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer(lr=self.optimizer.defaults.get('lr'))
