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
from sksurv.metrics import concordance_index_censored
import tv_resnet_ae
from histo_utils import Accuracy_Logger, Regression_Logger
from histo_losses import VAEUsingDistLoss, NLLSurvLoss, CrossEntropySurvLoss, CoxSurvLoss, FocalLoss
from torch.hub import load_state_dict_from_url
from tensorboardX import SummaryWriter

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


class LossFn(nn.Module):
    def __init__(self, device, **kwargs):
        super(LossFn, self).__init__()
        self.label_col_dict = kwargs.get('label_col_dict')
        self.classification_dict_all = kwargs.get('classification_dict_all')
        self.classification_loss_weights_dict = kwargs.get('classification_loss_weights_dict')
        self.ignore_index_dict = kwargs.get('ignore_index_dict')
        self.regression_list_all = kwargs.get('regression_list_all')
        self.cls_task_keys = kwargs.get('cls_task_keys')
        self.reg_task_keys = kwargs.get('reg_task_keys')
        self.image_size = kwargs.get('image_size')
        self.model_name = kwargs.get('model_name')
        self.backbone = kwargs.get('backbone')
        self.fixed_backbone = kwargs.get('fixed_backbone')
        self.dropout = kwargs.get('dropout')
        self.surv_loss_type = kwargs.get('surv_loss_type')
        self.surv_loss_coeff = kwargs.get('surv_loss_coeff')
        self.surv_alpha = kwargs.get('surv_alpha')
        self.vae_loss_type = kwargs.get('vae_loss_type')
        self.z_dim = kwargs.get('z_dim')
        self.bce_loss_coeff = kwargs.get('bce_loss_coeff')
        self.kl_loss_coeff = kwargs.get('kl_loss_coeff')
        self.cls_loss_type = kwargs.get('cls_loss_type')
        self.cls_loss_coeff = kwargs.get('cls_loss_coeff')
        self.cls_loss_coeff = [float(v) for v in self.cls_loss_coeff.split(',')]
        self.focal_gamma = kwargs.get('focal_gamma')
        self.reg_loss_type = kwargs.get('reg_loss_type')
        self.reg_loss_coeff = kwargs.get('reg_loss_coeff')
        self.reg_loss_coeff = [float(v) for v in self.reg_loss_coeff.split(',')]
        self.regu_loss_type = kwargs.get('regu_loss_type')
        self.regu_loss_coeff = kwargs.get('regu_loss_coeff')
        self.moe_type = kwargs.get('moe_type')
        self.image_mean = kwargs.get('image_mean')
        self.image_std = kwargs.get('image_std')
        self.num_channels = kwargs.get('num_channels')

        self.has_surv = self.surv_loss_type != 'None'
        self.has_vae = self.vae_loss_type != 'None'
        self.has_moe = self.moe_type != 'None'

        if self.cls_task_keys == 'None':
            self.classification_dict = {}
        else:
            cls_task_keys = self.cls_task_keys.split(',')
            self.classification_dict = {}
            for cls_task_key in cls_task_keys:
                self.classification_dict[cls_task_key] = self.classification_dict_all[cls_task_key]

        if self.reg_task_keys == 'None':
            self.regression_list = []
        else:
            reg_task_keys = self.reg_task_keys.split(',')
            self.regression_list = []
            for reg_task_key in reg_task_keys:
                if reg_task_key in self.regression_list_all:
                    self.regression_list.append(reg_task_key)

        print('classification tasks: ', self.classification_dict)
        print('regression tasks: ', self.regression_list)

        if len(self.classification_dict) != len(self.cls_loss_coeff):
            print('inequal classification coefficients!')
            print('cls_loss_coeff', self.cls_loss_coeff)
            temp_loss_coeff = self.cls_loss_coeff[0]
            self.cls_loss_coeff = [temp_loss_coeff for _ in range(len(self.classification_dict))]
        if len(self.regression_list) != len(self.reg_loss_coeff):
            print('inequal regression coefficients!')
            print('reg_loss_coeff', self.reg_loss_coeff)
            temp_loss_coeff = self.reg_loss_coeff[0]
            self.reg_loss_coeff = [temp_loss_coeff for _ in range(len(self.regression_list))]

        print('\nSetup Autoencoder Losses ...')
        if self.vae_loss_type == 'vae_dist_loss':
            self.vae_loss_fn = VAEUsingDistLoss()
        else:
            self.vae_loss_fn = None

        print('\nSetup Survival Losses ...')
        if self.surv_loss_type == 'ce_surv':
            self.surv_loss_fn = CrossEntropySurvLoss(alpha=self.surv_alpha)
        elif self.surv_loss_type == 'nll_surv':
            self.surv_loss_fn = NLLSurvLoss(alpha=self.surv_alpha)
        elif self.surv_loss_type == 'cox_surv':
            self.surv_loss_fn = CoxSurvLoss()
        else:
            self.surv_loss_fn = None

        print('\nSetup Classification Losses ...\n')
        if self.cls_loss_type == 'ce':
            self.cls_loss_fn_dict = {k: nn.CrossEntropyLoss(ignore_index=self.ignore_index_dict[k]) for k in
                                     self.classification_dict.keys()}
        elif self.cls_loss_type == 'weighted_ce':
            self.cls_loss_fn_dict = {}
            for k in self.classification_dict.keys():
                # class_weights = get_class_weights(train_dataset, len(classification_dict[k]), label_col_dict[k])
                class_weights = torch.FloatTensor(self.classification_loss_weights_dict[k])
                print('class weights: ', k, class_weights)
                self.cls_loss_fn_dict[k] = nn.CrossEntropyLoss(weight=class_weights.cuda(),
                                                               ignore_index=self.ignore_index_dict[k])
        elif self.cls_loss_type == 'focal':
            self.cls_loss_fn_dict = {k: FocalLoss(gamma=self.focal_gamma,
                                                  ignore_index=self.ignore_index_dict[k]) for k in
                                     self.classification_dict.keys()}
        elif self.cls_loss_type == 'weighted_focal':
            self.cls_loss_fn_dict = {}
            for k in self.classification_dict.keys():
                # class_weights = get_class_weights(train_dataset, len(classification_dict[k]), label_col_dict[k])
                class_weights = torch.FloatTensor(self.classification_loss_weights_dict[k])
                print('class weights: ', k, class_weights)
                self.cls_loss_fn_dict[k] = FocalLoss(alpha=class_weights.cuda(), gamma=self.focal_gamma,
                                                     ignore_index=self.ignore_index_dict[k])
        else:
            self.cls_loss_fn_dict = None

        print('\nSetup Regression Losses ...')
        if self.reg_loss_type == 'mse':
            self.reg_loss_fn_dict = {k: nn.MSELoss() for k in self.regression_list}
        else:
            self.reg_loss_fn_dict = None

        print('\nSetup Regularization Losses ...')
        if self.regu_loss_type == 'l1':
            self.regu_loss_fn = None
        else:
            self.regu_loss_fn = None
        self.__name__ = 'histo_mtl_loss'

    def forward(self, results_dict, labeled_batch):
        losses_dict = {
            'bce': 0., 'kld': 0.,
            'surv': 0., 'regu': 0.
        }
        for k in self.classification_dict.keys():
            losses_dict[k] = 0.
        for k in self.regression_list:
            losses_dict[k] = 0.

        if self.has_vae:
            # for autoencoder
            bce_loss, kl_loss = self.vae_loss_fn(results_dict['patches'], results_dict['x_hat'],
                                                     results_dict['p'], results_dict['q'])
            losses_dict['bce'] += bce_loss.item()
            losses_dict['kld'] += kl_loss.item()

        if self.has_surv:
            hazards, S, Y_hat = results_dict['hazards'], results_dict['S'], results_dict['Y_hat']
            surv_loss = self.surv_loss_fn(hazards=hazards, S=S, Y=labeled_batch['label'], c=labeled_batch['c'])
            losses_dict['surv'] += surv_loss.item()

        classification_losses = {}
        for k in self.classification_dict.keys():
            classification_losses[k] = self.cls_loss_fn_dict[k](results_dict[k + '_logits'], labeled_batch[k])
            losses_dict[k] += classification_losses[k].item()
            # loggers_dict[k].log(results_dict[k + '_Y_hat'], labeled_batch[k], results_dict[k + '_Y_prob'])

        regression_losses = {}
        for k in self.regression_list:
            regression_losses[k] = self.reg_loss_fn_dict[k](results_dict[k + '_logits'], labeled_batch[k])
            losses_dict[k] += regression_losses[k].item()
            # reg_loggers_dict[k].log(results_dict[k + '_logits'], labeled_batch[k])

        total_loss = sum([self.cls_loss_coeff[vi] * v for vi, v in enumerate(classification_losses.values())]) + \
                     sum([self.reg_loss_coeff[vi] * v for vi, v in enumerate(regression_losses.values())])

        if self.has_surv:
            total_loss += self.surv_loss_coeff * surv_loss

        if self.has_vae:
            total_loss += self.bce_loss_coeff * bce_loss + self.kl_loss_coeff * kl_loss

        return total_loss


class PyTorchFederatedHistoCNN(PyTorchTaskRunner):
    """Simple CNN for classification."""

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional arguments to pass to the function
        """

        super().__init__(loss_fn=LossFn(device=device, **kwargs), **kwargs)

        print('kwargs: ', kwargs.keys())
        self.label_col_dict = kwargs.get('label_col_dict')
        self.classification_dict_all = kwargs.get('classification_dict_all')
        self.classification_loss_weights_dict = kwargs.get('classification_loss_weights_dict')
        self.ignore_index_dict = kwargs.get('ignore_index_dict')
        self.regression_list_all = kwargs.get('regression_list_all')
        self.cls_task_keys = kwargs.get('cls_task_keys')
        self.reg_task_keys = kwargs.get('reg_task_keys')
        self.image_size = kwargs.get('image_size')
        self.model_name = kwargs.get('model_name')
        self.backbone = kwargs.get('backbone')
        self.fixed_backbone = kwargs.get('fixed_backbone')
        self.dropout = kwargs.get('dropout')
        self.surv_loss_type = kwargs.get('surv_loss_type')
        self.surv_loss_coeff = kwargs.get('surv_loss_coeff')
        self.surv_alpha = kwargs.get('surv_alpha')
        self.vae_loss_type = kwargs.get('vae_loss_type')
        self.z_dim = kwargs.get('z_dim')
        self.bce_loss_coeff = kwargs.get('bce_loss_coeff')
        self.kl_loss_coeff = kwargs.get('kl_loss_coeff')
        self.cls_loss_type = kwargs.get('cls_loss_type')
        self.cls_loss_coeff = kwargs.get('cls_loss_coeff')
        self.cls_loss_coeff = [float(v) for v in self.cls_loss_coeff.split(',')]
        self.focal_gamma = kwargs.get('focal_gamma')
        self.reg_loss_type = kwargs.get('reg_loss_type')
        self.reg_loss_coeff = kwargs.get('reg_loss_coeff')
        self.reg_loss_coeff = [float(v) for v in self.reg_loss_coeff.split(',')]
        self.regu_loss_type = kwargs.get('regu_loss_type')
        self.regu_loss_coeff = kwargs.get('regu_loss_coeff')
        self.moe_type = kwargs.get('moe_type')
        self.image_mean = kwargs.get('image_mean')
        self.image_std = kwargs.get('image_std')
        self.num_channels = kwargs.get('num_channels')

        self.has_surv = self.surv_loss_type != 'None'
        self.has_vae = self.vae_loss_type != 'None'
        self.has_moe = self.moe_type != 'None'

        if self.cls_task_keys == 'None':
            self.classification_dict = {}
        else:
            cls_task_keys = self.cls_task_keys.split(',')
            self.classification_dict = {}
            for cls_task_key in cls_task_keys:
                self.classification_dict[cls_task_key] = self.classification_dict_all[cls_task_key]

        if self.reg_task_keys == 'None':
            self.regression_list = []
        else:
            reg_task_keys = self.reg_task_keys.split(',')
            self.regression_list = []
            for reg_task_key in reg_task_keys:
                if reg_task_key in self.regression_list_all:
                    self.regression_list.append(reg_task_key)

        print('classification tasks: ', self.classification_dict)
        print('regression tasks: ', self.regression_list)

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.feature_extractor = tv_resnet_ae.__dict__[self.backbone](pretrained=True)

        # for param in self.feature_extractor.parameters():   # fix the feature extraction network
        #     param.requires_grad = False
        latent_dim = self.feature_extractor.latent_dim
        L = latent_dim
        D = latent_dim
        K = 1
        a1 = [nn.Linear(L, D), nn.Tanh()]
        a2 = [nn.Linear(L, D), nn.Sigmoid()]
        if 0 < self.dropout < 1:
            a1.append(nn.Dropout(self.dropout))
            a2.append(nn.Dropout(self.dropout))
        self.attention_V = torch.nn.Sequential(*a1)
        self.attention_U = torch.nn.Sequential(*a2)
        self.attention_weights = torch.nn.Linear(D, K)

        if self.has_surv:
            self.classifier = nn.Linear(L, 4)
        classifiers = {}
        for k, labels in self.classification_dict.items():
            classifiers[k] = nn.Linear(L, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in self.regression_list:
            regressors[k] = nn.Linear(L, 1)
        self.regressors = nn.ModuleDict(regressors)

        if self.has_vae:
            # for AutoEncoder
            self.first_conv = False
            self.maxpool1 = False
            # self.fc = nn.Linear(latent_dim, self.z_dim)
            self.fc_mu = nn.Linear(latent_dim, self.z_dim)
            self.fc_var = nn.Linear(latent_dim, self.z_dim)
            self.decoder = tv_resnet_ae.__dict__['{}_decoder'.format(self.backbone)](latent_dim=self.z_dim,
                                                                                     image_height=self.image_size,
                                                                                     first_conv=self.first_conv,
                                                                                     maxpool1=self.maxpool1)

        self.register_buffer('mean', torch.Tensor(self.image_mean).reshape([1, self.num_channels, 1, 1]))
        self.register_buffer('std', torch.Tensor(self.image_std).reshape([1, self.num_channels, 1, 1]))

        self.initialize_weights()

        state_dict = load_state_dict_from_url(model_urls[self.backbone], progress=False)
        self.feature_extractor.load_state_dict(state_dict, strict=False)

        if self.fixed_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        print('kwargs', kwargs.keys())
        self.to(device)

        self._init_optimizer(lr=kwargs.get('lr'))
        self.initialize_tensorkeys_for_functions()
        self.save_dir = './save/'
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'), flush_secs=30)

    def _init_optimizer(self, lr):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=float(lr or 1e-4))

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

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def generate(self, num_samples):
        z = torch.randn(num_samples, self.z_dim).to(self.mean.device)
        return self.decoder(z)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        # x: (BS x num_patches, num_channels, h, w)
        results_dict = {'patches': x.detach().clone()}
        batch_size, num_patches, num_channels, h, w = x.size()
        x = torch.reshape(x, (batch_size * num_patches, num_channels, h, w))
        feat = self.feature_extractor(x)  # features after ResNet-18 512

        if self.has_vae:
            # for autoencoder task
            mu = self.fc_mu(feat)
            log_var = self.fc_var(feat)
            p, q, z = self.sample(mu, log_var)
            x_hat = self.decoder(z).sub_(self.mean).div_(self.std)
            x_hat = torch.reshape(x_hat, (batch_size, num_patches, num_channels, h, w))
        else:
            p, q, x_hat = None, None, None
        # for each slide, 10000 patches with size of 128x128
        # 2 x 10000 x 512 features
        # for other tasks
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

        if self.has_surv:
            logits = self.classifier(h)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        else:
            hazards, S, Y_hat = None, None, None

        results_dict.update({'x_hat': x_hat, 'p': p, 'q': q,
                        'hazards': hazards, 'S': S, 'Y_hat': Y_hat})
        for k, classifier in self.classifiers.items():
            logits_k = classifier(h)
            Y_hat_k = torch.topk(logits_k, 1, dim=1)[1]
            Y_prob_k = F.softmax(logits_k, dim=1)
            results_dict.update({k + '_logits': logits_k,
                                 k + '_Y_hat': Y_hat_k.squeeze(1),
                                 k + '_Y_prob': Y_prob_k})

        for k, regressor in self.regressors.items():
            values_k = regressor(h).squeeze(1)
            results_dict.update({k + '_logits': values_k})

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
        epoch = round_num
        save_dir = self.save_dir
        writer = self.writer

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='validate')

        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        losses_dict = {
            'bce': 0., 'kld': 0.,
            'surv': 0., 'regu': 0.
        }
        for k in self.classification_dict.keys():
            losses_dict[k] = 0.
        for k in self.regression_list:
            losses_dict[k] = 0.

        loggers_dict = {}
        for k, v in self.classification_dict.items():
            loggers_dict[k] = Accuracy_Logger(n_classes=len(v), task_name=k, label_names=v)
        reg_loggers_dict = {}
        for k in self.regression_list:
            reg_loggers_dict[k] = Regression_Logger()

        bce_loss, kl_loss = None, None
        for batch_idx, (patches, labeled_batch) in enumerate(loader):
            patches = patches.cuda()
            for k in labeled_batch.keys():
                # print(k, type(sampled_batch[k]), isinstance(sampled_batch[k], torch.FloatTensor))
                # if k in ['MDSC', 'M2', 'CAF', 'Dysfunction']:
                #     sampled_batch[k] = sampled_batch[k].float()
                if k != 'svs_filename':
                    labeled_batch[k] = labeled_batch[k].cuda()

            with torch.no_grad():
                results_dict = self(patches)

            # for autoencoder
            if self.has_vae:
                bce_loss, kl_loss = self.vae_loss_fn(patches, results_dict['x_hat'],
                                                     results_dict['p'], results_dict['q'])
                losses_dict['bce'] += bce_loss.item()
                losses_dict['kld'] += kl_loss.item()

            if self.has_surv:
                hazards, S, Y_hat = results_dict['hazards'], results_dict['S'], results_dict['Y_hat']
                surv_loss = self.surv_loss_fn(hazards=hazards, S=S, Y=labeled_batch['label'], c=labeled_batch['c'])
                losses_dict['surv'] += surv_loss.item()

            classification_losses = {}
            for k in self.classification_dict.keys():
                classification_losses[k] = self.cls_loss_fn_dict[k](results_dict[k + '_logits'], labeled_batch[k])
                losses_dict[k] += classification_losses[k].item()
                loggers_dict[k].log(results_dict[k + '_Y_hat'], labeled_batch[k], results_dict[k + '_Y_prob'])

            regression_losses = {}
            for k in self.regression_list:
                regression_losses[k] = self.reg_loss_fn_dict[k](results_dict[k + '_logits'], labeled_batch[k])
                losses_dict[k] += regression_losses[k].item()
                reg_loggers_dict[k].log(results_dict[k + '_logits'], labeled_batch[k])

            if self.regu_loss_fn is None:
                loss_regu = 0
            else:
                loss_regu = self.regu_loss_fn(self.model)
                losses_dict['regu'] += loss_regu.item()

            total_loss = sum([self.cls_loss_coeff[vi] * v for vi, v in enumerate(classification_losses.values())]) + \
                         sum([self.reg_loss_coeff[vi] * v for vi, v in enumerate(regression_losses.values())]) + \
                         self.regu_loss_coeff * loss_regu

            if self.has_surv:
                total_loss += self.surv_loss_coeff * surv_loss

            if self.has_vae:
                total_loss += self.bce_loss_coeff * bce_loss + self.kl_loss_coeff * kl_loss

            if batch_idx % 20 == 0:
                print(
                    'Batch {:03d}, total_loss: {:.4f}, bce: {:.4f}, kld: {:.4f}, surv: {:.4f}, {}, {}, Regu: {:.4f}'.format(
                        batch_idx, total_loss.item(),
                        bce_loss.item() if self.has_vae else 0, kl_loss.item() if self.has_vae else 0,
                        surv_loss.item() if self.has_surv else 0,
                        ', '.join(['{}: {:.4f}'.format(k, v) for k, v in classification_losses.items()]),
                        ', '.join(['{}: {:.4f}'.format(k, v) for k, v in regression_losses.items()]),
                        loss_regu
                    ))

            if self.has_surv:
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores.append(risk)
                all_censorships.append(labeled_batch['c'].detach().cpu().numpy())
                all_event_times.append(labeled_batch['event_time'].detach().cpu().numpy())

        print('Epoch: {}'.format(epoch))
        if self.has_surv:
            all_risk_scores = np.concatenate(all_risk_scores)
            all_censorships = np.concatenate(all_censorships)
            all_event_times = np.concatenate(all_event_times)
            c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                                 tied_tol=1e-08)[0]
            print('val_c_index: {:.4f}'.format(c_index))

        for k in losses_dict.keys():
            losses_dict[k] /= len(loader)
            if writer:
                writer.add_scalar('val/loss_{}'.format(k), losses_dict[k], epoch)
                writer.add_scalar('val/c_index', c_index if self.has_surv else 0, epoch)
                writer.add_scalar('val/total_loss', sum([v for v in losses_dict.values()]), epoch)

        print('Epoch {}, {}'.format(
            epoch, ', '.join(['{}: {:.4f}'.format(k, v) for k, v in losses_dict.items()])))

        print('Val cls accs:')
        for name, labels in self.classification_dict.items():
            print(20 * '=')
            print('{} confusion matrix'.format(name))
            print(loggers_dict[name].get_confusion_matrix())

            for average in ['micro', 'macro', 'weighted']:
                score = loggers_dict[name].get_f1_score(average=average)
                losses_dict['{}_f1_{}'.format(name, average)] = score
                print('f1_score({}) = {}'.format(average, score))
                if writer:
                    writer.add_scalar('val/{}_f1_{}'.format(name, average), score, epoch)

            for average in ['macro', 'weighted']:
                auc = loggers_dict[name].get_auc_score(average=average)
                losses_dict['{}_auc_{}'.format(name, average)] = auc
                print('auc_score({}) = {}'.format(average, auc))
                if writer:
                    writer.add_scalar('val/{}_auc_{}'.format(name, average), auc, epoch)

            print('generated ROC curves ...')
            loggers_dict[name].get_roc_curve(os.path.join(save_dir, 'epoch_{:03}_{}_val_ROC.jpg'.format(epoch, name)))

            print('save data')
            loggers_dict[name].save_data(os.path.join(save_dir, 'epoch_{:03}_{}_val_data.txt'.format(epoch, name)))

            for j in range(len(labels)):
                acc, correct, count = loggers_dict[name].get_summary(j)
                print('task {}, class {}({}): acc {}, correct {}/{}'.format(name, j, self.classification_dict[name][j],
                                                                            acc,
                                                                            correct, count))

                losses_dict['{}_{}_acc'.format(name, self.classification_dict[name][j])] = acc
                losses_dict['{}_{}_correct'.format(name, self.classification_dict[name][j])] = correct
                losses_dict['{}_{}_count'.format(name, self.classification_dict[name][j])] = count
                if writer:
                    writer.add_scalar('val/{}_{}_{}_acc'.format(name, j, self.classification_dict[name][j]), acc, epoch)

        print('Val reg mses:')
        for name in self.regression_list:
            print(20 * '=')
            mse = reg_loggers_dict[name].mean_squared_error()
            losses_dict['{}_mse'.format(name)] = mse
            print('{} mean squared error'.format(mse))
            if writer:
                writer.add_scalar('val/{}_mse'.format(name), mse, epoch)

        losses_dict['epoch'] = epoch
        losses_dict['c_index'] = c_index if self.has_surv else 0

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
