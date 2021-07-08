import copy
import glob
import inspect
import itertools
import pickle
import random
import time
from typing import Union, Any, cast, Type, Tuple
import os
import sys

import sklearn.preprocessing
import torchvision
from matplotlib.legend_handler import HandlerTuple

import waipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import math
from collections import Iterable

from PIL import Image
import matplotlib.patheffects as PathEffects

from utils import HiddenPrints
from scipy.stats.stats import pearsonr

all_pretrained_models = {
    'resnet18': torchvision.models.resnet18,
    'alexnet': torchvision.models.alexnet,
    'squeezenet': torchvision.models.squeezenet1_0,
    'vgg16': torchvision.models.vgg16,
    'densenet': torchvision.models.densenet161,
    # 'inception': torchvision.models.inception_v3,
    # 'googlenet': torchvision.models.googlenet,
    'shufflenet': torchvision.models.shufflenet_v2_x1_0,
    'mobilenet_v2': torchvision.models.mobilenet_v2,
    'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
    'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
    'resnext50_32x4d': torchvision.models.resnext50_32x4d,
    'wide_resnet50_2': torchvision.models.wide_resnet50_2,
    'mnasnet': torchvision.models.mnasnet1_0
}

estimable_parameters = {
    'body_speed': {
        'index': 0,
        'name': 'speed',
        'unit': 'm/s',
        'multiplier': 1, #/3.6,
        'apply_tight_layout': True,
        'scaler_name': 'walking_speed'
    },
    'footstep_width': {
        'index': 1,
        'name': 'footstep width',
        'unit': 'cm',
        'multiplier': 100.0,
        'apply_tight_layout': True,
        'scaler_name': 'footstep_width'
    },
    'cycle_time': {
        'index': 2,
        'name': 'cycle time',
        'unit': 's',
        'multiplier': 1.0,
        'apply_tight_layout': True,
        'scaler_name': 'cycle_time'
    },
    'stride_length': {
        'index': 3,
        'name': 'stride length',
        'unit': 'cm',
        'multiplier': 100.0,
        'apply_tight_layout': True,
        'scaler_name': 'stride_length'
    },
    'corridor_distance': {
        'index': 4,
        'name': 'distance from corridor',
        'unit': 'cm',
        'multiplier': -100.0,
        'apply_tight_layout': False,
        'scaler_name': 'corridor_position'
    },
}

cpu = torch.device('cpu')
cuda = torch.device('cuda')

device = cuda if torch.cuda.is_available() else cpu
use_cuda = True


def correlation_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = abs(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) - 1)
    cost[cost != cost] = 1
    return cost


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
        self.use_concat_pir = False
        self.number_of_outputs = 1

    def update_device(self):
        global device, use_cuda
        if 'use_cuda' not in self.__dict__:
            self.use_cuda = True
        use_cuda = self.use_cuda
        if self.use_cuda:
            device = cuda
        else:
            device = cpu

    def __str__(self):
        var_name_length = 25
        out = f"{self.__class__.__name__}:" + "\n\t"
        out += "\n\t".join([
            f"{str(var_name).replace('_', ' ')}{' ' * (var_name_length - len(str(var_name)))} = {getattr(self, var_name)}"
            for var_name in self.__init__.__code__.co_varnames[1:] if var_name in self.__dict__])
        return out

    def __repr__(self):
        return str(self)


class PretrainedModel(CustomModel):
    def __init__(self, trained_model, pretrained=False, use_exponential=False, number_of_outputs=1, **useless):
        super().__init__()
        self.number_of_outputs = number_of_outputs
        if not isinstance(use_exponential, Iterable):
            use_exponential = [use_exponential] * number_of_outputs
        self.use_exponential = use_exponential
        self.pretrained = pretrained

        try:
            self.module = trained_model(pretrained=self.pretrained, init_weights=True)
        except:
            self.module = trained_model(pretrained=self.pretrained)
        for param in self.module.parameters():
            param.requires_grad = not self.pretrained

        self.trained_model = self.module._get_name()

        self.num_ftrs = 0
        if trained_model is all_pretrained_models['resnet18'] or trained_model is all_pretrained_models['shufflenet']:
            self.num_ftrs = self.module.fc.in_features
            self.module.fc = nn.Identity()  # nn.Linear(self.num_ftrs, 1)
        elif trained_model is all_pretrained_models['alexnet'] \
                or trained_model is all_pretrained_models['mobilenet_v2'] \
                or trained_model is all_pretrained_models['mobilenet_v3_small'] \
                or trained_model is all_pretrained_models['mobilenet_v3_large'] \
                or trained_model is all_pretrained_models['mnasnet']:
            self.num_ftrs = self.module.classifier[-1].in_features
            self.module.classifier[-1] = nn.Identity()  # nn.Linear(self.module.classifier[-2].out_features, 1)
        elif trained_model is all_pretrained_models['squeezenet']:
            self.num_ftrs = 13 * 13
            # self.module.classifier = nn.Identity()  # nn.Linear(self.module.classifier[-2].out_features, 1)
            self.module.classifier = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        elif trained_model is all_pretrained_models['vgg16']:
            self.use_cuda = False
            self.num_ftrs = self.module.classifier[0].in_features
            self.module.classifier = nn.Identity()
        elif trained_model is all_pretrained_models['densenet']:
            self.use_cuda = False
            self.num_ftrs = self.module.classifier.in_features
            self.module.classifier = nn.Identity()
        # elif trained_model is all_pretrained_models['googlenet']:
        #     pass
        elif trained_model is all_pretrained_models['resnext50_32x4d'] or \
                trained_model is all_pretrained_models['wide_resnet50_2']:
            # or trained_model is all_pretrained_models['inception']
            self.use_cuda = False
            self.num_ftrs = self.module.fc.in_features
            self.module.fc = nn.Identity()

        self.first_conv = nn.Conv2d(4, 3, kernel_size=(3, 3), padding=1)
        self.endings = nn.ModuleList([nn.Linear(self.num_ftrs, 1) for _ in range(number_of_outputs)])

        self.use_concat_pir = True

        self.update_device()

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        x = self.first_conv(concat_pir)
        x = self.module(x)

        final_tensor = [2 ** self.endings[i](x) if self.use_exponential[i] else self.endings[i](x) for i in
                        range(self.number_of_outputs)]
        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor


class MultipleWaveletsModule(CustomModel):
    def __init__(self, input_size=512, mid_output_size=10, kernel_size=3, number_of_filters=2,
                 number_of_layers=8, hidden_cells=[], dropout=False, use_concat_pir=True, number_of_conv_per_layer=1,
                 pure_CNN=False, layer_relu=None, reduce_last_conv=0, use_exponential=False, number_of_outputs=1,
                 using_batch_norm = True,
                 **useless):
        super().__init__()
        self.input_size = input_size
        self.mid_output_size = mid_output_size
        self.kernel_size = kernel_size
        self.number_of_filters = number_of_filters
        self.number_of_layers = number_of_layers
        self.hidden_cells = hidden_cells
        self.dropout = dropout
        self.use_concat_pir = use_concat_pir
        self.number_of_conv_per_layer = number_of_conv_per_layer
        self.layer_relu = layer_relu
        self.using_batch_norm = using_batch_norm
        self.pure_CNN = pure_CNN
        self.reduce_last_conv = reduce_last_conv
        self.number_of_outputs = number_of_outputs
        if not isinstance(use_exponential, Iterable):
            use_exponential = [use_exponential] * number_of_outputs
        self.use_exponential = use_exponential

        # Create one module per PIR (used when not using the concat_pir version)
        self.multiple_wavelet_modules = nn.ModuleList([WaveletModule(input_size=input_size, output_size=mid_output_size,
                                                                     kernel_size=kernel_size,
                                                                     number_of_filters=number_of_filters,
                                                                     number_of_layers=number_of_layers,
                                                                     hidden_cells=hidden_cells,
                                                                     dropout=dropout,
                                                                     number_of_conv_per_layer=number_of_conv_per_layer,
                                                                     use_concat=False,
                                                                     pure_CNN = pure_CNN,
                                                                     reduce_last_conv=reduce_last_conv,
                                                                     layer_relu=layer_relu,
                                                                     using_batch_norm = self.using_batch_norm,
                                                                     lastActivation=None) for _ in
                                                       range(4)])
        # Create one unique module that take the 4 PIR data at once
        self.unique_wavelet_module = WaveletModule(input_size=input_size, output_size=4 * mid_output_size,
                                                   kernel_size=kernel_size,
                                                   number_of_filters=number_of_filters,
                                                   number_of_layers=number_of_layers,
                                                   hidden_cells=hidden_cells,
                                                   dropout=dropout,
                                                   number_of_conv_per_layer=number_of_conv_per_layer,
                                                   use_concat=True,
                                                   pure_CNN = pure_CNN,
                                                   reduce_last_conv=reduce_last_conv,
                                                   layer_relu=layer_relu,
                                                   lastActivation=None)

        # In any case, the activation is a Dense of shape (4*output x 1)
        self.activation_functions = nn.ModuleList([nn.Linear(4 * mid_output_size, 1) for _ in range(number_of_outputs)])

        self.update_device()

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            x = self.unique_wavelet_module(concat_pir)
        else:
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.multiple_wavelet_modules[i](pirs[i])
            x = torch.cat(tuple([tens for tens in pirs]), 1)

        # Need this because I created the "number of outputs" variable after creating some models...
        if not hasattr(self, 'number_of_outputs'):
            return self.activation_function(x) ** 2 if self.use_exponential == 'pow' else \
                2 ** self.activation_function(x) if self.use_exponential else self.activation_function(x)

        final_tensor = [
            self.activation_functions[i](x) ** 2 if self.use_exponential == 'pow' else \
                2 ** self.activation_functions[i](x) if self.use_exponential[i] else self.activation_functions[i](x)
            for i in range(self.number_of_outputs)]
        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor

    # def __str__(self):
    #     var_name_length = 25
    #     out = super().__str__()
    #     out += "\n\t" \
    #            f"{str('layer_relu').replace('_', ' ')}{' ' * (var_name_length - len(str('layer_relu')))} = " \
    #            f"{self.layer_relu.__class__.__name__} "
    #     return out


class WaveletModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1, use_concat=True,
                 hidden_cells=None, dropout: Union[float, bool] = False, number_of_conv_per_layer=1,
                 layer_relu=None, using_batch_norm = True, pure_CNN = False,
                 lastActivation=None, reduce_last_conv=0, **useless):
        super().__init__()

        if lastActivation is None:
            lastActivation = layer_relu
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.number_of_filters = number_of_filters
        self.number_of_layers = number_of_layers
        self.hidden_cells = hidden_cells
        self.dropout = dropout
        self.use_concat = use_concat
        self.number_of_conv_per_layer = number_of_conv_per_layer
        self.reduce_last_conv = reduce_last_conv
        self.layer_relu = layer_relu
        self.using_batch_norm = using_batch_norm
        self.pure_CNN = pure_CNN

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]

        # If we want a binary output, we use 2 outputs and apply a Softmax function at the end
        is_binary = (output_size == 1)
        if output_size == 1:
            output_size = 2

        padding = int(kernel_size / 2)
        # First, a convolution. Here we specified an image with a unique channel (no RGB)
        elems = [nn.Conv2d(4 if use_concat else 1, number_of_filters, kernel_size, padding=padding)]

        # For every new \"layer\" (or block), we add a MaxPooling and a Convolution
        # The number of filters is not changed
        for _ in range(1, number_of_layers):
            elems.append(nn.MaxPool2d(2))
            input_size = input_size // 2
            for _ in range(number_of_conv_per_layer):
                if self.using_batch_norm:
                    elems.append(nn.BatchNorm2d(number_of_filters))
                elems.append(nn.Conv2d(number_of_filters, number_of_filters, kernel_size, padding=padding))
                if layer_relu:
                    elems.append(layer_relu)
        if reduce_last_conv > 0:
            elems.append(nn.Conv2d(number_of_filters, reduce_last_conv, kernel_size, padding=padding))
            number_of_filters = reduce_last_conv

        if self.pure_CNN:
            elems.append(nn.AvgPool2d(input_size))
            input_size = 1

        self.cnn_layers = nn.Sequential(*elems)

        input_size = int(input_size ** 2 * number_of_filters)
        linear = []

        # If an hidden layer is asked, create them
        for i_hidden in range(len(hidden_cells)):
            if i_hidden == 0:
                linear.append(nn.Linear(input_size, hidden_cells[i_hidden]))
            else:
                linear.append(nn.Linear(hidden_cells[i_hidden - 1], hidden_cells[i_hidden]))

        # Let a dropout on the hidden layer or on the convolutional outputs
        if dropout:
            # The dropout parameter can be a float (0 <= dropout <= 1)
            # or a boolean, in which case, we give a 50% probability
            dropout_val = dropout if isinstance(dropout, bool) else dropout
            linear.append(nn.Dropout(dropout_val))

        # Close the linear layers
        if len(hidden_cells) > 0:
            linear.append(nn.Linear(hidden_cells[-1], output_size))
        else:
            linear.append(nn.Linear(input_size, output_size))

        # Apply a final activation function on the output.
        if is_binary:
            linear.append(nn.Softmax(dim=1))
        elif lastActivation:
            linear.append(lastActivation)
        self.linear_layers = nn.Sequential(*linear)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x


class MultipleWaveletsLinearModule(CustomModel):
    def __init__(self, input_size=512, mid_output_size=10, kernel_size=3, number_of_filters=2,
                 number_of_layers=8, hidden_cells=None, dropout=False, use_concat_pir=True,
                 use_exponential=False, number_of_outputs=1, **useless):
        super().__init__()

        self.input_size = input_size
        self.mid_output_size = mid_output_size
        self.kernel_size = kernel_size
        self.number_of_filters = number_of_filters
        self.number_of_layers = number_of_layers
        self.hidden_cells = hidden_cells
        self.dropout = dropout
        self.use_concat_pir = use_concat_pir
        self.number_of_outputs = number_of_outputs
        if not isinstance(use_exponential, Iterable):
            use_exponential = [use_exponential] * number_of_outputs
        self.use_exponential = use_exponential

        if hidden_cells is None:
            hidden_cells = []
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
        # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU

        # Create one module per PIR (used when not using the concat_pir version)
        self.multiple_wavelet_modules = nn.ModuleList(
            [WaveletLinearModule(input_size=input_size, output_size=mid_output_size,
                                 kernel_size=kernel_size,
                                 number_of_filters=number_of_filters,
                                 number_of_layers=number_of_layers,
                                 hidden_cells=hidden_cells,
                                 dropout=dropout, lastActivation=None) for i in
             range(4)])
        # Create one unique module that take the 4 PIR data at once
        self.unique_wavelet_module = WaveletLinearModule(input_size=input_size * 2, output_size=mid_output_size * 4,
                                                         kernel_size=kernel_size,
                                                         number_of_filters=number_of_filters,
                                                         number_of_layers=number_of_layers,
                                                         hidden_cells=hidden_cells,
                                                         dropout=dropout, lastActivation=None)

        # In any case, the activation is a Dense of shape (4*output x 1)
        self.activation_functions = nn.ModuleList([nn.Linear(4 * mid_output_size, 1) for _ in range(number_of_outputs)])

        self.update_device()

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            x = self.unique_wavelet_module(concat_pir)
        else:
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.multiple_wavelet_modules[i](pirs[i])
            x = torch.cat(tuple([tens for tens in pirs]), 1)

        final_tensor = [
            self.activation_functions[i](x) ** 2 if self.use_exponential == 'pow' else \
            2 ** self.activation_functions[i](x) if self.use_exponential[i] else self.activation_functions[i](x)
            for i in range(self.number_of_outputs)]
        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor


class WaveletLinearModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1,
                 hidden_cells=None, dropout: Union[float, bool] = False, lastActivation=None, **useless):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.number_of_filters = number_of_filters
        self.number_of_layers = number_of_layers
        self.hidden_cells = hidden_cells
        self.dropout = dropout

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]

        elems = []
        # For every new "layer" (or block), we add a MaxPooling
        for i in range(1, number_of_layers):
            elems.append(nn.MaxPool2d(2))
            input_size = input_size // 2
        self.pooling_layers = nn.Sequential(*elems)

        input_size = int(input_size ** 2)
        linear = []

        # If an hidden layer is asked, create them
        for i_hidden in range(len(hidden_cells)):
            if i_hidden == 0:
                linear.append(nn.Linear(input_size, hidden_cells[i_hidden]))
            else:
                linear.append(nn.Linear(hidden_cells[i_hidden - 1], hidden_cells[i_hidden]))

        # Let a dropout on the hidden layer or on the convolutional outputs
        if dropout:
            # The dropout parameter can be a float (0 <= dropout <= 1)
            # or a boolean, in which case, we give a 50% probability
            dropout_val = dropout if isinstance(dropout, bool) else dropout
            linear.append(nn.Dropout(dropout_val))

        # Close the linear layers
        if len(hidden_cells) > 0:
            linear.append(nn.Linear(hidden_cells[-1], output_size))
        else:
            linear.append(nn.Linear(input_size, output_size))

        # Apply a final activation function on the output.
        if lastActivation:
            linear.append(lastActivation())
        self.linear_layers = nn.Sequential(*linear)

    def forward(self, x):
        x = self.pooling_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x

class ResidualNet(CustomModel):
    def __init__(self, input_size: int, output_size: int = 1, number_of_outputs:int = 1, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1,
                 hidden_cells=None, number_of_conv_per_layer = 1, layer_relu = None,
                 lastActivation = None, use_concat_pir = True, **useless):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.kernel_size = kernel_size
        self.number_of_filters = number_of_filters
        self.hidden_cells = hidden_cells
        self.number_of_conv_per_layer = number_of_conv_per_layer
        self.layer_relu = layer_relu
        self.use_concat_pir = use_concat_pir

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]

        # If we want a binary output, we use 2 outputs and apply a Softmax function at the end
        is_binary = (output_size == 1)
        if output_size == 1:
            output_size = 2

        padding = int(kernel_size / 2)
        # First, a convolution. Here we specified an image with a unique channel (no RGB)
        self.init_conv = nn.Conv2d(4 if self.use_concat_pir else 1, number_of_filters, kernel_size, padding=padding)

        residual_blocks = []
        for i_block in range(number_of_layers):
            residual_blocks.append([])
            for _ in range(number_of_conv_per_layer):
                residual_blocks[i_block].append(nn.BatchNorm2d(number_of_filters))
                if layer_relu:
                    residual_blocks[i_block].append(layer_relu)
                residual_blocks[i_block].append(nn.Conv2d(number_of_filters, number_of_filters, kernel_size, padding=padding))
            # residual_blocks[i_block].append(nn.MaxPool2d(2))
            residual_blocks[i_block] = nn.Sequential(*residual_blocks[i_block])
            input_size = input_size / 2
        self.residual_blocks = nn.ModuleList(residual_blocks)

        input_size = int((input_size ** 2) * number_of_filters)
        linear = []

        # If an hidden layer is asked, create them
        for i_hidden in range(len(hidden_cells)):
            if i_hidden == 0:
                linear.append(nn.Linear(input_size, hidden_cells[i_hidden]))
            else:
                linear.append(nn.Linear(hidden_cells[i_hidden - 1], hidden_cells[i_hidden]))

        # Close the linear layers
        if len(hidden_cells) > 0:
            linear.append(nn.Linear(hidden_cells[-1], output_size))
        else:
            linear.append(nn.Linear(input_size, output_size))

        # Apply a final activation function on the output.
        if is_binary:
            linear.append(nn.Softmax(dim=1))
        elif lastActivation:
            linear.append(lastActivation)
        self.linear_layers = nn.Sequential(*linear)

        # In any case, the activation is a Dense of shape (4*output x 1)
        # self.activation_functions = nn.ModuleList([nn.Linear(4 * mid_output_size, 1) for _ in range(number_of_outputs)])

        self.update_device()

    def forward_process(self, x: torch.Tensor):
        x = self.init_conv(x)
        for i in range(len(self.residual_blocks)):
            original_x = torch.clone(x)
            x = original_x + self.residual_blocks[i](original_x)
            x = nn.MaxPool2d(2)(x)

        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            return self.forward_process(concat_pir)
        else:
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.forward_process(pirs[i])
            return torch.cat(tuple([tens for tens in pirs]), 1)


# Train function, give it your dataset, the parameters of your model, and there you go.
# Arguments:
# - trainloader : the training part of your dataset
# - validationloader : the testing / validation part of your dataset
# - model_class : Specify the class of the model you want to train
# - model_parameters : parameters that will be given to the __init__() function of your model
# - loss_function : Specify the loss function you want to use. default is Cross Entropy
# - training_for_autoencoders : will simply change the target for the loss
# - optimizer : Specify your optimizer. default is SGD
# - learning rate : parameter given to the optimizer
# - num_epochs : number of epochs for the training
# - early stopping : if True, will stop if the validation loss starts increasing
#                   If a integer value, defines the "patience" (number of minibatch to study before officially stops for overfitting)
# - display_precision : if verbose, number of minibatch seen when printing results.
#                 And because we didn't work very well, it also defines the precision for early stopping...
# - verbose : level of verbosity
def train_model(trainloader,
                validationloader,
                model_class: Type[CustomModel] = MultipleWaveletsModule,
                model_parameters=None,
                loss_function: Type[nn.Module] = nn.CrossEntropyLoss,
                training_for_autoencoder=False,
                optimizer: Any = None,
                learning_rate=0.010,
                num_epochs=5,
                early_stopping=False,
                display_precision: float = 100,
                verbose: Union[bool, int] = True,
                estimated_parameter='body_speed'):
    if model_parameters is None:
        raise Exception('No model parameters have been provided. Possible keys are: ' + '\n - '.join(
            model_class.__init__.__code__.co_varnames[1:]))

    estimated_parameter_index = estimable_parameters[estimated_parameter]['index']

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
    minibatch_size = 1
    # If not done, transform datasets in dataLoader
    # We've chose high batch size to be more efficient in time
    if isinstance(trainloader, torch.utils.data.Dataset):
        trainloader = torch.utils.data.DataLoader(trainloader, batch_size=minibatch_size,
                                                  shuffle=False, num_workers=0)
    if isinstance(validationloader, torch.utils.data.Dataset):
        validationloader = torch.utils.data.DataLoader(validationloader, batch_size=2 * minibatch_size,
                                                       shuffle=False, num_workers=0)
    # Create the CNN from the given parameters :
    net = model_class(**model_parameters)
    net.update_device()
    net.to(device)  # move it to GPU if available
    if verbose:
        print(net)

    # Defining loss function and optimizer
    # criteria = [loss_function() for _ in range(net.number_of_outputs)]
    criterion = loss_function()
    optimizer = optimizer(net.parameters(), lr=learning_rate)
    # if not optimizer:
    #     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Definition of arrays to store the results and draw the learning curves
    val_err_array = np.array([])
    train_err_array = np.array([])
    nb_sample_array = np.array([])

    # best system results
    best_val_loss = math.inf
    best_nb_sample = 0
    best_model = copy.deepcopy(net)

    nb_used_sample = 0
    running_loss = 0.0
    display_precision = display_precision if display_precision > 1 else len(trainloader) * display_precision - 1
    print_every = display_precision

    # Stop if is_overfitting turns to True
    is_overfitting = False
    # number of iteration that are still allowed (depending on "early_stopping")
    left_before_early_stop = False

    # Know when you have to display losses
    next_print_in = display_precision

    total_acc = 0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if is_overfitting:
            break

        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            if is_overfitting:
                break
            # get the inputs
            pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, \
            _, _, footstep_width, cycle_time, stride_length, corridor_distance = data

            # if possible, move them to GPU
            body_speed, footstep_width, cycle_time, stride_length, corridor_distance = \
                body_speed.to(device), footstep_width.to(device), cycle_time.to(device), stride_length.to(device), \
                corridor_distance.to(device)
            if net.use_concat_pir:
                concat_pir = concat_pir.to(device)
            else:
                pirs0, pirs1, pirs2, pirs3 = pirs0.to(device), pirs1.to(device), \
                                             pirs2.to(device), pirs3.to(device)

            targets = [body_speed, footstep_width, cycle_time, stride_length, corridor_distance]

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                current_loss = 0

                # _pirs0, _pirs1, _pirs2, _pirs3, _concat_pir, _labels = pirs0.clone(), pirs1.clone(), pirs2.clone(), pirs3.clone(), concat_pir.clone(), labels.clone()
                # forward + backward + optimize
                outputs = net(pirs0, pirs1, pirs2, pirs3, concat_pir)
                if training_for_autoencoder:  # If we train an autoencoder, set the input also as output
                    raise Exception("No autoencoder allowed in this program")
                    # loss = criterion(outputs, inputs)
                else:
                    # for output, target, criterion in zip(outputs, targets, criteria):
                    #     current_loss = current_loss + criterion(output, target)
                    current_loss = criterion(outputs.reshape(-1).float(),
                                             targets[estimated_parameter_index].reshape(
                                                 -1).float())  # Otherwise, use the label
                try:
                    current_loss.backward()
                except:
                    pass
                return current_loss

            # if isinstance(optimizer, optim.LBFGS) or isinstance(optimizer, optim.):
            #     optimizer.step(closure)
            #     loss = 1.0  # No idea what to do here...
            # else:
            loss = closure()
            optimizer.step(closure)  # Need to be improved

            # count how many samples have been used during the training
            nb_used_sample += len(pirs0)
            running_loss += float(loss.item())  # .item()
            del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, footstep_width, \
                cycle_time, stride_length, corridor_distance

            next_print_in -= 1
            if next_print_in <= 0:  # print every ""print_every"" mini-batches
                next_print_in = display_precision
                train_err = (running_loss / (print_every * minibatch_size))
                running_loss = 0.0
                # evaluation on validation set
                totalValLoss = 0.0
                # Evaluation part, don't update the weights
                with torch.no_grad():
                    for data in validationloader:
                        # get the inputs
                        pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, \
                        _, _, footstep_width, cycle_time, stride_length, corridor_distance = data

                        # if possible, move them to GPU
                        body_speed, footstep_width, cycle_time, stride_length, corridor_distance = \
                            body_speed.to(device), footstep_width.to(device), cycle_time.to(device), stride_length.to(
                                device), \
                            corridor_distance.to(device)
                        if net.use_concat_pir:
                            concat_pir = concat_pir.to(device)
                        else:
                            pirs0, pirs1, pirs2, pirs3 = pirs0.to(device), pirs1.to(device), \
                                                         pirs2.to(device), pirs3.to(device)

                        targets = [body_speed, footstep_width, cycle_time, stride_length, corridor_distance]

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # # forward + backward + optimize
                        # outputs = net(pirs0, pirs1, pirs2, pirs3, concat_pir)
                        # if training_for_autoencoder:  # If we train an autoencoder, set the input also as output
                        #     raise Exception("No autoencoder allowed")
                        #     # loss = criterion(outputs, images)
                        # else:
                        #     loss = criterion(outputs.float(), labels.float())  # Otherwise, use the label
                        totalValLoss += float(closure().item())  # .item()
                        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, footstep_width, \
                            cycle_time, stride_length, corridor_distance

                val_err = (totalValLoss / len(validationloader))
                if verbose > 2:
                    print('\tEpoch %d batch %5d ' % (epoch + 1, i + 1))
                    print('\tTrain loss : %.3f' % train_err)
                    print('\tValidation loss mean : %.3f' % val_err)
                train_err_array = np.append(train_err_array, train_err)
                val_err_array = np.append(val_err_array, val_err)
                nb_sample_array = np.append(nb_sample_array, nb_used_sample)

                # save the model only when loss is better
                if val_err < best_val_loss:
                    best_val_loss = val_err
                    best_nb_sample = nb_used_sample
                    best_model = copy.deepcopy(net)
                    if early_stopping:  # If we want early stopping, we extend the patience
                        left_before_early_stop = int(early_stopping)  # Gives 1 if early_stopping === True
                    else:
                        left_before_early_stop = math.inf  # Just to be sure that we don't stop the learning

                else:  # If the validation loss is lower than before...
                    if early_stopping:
                        left_before_early_stop -= 1
                        if left_before_early_stop < 0:  # Stop the learning to avoid overfitting
                            is_overfitting = True
                            if verbose > 2:
                                print("Stopped due to overfitting!",
                                      (nb_used_sample - best_nb_sample) // minibatch_size,
                                      "minibatchs before, we were", val_err - best_val_loss, "loss lower")
                            break
        if verbose > 1:
            print("\tEpoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    return {
        'model': best_model,
        'model_name': best_model.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'train_err': train_err_array,
        'val_err': val_err_array,
        'nb_samples': nb_sample_array,
        'best_val_loss': best_val_loss,
        'best_nb_samples': best_nb_sample}  # Return statistics about the training


# Testing the model : we need the model and a dataset
def test_model(model, testloader, estimated_parameter='body_speed'):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
    if not isinstance(testloader, torch.utils.data.DataLoader):
        testloader = torch.utils.data.DataLoader(testloader, batch_size=16,
                                                 shuffle=False, num_workers=0)

    estimated_parameter_index = estimable_parameters[estimated_parameter]['index']

    mae = 0
    mse = 0
    mpe = 0
    real = []
    predicted = []

    model.update_device()
    model.to(device)

    for i, data in enumerate(testloader, 0):
        # get the inputs
        pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, \
        _, _, footstep_width, cycle_time, stride_length, corridor_distance = data

        # if possible, move them to GPU
        body_speed, footstep_width, cycle_time, stride_length, corridor_distance = \
            body_speed.to(device), footstep_width.to(device), cycle_time.to(device), stride_length.to(device), \
            corridor_distance.to(device)
        if model.use_concat_pir:
            concat_pir = concat_pir.to(device)
        else:
            pirs0, pirs1, pirs2, pirs3 = pirs0.to(device), pirs1.to(device), \
                                         pirs2.to(device), pirs3.to(device)

        targets = [body_speed, footstep_width, cycle_time, stride_length, corridor_distance]

        outputs = model(pirs0, pirs1, pirs2, pirs3, concat_pir)

        # real.append([targets[i].cpu().detach().numpy().tolist() for i in range(len(targets))])
        # predicted.append([outputs[i].cpu().detach().numpy().tolist() for i in range(len(targets))])
        real.extend(targets[estimated_parameter_index].cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, outputs

        # mae += torch.abs(outputs.float() - labels.float()).sum().data
        # mse += ((outputs.float() - labels.float()) * (outputs.float() - labels.float())).sum().data
        #
        # mpe += torch.abs((outputs.float() - labels.float()) / labels.float()).sum().data

    real, predicted = np.array(real), np.array(predicted)[:, 0]
    corr = np.corrcoef(np.array(real), np.array(predicted))[0, 1]
    mse = np.sqrt(((real - predicted) ** 2).mean())
    mae = abs(real - predicted).mean()
    mpe = abs((real - predicted) / real).mean() * 100

    return {'mae': mae, 'mse': mse, 'percent_error': mpe, 'correlation': corr}


# Create a dataloader that applies the desired transformations to the data before sending them as Tensors
class PIR_DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_file, transform=None, data_inflate=1.0, apply_translation = False):
        self.data = data_file
        self.transform = transform
        self.inflate = int(data_inflate)
        self.apply_translation = apply_translation

    def __getitem__(self, index):
        body_speed = 1000
        while body_speed > 3.0:
            # Get the file, load the image, apply transformation
            try:
                try:
                    pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction, \
                    footstep_width, cycle_time, stride_length, corridor_distance, raw = self.data[
                        index // self.inflate]
                except Exception as e:
                    pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction, raw = self.data[
                        index // self.inflate]
            except:
                pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction = self.data[index // self.inflate]
            index += 1

        if self.transform is not None:
            pir0 = self.transform(pir0)
            pir1 = self.transform(pir1)
            pir2 = self.transform(pir2)
            pir3 = self.transform(pir3)
            if self.apply_translation:
                trans_force = int(random.random() * pir0.shape[2] / 4) * (-1 if random.random() < 0.5 else 1)
                trans = transforms.Lambda(lambda x: torch.roll(x, trans_force, 0))
                pir0 = trans(pir0)
                pir1 = trans(pir1)
                pir2 = trans(pir2)
                pir3 = trans(pir3)

        concat_pir = torch.cat((pir0, pir1, pir2, pir3), 0)

        return pir0, pir1, pir2, pir3, concat_pir, body_speed, corridor, direction, footstep_width, cycle_time, stride_length, corridor_distance

    def __len__(self):
        return int(len(self.data) * self.inflate)


# Code taken from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    x_rotation = 0
    tick_size = "small"
    if len(classes) < 20:
        pass
    elif len(classes) < 50:
        x_rotation = 45
        tick_size = "x-small"
    else:
        x_rotation = 45
        tick_size = "xx-small"

    plt.xticks(tick_marks, classes, rotation=x_rotation, fontsize=tick_size)
    plt.yticks(tick_marks, classes, fontsize=tick_size)

    thresh = cm.max() / 2.
    if len(classes) < 20:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def display_test_results(results, classes):
    for classe in results:
        if classe == 'confusion':
            continue
        if classe == 'global':
            print("-> Global accuracy : %2d %% (%3d/%3d)" % (
                100 * results[classe]['acc'], results[classe]['correct_samples'], results[classe]['total_samples']))
            continue
        most_seen = classes[list(results['confusion'][classe]).index(max(results['confusion'][classe]))]
        if most_seen == classes[classe]:
            print('Accuracy of %6s : %2d %% (%3d/%3d) -> OK' % (
                classes[classe], 100 * results[classe]['acc'], results[classe]['correct_samples'],
                results[classe]['total_samples']))
        else:
            print('Accuracy of %6s : %2d %% (%3d/%3d) -> mostly seen as "%6s"' % (
                classes[classe], 100 * results[classe]['acc'], results[classe]['correct_samples'],
                results[classe]['total_samples'], most_seen))
    plot_confusion_matrix(results['confusion'], classes)


def visualize_model(model, testloader, estimated_parameter='body_speed', scalers = None):
    parameter_data = estimable_parameters[estimated_parameter]
    model.update_device()
    model.to(device)
    model.eval()
    print(model)
    if not isinstance(testloader, torch.utils.data.DataLoader):
        testloader = torch.utils.data.DataLoader(testloader, batch_size=1,
                                                 shuffle=False, num_workers=0)

    real = []
    predicted = []
    corridors = []
    directions = []
    for i, data in enumerate(testloader, 0):
        # get the inputs
        pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, \
        corridor, direction, footstep_width, cycle_time, stride_length, corridor_distance = data

        # if possible, move them to GPU
        body_speed, footstep_width, cycle_time, stride_length, corridor_distance = \
            body_speed.to(device), footstep_width.to(device), cycle_time.to(device), stride_length.to(device), \
            corridor_distance.to(device)
        if model.use_concat_pir:
            concat_pir = concat_pir.to(device)
        else:
            pirs0, pirs1, pirs2, pirs3 = pirs0.to(device), pirs1.to(device), \
                                         pirs2.to(device), pirs3.to(device)

        targets = [body_speed, footstep_width, cycle_time, stride_length, corridor_distance]

        outputs = model(pirs0, pirs1, pirs2, pirs3, concat_pir)

        # real.append([targets[i].cpu().detach().numpy().tolist() for i in range(len(targets))])
        # predicted.append([outputs[i].cpu().detach().numpy().tolist() for i in range(len(targets))])
        real.extend(targets[parameter_data['index']].cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, outputs

        corridors.extend(corridor.cpu().detach().numpy().tolist())
        directions.extend(direction.cpu().detach().numpy().tolist())


    real, predicted = np.array(real), np.array(predicted)[:, 0]
    directions = np.array(directions)[real > 0]
    corridors = np.array(corridors)[real > 0]
    predicted = predicted[real > 0]
    real = real[real > 0]

    pd.DataFrame({'real': real, 'predicted': predicted}).to_csv("CNN_results.csv")
    errors = pd.Series(real - predicted)
    errors_percent = pd.Series(100 * (real - predicted) / real)
    real, predicted = np.array(real) * parameter_data['multiplier'], np.array(predicted) * parameter_data[
        'multiplier']
    if scalers is not None: # and parameter_data['scaler_name'] in scalers:
        predicted = scalers[parameter_data['scaler_name']].inverse_transform(predicted.reshape(1, -1)).reshape(-1)
        real = scalers[parameter_data['scaler_name']].inverse_transform(real.reshape(1, -1)).reshape(-1)

    scaled_error = pd.Series(real - predicted)
    abs_errors = abs(real - predicted)
    abs_errors_percent = abs(100 * (real - predicted) / real)
    sorted_abs_errors = pd.Series([e for _, e in sorted(zip(real, abs_errors))], name='Errors')
    sorted_abs_errors_percent = pd.Series([e for _, e in sorted(zip(real, abs_errors_percent))], name='Error (%)')
    pearson_coef, p_value = pearsonr(real, predicted)

    plt.subplot(3, 2, 1)
    colors = [(1, 0, 0, 1) if c == -1 else (0, 1, 0, 1) if c == 0 else (0, 0, 1, 1) for c in corridors]
    directions = np.array(directions)
    corridors = np.array(corridors)
    colors = np.array(colors)
    plt.scatter(real, predicted, marker=".", c="black")
    # plt.scatter(real[directions < 0], predicted[directions < 0], marker="<", c=colors[directions < 0])
    # plt.scatter(real[directions > 0], predicted[directions > 0], marker=">", c=colors[directions > 0])
    plt.plot(list(plt.xlim()), list(plt.xlim()), '--', c=(0, 0, 0, .2))
    plt.ylabel(f"Measured {parameter_data['name']} ({parameter_data['unit']})")
    plt.xlabel(f"Reference {parameter_data['name']} ({parameter_data['unit']})")
    plt.title(f"Results from the CNN (correlation : {round(pearson_coef, 2)} | "
              f"{'p < 0.001' if p_value < 0.001 else 'p = ' + str(round(p_value, 4))})")
    # plt.legend([(plt.plot([], 'b<')[0], plt.plot([], 'b>')[0]),
    #             matplotlib.patches.Patch(color=(1, 0, 0)),
    #             matplotlib.patches.Patch(color=(0, 1, 0)),
    #             matplotlib.patches.Patch(color=(0, 0, 1))],
    #            ['Direction', 'East', 'Center', 'West'], handler_map={tuple: HandlerTuple(ndivide=None)})

    # plt.show()
    plt.subplot(3, 2, 2)
    plt.title('Error distribution')
    plt.hist(abs_errors, bins=30, color=(237 / 255, 190 / 255, 36 / 255, 1), density=1)
    # Calculate percentiles
    quant_25, quant_50, quant_75, quant_95 = np.percentile(abs_errors, [25, 50, 75, 95])
    quants = [[quant_25, 2, 25], [quant_50, 1.8, 50], [quant_75, 1.6, 75],
              [quant_95, 1.4, 95]]
    for i in quants:
        plt.plot([i[0], i[0]], [0, i[1]], ':', c='blue')
        plt.text(i[0], i[1] * 1.05, f"{i[2]}th", size=11, alpha=0.85, ha='center', va='bottom')

    plt.xlabel(f"Error in {parameter_data['unit']}")
    plt.ylabel('Frequency')

    # plt.show()
    plt.subplot(3, 2, 3)
    plt.axhline(np.median(abs_errors), color=(0, 0, 0, .5), linestyle='--')
    plt.text((min(real) + max(real)) / 2, np.median(abs_errors),
             f"median = {round(np.median(abs_errors), 2)}{parameter_data['unit']}", va='bottom', ha='center')
    plt.scatter(real, abs_errors, marker='.', c='black')
    # plt.scatter(real[directions < 0], errors[directions < 0], marker='<', c=colors[directions < 0])
    # plt.scatter(real[directions > 0], errors[directions > 0], marker='>', c=colors[directions > 0])
    plt.title(f"Error depending on the real {parameter_data['name']}")
    plt.xlabel(f"Real {parameter_data['name']}")
    plt.ylabel(f"Error in {parameter_data['unit']}")

    # plt.show()
    plt.subplot(3, 2, 4)
    plt.axhline(np.median(abs_errors_percent), color=(0, 0, 0, .5), linestyle='--')
    plt.text((min(real) + max(real)) / 2, np.median(abs_errors_percent), f"median = {round(np.median(abs_errors_percent), 2)}%",
             va='bottom', ha='center')
    plt.scatter(real, abs_errors_percent, marker='.', c='black')
    # plt.scatter(real[directions < 0], errors_percent[directions < 0], marker='<', c=colors[directions < 0])
    # plt.scatter(real[directions > 0], errors_percent[directions > 0], marker='>', c=colors[directions > 0])
    plt.title(f"% Error depending on the real {parameter_data['name']}")
    plt.xlabel(f"Real {parameter_data['name']}")
    plt.ylabel('% Error')

    # plt.show()
    plt.subplot(3, 2, 5)
    plt.title('Error distribution (%)')
    plt.hist(abs_errors_percent, bins=30, color=(237 / 255, 190 / 255, 36 / 255, 1), density=1)
    quant_25, quant_50, quant_75, quant_95 = np.percentile(abs_errors_percent, [25, 50, 75, 95])
    quants = [[quant_25, .10, 25], [quant_50, .09, 50], [quant_75, .08, 75],
              [quant_95, .07, 95]]
    for i in quants:
        plt.plot([i[0], i[0]], [0, i[1]], ':', c='blue')
        plt.text(i[0], i[1] * 1.05, f"{i[2]}th", size=11, alpha=0.85, ha='center', va='bottom')
    plt.xlabel("Error in %")
    plt.ylabel('Frequency')


    plt.subplot(3, 2, 6)
    plt.title('Error distribution')
    scaled_error.plot(kind='hist', density=1)
    scaled_error.plot(kind='kde')
    # plt.hist(scaled_error, bins=20, color=(237 / 255, 190 / 255, 36 / 255, 1), density=1)
    # Calculate percentiles
    quant_25, quant_50, quant_75, quant_95 = np.percentile(abs_errors, [25, 50, 75, 95])
    quants = [[quant_25, 2, 25], [quant_50, 1.8, 50], [quant_75, 1.6, 75],
              [quant_95, 1.4, 95]]
    # for i in quants:
    #     plt.plot([i[0], i[0]], [0, i[1]], ':', c='blue')
    #     plt.text(i[0], i[1] * 1.05, f"{i[2]}th", size=11, alpha=0.85, ha='center', va='bottom')

    plt.xlabel(f"Error in {parameter_data['unit']}")
    plt.ylabel('Frequency')

    if parameter_data['apply_tight_layout']:
        plt.tight_layout()

    print(pd.concat([scaled_error, errors_percent, sorted_abs_errors, sorted_abs_errors_percent,
                     pd.Series(sorted_abs_errors_percent.values[directions < 0], name='Error (%) South-North'),
                     pd.Series(sorted_abs_errors_percent.values[directions > 0], name='Error (%) North-South'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == -1], name='Error (%) East'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == 0], name='Error (%) Center'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == 1], name='Error (%) West')],
                    axis=1).describe())


# From a set of parameters, create all the combinations possible
def create_all_combinaisons(hyperparameters):
    combinaisons = [hyperparameters]
    for param in hyperparameters:  # For each parameter
        if isinstance(hyperparameters[param], Iterable):  # Check if it's iterable (if not, it's a fix value)
            i = 0
            while i < len(combinaisons):  # Iterate on each stored combination
                if isinstance(combinaisons[i][param], list):  # Check if this parameter is unfix
                    combinaisons.append(copy.deepcopy(combinaisons[i]))
                    combinaisons[i][param] = combinaisons[-1][param].pop(0)  # In this case, duplicate the parameter
                    if len(combinaisons[-1][param]) == 1:
                        combinaisons[-1][param] = combinaisons[-1][param][0]  # and add a fix value
                i += 1
    return combinaisons


# Launch the training function for every possible combination.
# Arguments:
# - hyperparameters : all the parameters that can be changed in the model
# - train_set : the training set
# - validation_set : validation set
# - testing_set : test set
# - model_class : Specify the class of the models you want to train
# - loss_function : Specify the loss function you want to use. default is Cross Entropy
# - training_for_autoencoders : will simply change the target for the loss
# - optimizer : Specify your optimizer. default is SGD
# - learning rate : parameter given to the optimizer
# - num_epochs : number of epochs for the training
# - early stopping : if True, will stop if the validation loss starts increasing
#                   If a integer value, defines the "patience" (number of minibatch to study before officially stops for overfitting)
# - save_as : specify the name of the file where we save all the statistics (including the model itself)
#                   a timestamp is added to the file name
# - start_from_beginning : remove all previously saved models if True, otherwise will ignore previously calculated combinations
# - verbose : level of verbosity

def try_all_models(hyperparameters,
                   train_set,
                   validation_set,
                   testing_set,
                   training_parameters: dict = {},
                   model_class: Type[nn.Module] = MultipleWaveletsModule,
                   loss_function: Type[nn.modules.loss.Module] = nn.CrossEntropyLoss,
                   optimizer=False,
                   training_for_autoencoder=False,
                   learning_rate=0.001,
                   num_epochs=5,
                   display_precision=1.0,
                   early_stopping: Union[bool, int] = True,
                   save_as: Union[bool, str] = False,
                   start_from_beginning=True,
                   verbose=3,
                   ignored_files='_useless_',
                   max_number_of_models=math.inf,
                   estimated_parameter='body_speed'):
    models = []
    combinations = create_all_combinaisons(hyperparameters)  # Get all possible combinations
    combinations_done = []

    parameters_combinations = create_all_combinaisons(training_parameters)

    if start_from_beginning:  # Remove previously saved models if "start_from_beginning" is True
        for file in glob.glob(os.path.splitext(save_as)[0] + "*" + os.path.splitext(save_as)[1]):
            os.remove(file)
    else:
        if not save_as:
            if verbose > 0:
                print("No filename has been given as the 'save_as' argument. We cannot retrieve pretrained models")
        else:
            # Look for all pretrained models
            for file in glob.glob(os.path.splitext(save_as)[0] + "*" + os.path.splitext(save_as)[1]):
                models = pickle.load(open(file, "rb"))
                for model in models:
                    combinations_done.append(model['parameters'])
                del models  # save a little bit of memory
                torch.cuda.empty_cache()

            if verbose > 1:
                print(len(combinations_done), "models have been retrieved and ignored")

    if len(combinations_done) >= min(max_number_of_models, len(combinations)):  # If there is nothing to train
        if verbose > 0:
            print("No new model to train")
        return []

    random.shuffle(combinations)  # Shuffle to train in a random order,
    # so that you can stop early and still have interesting data
    current_combination_number = 0
    for i, parameters in enumerate(combinations, 1):
        torch.cuda.empty_cache()
        try:
            if len(combinations_done) + current_combination_number >= max_number_of_models:
                return []
            current_combination_number += 1
            start_time = time.time()
            if verbose > 0:
                print(
                    F"Trying combination {current_combination_number} / {len(combinations) - len(combinations_done)} ...")
            if parameters in combinations_done:
                current_combination_number -= 1
                if verbose > 0:
                    print("Ignored (pretrained)")
                continue

            used_training_parameters = copy.deepcopy(random.choice(parameters_combinations))
            if 'model_class' not in used_training_parameters:
                used_training_parameters['model_class'] = model_class
            if 'model_parameters' not in used_training_parameters:
                used_training_parameters['model_parameters'] = parameters
            if 'loss_function' not in used_training_parameters:
                used_training_parameters['loss_function'] = loss_function
            if 'training_for_autoencoder' not in used_training_parameters:
                used_training_parameters['training_for_autoencoder'] = training_for_autoencoder
            if 'optimizer' not in used_training_parameters:
                used_training_parameters['optimizer'] = optimizer
            if 'learning_rate' not in used_training_parameters:
                used_training_parameters['learning_rate'] = learning_rate
            if 'num_epochs' not in used_training_parameters:
                used_training_parameters['num_epochs'] = num_epochs
            if 'early_stopping' not in used_training_parameters:
                used_training_parameters['early_stopping'] = early_stopping
            if 'display_precision' not in used_training_parameters:
                used_training_parameters['display_precision'] = display_precision
            if 'verbose' not in used_training_parameters:
                used_training_parameters['verbose'] = verbose
            if 'estimated_parameter' not in used_training_parameters:
                used_training_parameters['estimated_parameter'] = estimated_parameter

            current_model = {}

            current_model['parameters'] = parameters
            current_model['training_parameters'] = used_training_parameters
            if current_model['parameters'] != current_model['training_parameters']['model_parameters']:
                print('Wooosh!')

            # Save the training statistics (with the model itself), the testing statistics and the parameters used
            current_model['training'] = train_model(trainloader=train_set,
                                                    validationloader=validation_set,
                                                    **used_training_parameters)
            if verbose:
                print("Training ended, testing the model")
            torch.cuda.empty_cache()
            current_model['test'] = test_model(current_model['training']['model'], testing_set,
                                               estimated_parameter=estimated_parameter)
            if verbose:
                print(current_model['test'])
            current_model['parameters'] = parameters
            current_model['training_parameters'] = used_training_parameters

            if current_model['parameters'] != current_model['training_parameters']['model_parameters']:
                print('Wooosh!')

            # if verbose > 1:
            #     print("Found an accuracy of ", current_model['test']['global']['acc'], "(",
            #           current_model['test']['global']['total_samples'], " used) from the ",
            #           current_model['training']['best_nb_samples'], "samples")

            # Save file at every iteration to by able to stop anytime
            if save_as:
                file = os.path.split(os.path.splitext(save_as)[0])[-1]
                ext = os.path.splitext(save_as)[1]
                directory = os.path.join(*os.path.split(os.path.splitext(save_as)[0])[:-1])
                if not os.path.exists(directory):
                    os.makedirs(directory)
                pickle.dump([current_model], open(os.path.join(directory, file + "_" + str(time.time()) + ext), "wb+"))
            del current_model
            torch.cuda.empty_cache()
            if verbose > 1:
                print("Trained in", round(time.time() - start_time), "seconds")
        except Exception as e:
            # raise e
            print('Exception occured : ', e)
    return []


#
# def retest_all_models(save_as: str, testing_set):
#     retested_models = []
#     # Look for all pretrained models
#     for file in glob.glob(os.path.splitext(save_as)[0] + "*" + os.path.splitext(save_as)[1]):
#         models = pickle.load(open(file, "rb"))
#         for model in models:
#             model['test'] = test_model(model['training']['model'], copy.deepcopy(testing_set))
#             retested_models.append(model)
#         os.remove(file)
#
#     file = os.path.split(os.path.splitext(save_as)[0])[-1]
#     ext = os.path.splitext(save_as)[1]
#     directory = os.path.join(*os.path.split(os.path.splitext(save_as)[0])[:-1])
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     pickle.dump(retested_models, open(os.path.join(directory, file + "_" + str(time.time()) + ext), "wb+"))


def display_models(filename, test_loader, ignore_if_accuracy_lower=0.0, ignore_if_val_err_higher=math.inf,
                   max_y_val=0.20, ignored_files='_useless_', estimated_parameter='body_speed',
                   scalers = None, order_by='correlation', bigger_is_better = True,
                   display_all_models = False) -> Tuple[
    Any, pd.DataFrame]:
    models_results = []
    validation_results = []
    best_results = None
    best_index = 0

    all_plots = []
    top_plots = []

    top_models = []
    min_score_of_tops = math.inf

    for i, file in enumerate(glob.glob(os.path.splitext(filename)[0] + "*" + os.path.splitext(filename)[1])):
        if ignored_files in file:
            continue
        models = pickle.load(open(file, "rb"))
        for model in models:
            try:
                if model['training']['val_err'][-1] <= max_y_val:
                    all_plots.append({'x': np.array(model['training']['nb_samples']) / 1000,
                                      'y': model['training']['val_err'],
                                      'y_train': model['training']['train_err'],
                                      'result': model['test']['correlation'],
                                      })
                _models_results = {**model['training_parameters']['model_parameters'],
                                   'number_of_samples': f"{model['training']['best_nb_samples']:,}",
                                   'test_results': model['test']['correlation'], 'mae': model['test']['mae'],
                                   'mse': model['test']['mse'], 'percent_error': model['test']['percent_error'],
                                   'id': i,
                                   **model['training_parameters']}
                models_results.append(copy.deepcopy(_models_results))
                for attr in _models_results:
                    if hasattr(_models_results[attr], 'dec') and hasattr(_models_results[attr], 'enc'):
                        models_results[-1]['number_of_layers'] = len(_models_results[attr].enc) // 3
                        models_results[-1]['kernel_size'] = str(
                            list(_models_results[attr].enc[0].parameters())[0][0].size(1))
                        models_results[-1] = dict(models_results[-1])
                        del models_results[-1][attr]

                if (best_results is None or np.isnan(best_results['test'][order_by])) or \
                        (not np.isnan(model['test'][order_by]) and
                         model['test'][order_by] * (-1 if bigger_is_better else 1) < best_results['test'][order_by] * (-1 if bigger_is_better else 1)):
                    best_results = model
                    best_index = i

                try:
                    print(model['test'])
                    print(model['training_parameters']['estimated_parameter'])
                    if display_all_models:
                        torch.cuda.empty_cache()
                        visualize_model(model['training']['model'], testloader=test_loader,
                                        estimated_parameter=model['training_parameters']['estimated_parameter'],
                                        scalers = scalers)
                        plt.show()
                except Exception as e:
                    raise e
                    print("Whoops, CUDA error out of memory")
                    plt.close()
                    pass

                # print(i)
                # print(best_results['test'])
                # print(best_results['training']['model'])
                # print()

                if len(top_models) < 10 and not np.isnan(model['test'][order_by]):
                    top_models.append(model)
                    min_score_of_tops = min([m['test'][order_by] for m in top_models])
                elif not np.isnan(model['test'][order_by]) and model['test'][order_by] * (-1 if bigger_is_better else 1) > min_score_of_tops * (-1 if bigger_is_better else 1):
                    top_models.remove([m for m in top_models if m['test'][order_by] == min_score_of_tops][0])
                    top_models.append(model)
                    min_score_of_tops = min([m['test'][order_by] for m in top_models])

            except Exception as e:
                print(e)
                pass

    for model in top_models:
        if model['training']['val_err'][-1] <= max_y_val:
            top_plots.append({'x': np.array(model['training']['nb_samples']) / 1000,
                              'y': model['training']['val_err'],
                              'y_train': model['training']['train_err'],
                              'result': model['test']['correlation'],
                              'name': model['training']['model'].__class__.__name__,
                              })
    top_plots.sort(key=lambda x: x['result'], reverse=True)
    legend_train_val = plt.legend([plt.plot([], c=(1, 0, 0, 1))[0], plt.plot([], c=(1, 0, 0, .4))[0]],
                                  ['Validation', 'Training'], loc=3)
    for plot in top_plots:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(plot['x'], plot['y_train'], color=color, alpha=.4)
        plt.plot(plot['x'], plot['y'], color=color, label=plot['name'])
        plt.text(plot['x'][-1], plot['y'][-1],
                 str(round(plot['result'], 2)), horizontalalignment="left",
                 fontsize="x-small", path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
        plt.xlabel('Number of samples (x1000)')
        plt.ylabel('L2 Loss')
        plt.title('Learning curve for training and validation\nWith resulting correlation on the test set')
    plt.yscale('log')
    plt.legend()
    plt.gca().add_artist(legend_train_val)
    plt.tight_layout()
    plt.show()
    # plt.close()

    df = pd.DataFrame(models_results)
    df[['test_results', 'mae', 'mse', 'percent_error']] = round(df[['test_results', 'mae', 'mse', 'percent_error']], 3)
    df['model_class'] = df['model_class'].apply(lambda model: model.__name__)
    df['trained_model'] = df.apply(axis=1, func=
    lambda row: row['trained_model'].__name__ if 'trained_model' in row
                                                 and row['model_class'] == 'PretrainedModel'
                                                 and hasattr(row['trained_model'], '__name__') else None)

    df.rename(columns={'kernel_size': 'kernel', 'number_of_filters': 'filters', 'number_of_layers': 'layers',
                       'test_results': 'corr', 'number_of_conv_per_layer': 'conv/layer', 'reduce_last_conv': 'reduce',
                       'use_concat_pir': 'concat_pirs', 'percent_error': '%err'},
              inplace=True)
    full_df = df
    df = df[['mid_output_size', 'kernel', 'filters', 'layers', 'conv/layer',
             'hidden_cells', 'dropout',
             'concat_pirs', 'model_class', 'corr', 'mae', 'mse', '%err'] +
            (['trained_model'] if 'trained_model' in df.columns else []) +
            (['reduce'] if 'reduce' in df.columns else []) +
            (['layer_relu'] if 'layer_relu' in df.columns else []) +
            (['use_exponential'] if 'use_exponential' in df.columns else [])
            ]

    # no_na = df.dropna(subset=['corr'])
    # index_index = abs(no_na['corr'] - max(no_na['corr'])) < 1
    # index = no_na.index[index_index][0]
    # best_results = all_models[0]  # [index]

    # Must delete all variables that may contain models on CUDA, to avoid falling out of memory!
    # del all_models
    del models
    torch.cuda.empty_cache()
    print(best_results['test'])
    try:
        if 'estimated_parameter' in best_results['training_parameters']:
            visualize_model(best_results['training']['model'], testloader=test_loader,
                            estimated_parameter=best_results['training_parameters']['estimated_parameter'],
                            scalers = scalers)
        else:
            visualize_model(best_results['training']['model'], testloader=test_loader,
                            scalers = scalers)
    except Exception as e:
        raise e
        pass
    plt.show()

    return df, full_df


def display_morlet(pir_data):
    data_norm = waipy.normalize(pir_data)

    alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]
    # print("Lag-1 autocorrelation = {:4.2f}".format(alpha))
    result = waipy.cwt(data_norm, 1 / 3.2, 1, 1 / 100, 2, 4 / 0.25, alpha, 6, mother='Morlet', name='name')
    # plt.imshow(result['power'])
    return result['power']


def compute_morlet_transform(pir_data):
    data_norm = waipy.normalize(pir_data)
    alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]
    dt = 1 / 40
    dj = 1 / 100
    lowerPeriod = 2 * dt
    upperPeriod = len(data_norm) * dt / 3
    result = waipy.cwt(data_norm, dt, 1, dj, lowerPeriod, upperPeriod, alpha, 6, mother='Morlet', name='name')
    powers = np.array(result['power']).astype('uint8')
    tmp_img = Image.fromarray(powers)  # , mode = 'I;16')
    # if(powers.max() > 255):
    #     plt.imshow(tmp_img)
    #     tmp_img.show()
    #     plt.show()
    res = Image.fromarray(np.zeros((1024, 2048)))  # , dtype=np.uint16), mode='I;16')
    res.paste(tmp_img, (0, 1024 - tmp_img.height))
    res = res.resize((512, 512), resample=Image.NEAREST)
    return res


def normalize(df, concerned_columns: Union[list, str] = 'all', ignored_columns: list = None):
    result = df.copy()
    if concerned_columns == 'all':
        concerned_columns = result.columns.tolist()
    if ignored_columns is not None:
        for col in ignored_columns:
            if col in concerned_columns:
                concerned_columns.remove(col)

    scalers = {}
    # scalers = {feature_name: sklearn.preprocessing.StandardScaler().fit(df[feature_name])
    #            for feature_name in concerned_columns}
    for feature_name in concerned_columns:
        scaler = sklearn.preprocessing.MinMaxScaler()
        df[feature_name] = scaler.fit_transform(df[feature_name].values.reshape(-1, 1))
        scalers[feature_name] = scaler
    #     # max_value = df[feature_name].max()
    #     # min_value = df[feature_name].min()
    #     # result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df, scalers


def precompute_morlet_transforms(raw_data, pir_transforms_filename='all_pirs_transformed.data', output_filename=None):
    if isinstance(output_filename, bool) and output_filename is True:
        output_filename = pir_transforms_filename
    start_time = time.time()
    sequence_ids = pd.unique(raw_data['sequence_id'])
    data, scalers = normalize(raw_data, 'all', ignored_columns=['time', 'nb_steps_motion', 'nb_steps_distance', 'direction',
                                                   'corridor', 'sequence_id', ])
    all_pirs_transformed = {'data': [], 'scalers': scalers}
    if os.path.exists(pir_transforms_filename):
        all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))
    all_pirs_transformed['data'] = list(filter(lambda seq: seq[5] < 10 and not np.isnan(seq[6]) and seq[7] != 0,
                                       all_pirs_transformed['data']))
    ignored_sequences = set([seq[4] for seq in list(filter(lambda seq: seq[5] > 10 or np.isnan(seq[6]) or seq[7] == 0,
                                                           all_pirs_transformed['data']))])
    preexisting_sequences = set([seq[4] for seq in all_pirs_transformed['data']])

    for i, seq_id in enumerate(sequence_ids):
        print(f"{i + 1}/{len(sequence_ids)} ({seq_id}): {round(100 * i / len(sequence_ids), 1)}%")
        if seq_id in preexisting_sequences:
            print("Loaded from cache")
            continue
        if seq_id in ignored_sequences:
            print("Ignored")
            continue

        try:
            seq = data[data['sequence_id'] == seq_id]
            speed = seq['walking_speed'].median()
            corridor = seq['corridor'].median()
            direction = seq['direction'].median()
            footstep_width = seq['footstep_width'].median()
            cycle_time = seq['cycle_time'].median()
            stride_length = seq['stride_length'].median()
            corridor_position = seq['corridor_position'].median()
            raw = {
                'pir0': seq['pir0'],
                'pir1': seq['pir1'],
                'pir2': seq['pir2'],
                'pir3': seq['pir3'],
            }
            if speed < 10 and direction != 0 and not np.isnan(corridor):
                # print(speed, seq['time'].iloc[0], seq['time'].iloc[-1])
                with HiddenPrints(False):
                    pir0 = compute_morlet_transform(seq['pir0'])
                    pir1 = compute_morlet_transform(seq['pir1'])
                    pir2 = compute_morlet_transform(seq['pir2'])
                    pir3 = compute_morlet_transform(seq['pir3'])
                    all_pirs_transformed['data'].append([pir0, pir1, pir2, pir3, seq_id, speed, corridor, direction,
                                                 footstep_width, cycle_time, stride_length, corridor_position, raw])
            else:
                print(speed, direction, corridor)
        except Exception as e:
            pass
        time_estimation = (time.time() - start_time) * len(sequence_ids) / (i + 1)
        time_left = time_estimation - (time.time() - start_time)
        print(
            f"\tETA {int(time_left // 60)}min{round(time_left % 60, 2)}s -- "
            f"Full time estimation: {int(time_estimation // 60)}min{round(time_estimation % 60, 2)}s "
            f"({round(time_estimation, 2)}s)")
    print('100%')

    if output_filename is not None:
        print(f"Saving {len(all_pirs_transformed['data'])} sequences into {output_filename}")
        pickle.dump(all_pirs_transformed, open(output_filename, 'wb+'))
        print('Done.')
    return all_pirs_transformed


def training(path_to_save_cnn,
             pir_transforms_filename='all_pirs_transformed.data',
             estimated_parameter='body_speed',
             number_of_models_trained = math.inf):
    all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))
    # Required "resize" to fit the DNNs
    # input_size = 224
    # Required "resize" to fit the CNNs
    input_size = 128

    # Load the data through a DataLoader on which we can apply some transformation to create a more robust model
    # The "data_inflate" can increase the number of data by multiplying them and apply different transformations
    dataset = PIR_DataLoader(all_pirs_transformed['data'], transform=transforms.Compose([
        transforms.Resize((input_size, input_size), transforms.InterpolationMode.NEAREST),
        # transforms.RandomRotation([-1.0, 1.0]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x * (1 - torch.randn_like(x) * 0.01 + 0.005)),
        transforms.Lambda(lambda x: transforms.GaussianBlur(
            2 * int(random.random() * 20) + 1, (0.1, 10.0))(x)),
    ]),
                             data_inflate=3.0, apply_translation = True)
    # Train size = 70%, Validation = 10%, Testing = 20%
    ratio_train_set = 70 / 100
    ratio_valid_set = 10 / 100

    train_size = int(ratio_train_set * len(dataset))
    valid_size = int(ratio_valid_set * len(dataset))
    test_size = len(dataset) - (train_size + valid_size)
    training_set_PIR, validation_set_PIR, testing_set_PIR = torch.utils.data.random_split(dataset,
                                                                                          [train_size, valid_size,
                                                                                           test_size])
    minibatch_size = 1
    # We've chose high batch size to be more efficient in time
    trainloader = torch.utils.data.DataLoader(training_set_PIR, batch_size=minibatch_size,
                                              shuffle=True, num_workers=0)
    validationloader = torch.utils.data.DataLoader(validation_set_PIR, batch_size=minibatch_size,
                                                   shuffle=True, num_workers=0)
    testingloader = torch.utils.data.DataLoader(testing_set_PIR, batch_size=minibatch_size,
                                                shuffle=True, num_workers=0)
    # Number of epochs applied on our model during the training
    num_epochs = 20

    # Parameters to vary on the models
    # Be careful to avoid having too much params as the number of models is greatly increasing
    models_hyperparameters = {
        # Those params are for the CNNs (MultipleWaveletsModule and MultipleWaveletsLinearModule)
        'input_size': input_size,
        'mid_output_size': [20, 40],
        'kernel_size': [3, 5, 15],
        'number_of_filters': [4, 8, 16, 32],
        'number_of_layers': [2, 4, 6],
        'number_of_conv_per_layer': [2, 4, 5],
        'hidden_cells': [None, 10, 100, (40, 40, 40)],
        'dropout': False,
        'use_concat_pir': True,
        'reduce_last_conv': [0, 2],
        'pure_CNN': [True, False],
        'layer_relu': [None, nn.ReLU(), nn.Tanh(), nn.Sigmoid()],
        'lastActivation': [None, nn.ReLU()],
        'use_exponential': False, # [True, False, 'pow'],
        'number_of_outputs': 1,

        # These params are used for the transfer learning, it's the type of model used (AlexNet, SqueezeNet, etc...)
        'trained_model': list(all_pretrained_models.values()),
        'pretrained': True
    }

    # # Parameters to vary on the models
    # # Be careful to avoid having too much params as the number of models is greatly increasing
    # models_hyperparameters = {
    #     # Those params are for the CNNs (MultipleWaveletsModule and MultipleWaveletsLinearModule)
    #     'input_size': input_size,
    #     'mid_output_size': [20, 30, 40],
    #     'kernel_size': [3, 5],
    #     'number_of_filters': [2, 4, 8, 16, 32],
    #     'number_of_layers': [2, 3, 5, 6],
    #     'number_of_conv_per_layer': [3, 4, 5],
    #     'hidden_cells': [None, 10, 100, (10, 10, 10), (40, 40, 40)],
    #     'dropout': [True, False],
    #     'use_concat_pir': [True, False],
    #     'reduce_last_conv': [0, 1, 2, 3],
    #     # 'pure_CNN': [True, False],
    #     # These params are used for the transfer learning, it's the type of model used (AlexNet, SqueezeNet, etc...)
    #     # 'trained_model': list(all_pretrained_models.values())
    # }

    # Those parameters modify the type of training used for each network
    # The parameters used for a model are chosen randomly and thus, does not increase the number of trainings
    # In a normal use, you should choose only one optimizer, one lr, one loss function, etc...
    training_parameters = {
        'optimizer': [optim.Rprop, optim.Adagrad, optim.ASGD, optim.LBFGS, optim.Adam, optim.SGD, optim.Adadelta,
                      optim.Adamax, optim.AdamW],
        'learning_rate': [0.0001, 0.00001, 0.00005],
        'loss_function': [nn.MSELoss, nn.L1Loss],
        'model_class': [MultipleWaveletsModule, ResidualNet, PretrainedModel], # [Residualnet, MultipleWaveletsLinearModule, MultipleWaveletsModule, PretrainedModel],
        'estimated_parameter': estimated_parameter
        #
    }

    # Function that starts the training for all the models
    try_all_models(models_hyperparameters,
                   trainloader,
                   validationloader,
                   testingloader,
                   training_parameters=training_parameters,
                   # model_class=MultipleWaveletsLinearModule,
                   # loss_function=nn.MSELoss,
                   verbose=10,
                   # learning_rate=0.00005,
                   training_for_autoencoder=False,
                   save_as=path_to_save_cnn,
                   start_from_beginning=False,
                   display_precision=0.5,
                   early_stopping=10,
                   num_epochs=num_epochs,

                   max_number_of_models=number_of_models_trained)


def display(path_to_save_cnn, pir_transforms_filename='all_pirs_transformed.data', estimated_parameter='body_speed'):
    # Required "resize" to fit the DNNs
    # input_size = 224
    # Required "resize" to fit the CNNs
    input_size = 128

    all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))

    # Load the data through a DataLoader on which we can apply some transformation to create a more robust model
    # The "data_inflate" can increase the number of data by multiplying them and apply different transformations
    dataset = PIR_DataLoader(all_pirs_transformed['data'], transform=transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.RandomRotation([-1.0, 1.0]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        # transforms.Lambda(lambda x: x * (1 - torch.randn_like(x) * 0.1)),
        # transforms.Lambda(lambda x: transforms.GaussianBlur(
        #     2 * int(random.random() * 20) + 1, (0.1, 10.0))(x)),
    ]),
                             data_inflate=1.0, apply_translation=False)
    del all_pirs_transformed['data']

    # Train size = 70%, Validation = 10%, Testing = 20%
    ratio_train_set = 70 / 100
    ratio_valid_set = 10 / 100

    train_size = int(ratio_train_set * len(dataset))
    valid_size = int(ratio_valid_set * len(dataset))
    test_size = len(dataset) - (train_size + valid_size)
    training_set_PIR, validation_set_PIR, testing_set_PIR = torch.utils.data.random_split(dataset,
                                                                                          [train_size, valid_size,
                                                                                           test_size])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('expand_frame_repr', False)

    # Display the results with a learning curve (just to get an idea of the efficiency)
    # And display a recap table with your models, the training parameters, hyperparameters and results
    # Here, the "test_results" column is the coef of correlation between the estimation and ground truth
    #    "test_results" of 1 = perfect match, 0 = no significant match, -1 = what?, NaN = model is rubbish
    #    models that return a coef "NaN" = return same value, regardless of input
    # plt.title("Training of %d models for the speed estimator" % len(results_data))
    # ordering = ('correlation', True)
    ordering = ('mae', False)
    # ordering = ('mse', False)
    # ordering = ('percent_error', False)
    short_results, full_results = display_models(path_to_save_cnn, test_loader=testing_set_PIR, max_y_val=math.inf,
                                                 estimated_parameter=estimated_parameter,
                                                 scalers=all_pirs_transformed['scalers'],
                                                 order_by=ordering[0], bigger_is_better=ordering[1],
                                                 display_all_models = False)
    # full_results.sort_values(by="corr", ascending=False, inplace=True)
    # print(full_results)
    print(short_results.sort_values(by=ordering[0], ascending=ordering[1]).head(500))


def merge_models_in_one_file(path_to_cnn, ignored_files='_useless_'):
    filename = os.path.split(os.path.splitext(path_to_cnn)[0])[-1]
    ext = os.path.splitext(path_to_cnn)[1]
    directory = os.path.join(*os.path.split(os.path.splitext(path_to_cnn)[0])[:-1])

    models = []
    ignored_models = []
    files_to_delete = []
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + "*" + os.path.splitext(path_to_cnn)[1])
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        if ignored_files in file:
            continue
        try:
            these_models = pickle.load(open(file, "rb"))
            ignored_models = list(filter(lambda model: np.isnan(model['test']['correlation']), these_models))
            pickle.dump(ignored_models, open(os.path.join(directory, filename + ignored_files + str(i) +
                                                          '_' + str(time.time()) + ext), "wb+"))
            these_models = list(filter(lambda model: not np.isnan(model['test']['correlation']), these_models))
            models.extend(these_models)
            files_to_delete.append(file)
        except Exception as e:
            print(e)

    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Saving {len(models)} models...")
    pickle.dump(models, open(os.path.join(directory, filename + "_" + str(time.time()) + ext), "wb+"))
    for file in files_to_delete:
        os.remove(file)

    print('Merging useless models...')
    models = []
    files_to_delete = []
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + ignored_files + "*" + os.path.splitext(path_to_cnn)[1])
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        try:
            these_models = pickle.load(open(file, "rb"))
            models.extend(these_models)
            files_to_delete.append(file)
        except Exception as e:
            print(e)
    pickle.dump(models, open(os.path.join(directory, filename + ignored_files + str(time.time()) + ext), "wb+"))

    print('Cleaning')
    for file in files_to_delete:
        os.remove(file)

    print('Done.')


def remove_useless_models(path_to_cnn, ignored_files='_useless_'):
    filename = os.path.split(os.path.splitext(path_to_cnn)[0])[-1]
    ext = os.path.splitext(path_to_cnn)[1]
    directory = os.path.join(*os.path.split(os.path.splitext(path_to_cnn)[0])[:-1])

    models = []
    models_ignored = 0
    models_kept = 0
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + "*" + os.path.splitext(path_to_cnn)[1])
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        if ignored_files in file:
            continue
        try:
            these_models = pickle.load(open(file, "rb"))
            ignored_models = list(filter(lambda model: np.isnan(model['test']['correlation']), these_models))
            if len(ignored_models) > 0:
                pickle.dump(ignored_models, open(os.path.join(directory, filename + ignored_files + str(i) +
                                                              '_' + str(time.time()) + ext), "wb+"))
                models_ignored += 1

            these_models = list(filter(lambda model: not np.isnan(model['test']['correlation']), these_models))
            these_models = list(
                filter(lambda model: model['training']['model'].__class__.__name__ != 'PretrainedModel', these_models))
            if len(these_models) > 0:
                pickle.dump(these_models, open(file, "wb+"))
                models_kept += 1
            else:
                os.remove(file)
        except Exception as e:
            print(e)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if models_ignored == 0:
        print('No model to ignore. Done.')
        return

    print(f"Merging {models_ignored} useless models...")
    models = []
    files_to_delete = []
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + ignored_files + "*" + os.path.splitext(path_to_cnn)[1])
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        try:
            these_models = pickle.load(open(file, "rb"))
            models.extend(these_models)
            files_to_delete.append(file)
        except Exception as e:
            print(e)
    pickle.dump(models, open(os.path.join(directory, filename + ignored_files + str(time.time()) + ext), "wb+"))

    print('Cleaning')
    for file in files_to_delete:
        os.remove(file)

    print('Done.')


if __name__ == '__main__':

    # Example of usage: python direct_camera_to_db.py --local
    #  or
    #                   python direct_camera_to_db.py --remote
    import argparse

    parser = argparse.ArgumentParser(description='Train neural networks to estimate walking speed from the Wavelet '
                                                 'transform of the PIR sensors',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--train', dest='train', help='Train the neural networks', action='store_true')
    parser.add_argument('--display', dest='display', help='Display the results of the trained neural networks',
                        action='store_true')
    parser.add_argument('--update-data', dest='update_data', help='Update the dataset (coming from `gait_values.csv`)',
                        action='store_true')
    parser.add_argument('--merge-files', dest='merge_files', help='Merge all the files into a single one',
                        action='store_true')
    parser.add_argument('--clean-files', dest='clean_files', help='Clean all useless models into a single file',
                        action='store_true')
    # 'body_speed' 'footstep_width' 'cycle_time' 'stride_length' 'corridor_distance'
    parser.add_argument('--parameter', dest='estimated_parameter', type=str,
                        help='Choose the parameter on which you want to train/display your models. Choice:\n\t- ' +
                             '\n\t- '.join(estimable_parameters.keys()))

    args = parser.parse_args()
    do_training = args.train
    do_display = args.display
    update_data = args.update_data
    merge_files = args.merge_files
    clean_files = args.clean_files
    estimated_parameter = args.estimated_parameter

    if (do_training or do_display or clean_files or merge_files) and estimated_parameter is None:
        parser.print_help()
        print("Please select a parameter to train/display your models")
        exit()

    torch.manual_seed(42)

    sensors_height = 'ankles'

    # Path to save the binary files containing the data about our trained models
    root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{estimated_parameter}/"
    # root_to_cnn = "trained_models/cuda_batch_normed/"
    # root_to_cnn = "trained_models/transfer_learning/"
    path_to_save_cnn = root_to_cnn + "convolution_network.model"
    # pir_transforms_filename = 'all_pirs_transformed_16_bits.data'
    # pir_transforms_filename = 'all_pirs_transformed_ankles.data'
    pir_transforms_filename = f"all_pirs_transformed_{sensors_height}.data"

    number_of_models_per_parameter = 5

    if update_data:
        # All data comes from the CSV file created through fill_csv() in create_pir_csv.py
        data = pd.read_csv(f"gait_values_{sensors_height}.csv")
        # From the PIR values, apply a wavelet transform using Morlet waves and save them in a binary file
        precompute_morlet_transforms(data,
                                     pir_transforms_filename=pir_transforms_filename,
                                     output_filename=pir_transforms_filename)

    if do_training:
        if estimated_parameter == 'all':
            while True:
                for param in estimable_parameters:
                    root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{param}/"
                    path_to_save_cnn = root_to_cnn + "convolution_network.model"
                    training(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                             estimated_parameter=param, number_of_models_trained=number_of_models_per_parameter)
                number_of_models_per_parameter += 5
        else:
            training(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                     estimated_parameter=estimated_parameter)

    if do_display:
        if estimated_parameter == 'all':
            for param in estimable_parameters:
                root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{param}/"
                path_to_save_cnn = root_to_cnn + "convolution_network.model"
                display(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                        estimated_parameter=param)
        else:
            display(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                    estimated_parameter=estimated_parameter)

    if merge_files:
        merge_models_in_one_file(path_to_save_cnn)

    if clean_files:
        remove_useless_models(path_to_save_cnn)
