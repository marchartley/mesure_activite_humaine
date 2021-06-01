import copy
import glob
import inspect
import itertools
import pickle
import random
import time
from typing import Union, Any, cast, Type
import os
import sys

import torchvision
import waipy
import pywt
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

cpu = torch.device('cpu')
cuda = torch.device('cuda')

device = cuda if torch.cuda.is_available() else cpu
use_cuda = True


class PretrainedModel(nn.Module):
    def __init__(self, trained_model):
        super(PretrainedModel, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
        # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
        try:
            self.module = trained_model(pretrained=False, init_weights=True)
        except:
            self.module = trained_model(pretrained=False)
        for param in self.module.parameters():
            param.requires_grad = False

        global use_cuda
        use_cuda = True
        self.num_ftrs = 0
        if trained_model is all_pretrained_models['resnet18'] or trained_model is all_pretrained_models['shufflenet']:
            self.num_ftrs = self.module.fc.in_features
            self.module.fc = nn.Identity()  # nn.Linear(self.num_ftrs, 1)
        elif trained_model is all_pretrained_models['alexnet'] \
                or trained_model is all_pretrained_models['mobilenet_v2'] \
                or trained_model is all_pretrained_models['mobilenet_v3_small'] \
                or trained_model is all_pretrained_models['mobilenet_v3_large']\
                or trained_model is all_pretrained_models['mnasnet']:
            self.num_ftrs = self.module.classifier[-1].in_features
            self.module.classifier[-1] = nn.Identity()  # nn.Linear(self.module.classifier[-2].out_features, 1)
        elif trained_model is all_pretrained_models['squeezenet']:
            self.num_ftrs = 13*13
            # self.module.classifier = nn.Identity()  # nn.Linear(self.module.classifier[-2].out_features, 1)
            self.module.classifier = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        elif trained_model is all_pretrained_models['vgg16']:
            use_cuda = False
            self.num_ftrs = self.module.classifier[0].in_features
            self.module.classifier = nn.Identity()
        elif trained_model is all_pretrained_models['densenet']:
            use_cuda = False
            self.num_ftrs = self.module.classifier.in_features
            self.module.classifier = nn.Identity()
        # elif trained_model is all_pretrained_models['googlenet']:
        #     pass
        elif trained_model is all_pretrained_models['resnext50_32x4d'] or \
                trained_model is all_pretrained_models['wide_resnet50_2']:
                # or trained_model is all_pretrained_models['inception']
            use_cuda = False
            self.num_ftrs = self.module.fc.in_features
            self.module.fc = nn.Identity()

        global device
        self.use_cuda = use_cuda
        if use_cuda:
            device = cuda
        else:
            device = cpu

        self.module.to(device)
        self.first_conv = nn.Conv2d(4, 3, kernel_size=(3, 3), padding=1).to(device)


        self.ending = nn.Linear(self.num_ftrs, 1)

    def update_device(self):
        global device
        if self.use_cuda:
            device = cuda
        else:
            device = cpu

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        final_tensor = self.first_conv(concat_pir)
        final_tensor = self.module(final_tensor)
        final_tensor = self.ending(final_tensor)

        return final_tensor


class MultipleWaveletsModule(nn.Module):
    def __init__(self, input_size=512, mid_output_size=10, kernel_size=3, number_of_filters=2,
                 number_of_layers=8, hidden_cells=[], dropout=False, use_concat_pir=True, **useless):
        super(MultipleWaveletsModule, self).__init__()
        self.use_concat_pir = use_concat_pir
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
        # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU

        # Create one module per PIR (used when not using the concat_pir version)
        self.multiple_wavelet_modules = [WaveletModule(input_size=input_size, output_size=mid_output_size,
                                                       kernel_size=kernel_size,
                                                       number_of_filters=number_of_filters,
                                                       number_of_layers=number_of_layers,
                                                       hidden_cells=hidden_cells,
                                                       dropout=dropout, lastActivation=None).to(device) for i in
                                         range(4)]
        # Create one unique module that take the 4 PIR data at once
        self.unique_wavelet_module = WaveletModule(input_size=input_size * 4, output_size=mid_output_size,
                                                   kernel_size=kernel_size,
                                                   number_of_filters=number_of_filters,
                                                   number_of_layers=number_of_layers,
                                                   hidden_cells=hidden_cells,
                                                   dropout=dropout, lastActivation=None).to(device)

        # In any case, the activation is a Dense of shape (4*output x 1)
        self.activation_function = nn.Linear(4 * mid_output_size, 1)

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            final_tensor = self.unique_wavelet_module(concat_pir)
        else:
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.multiple_wavelet_modules[i](pirs[i])
            final_tensor = torch.cat(tuple([tens for tens in pirs]), 1)

        final_tensor = self.activation_function(final_tensor)

        return final_tensor


class WaveletModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1,
                 hidden_cells=None, dropout: Union[float, bool] = False, lastActivation=None, **useless):
        super(WaveletModule, self).__init__()
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
        elems = [nn.Conv2d(1, number_of_filters, kernel_size, padding=padding)]

        # For every new \"layer\" (or block), we add a MaxPooling and a Convolution
        # The number of filters is not changed
        for i in range(1, number_of_layers):
            elems.append(nn.MaxPool2d(2))
            input_size = input_size // 2
            elems.append(nn.Conv2d(number_of_filters, number_of_filters, kernel_size, padding=padding))
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
            linear.append(lastActivation())
        self.linear_layers = nn.Sequential(*linear)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x


class MultipleWaveletsLinearModule(nn.Module):
    def __init__(self, input_size=512, mid_output_size=10, kernel_size=3, number_of_filters=2,
                 number_of_layers=8, hidden_cells=None, dropout=False, use_concat_pir=True, **useless):
        super(MultipleWaveletsLinearModule, self).__init__()
        self.use_concat_pir = use_concat_pir
        if hidden_cells is None:
            hidden_cells = []
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
        # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU

        # Create one module per PIR (used when not using the concat_pir version)
        self.multiple_wavelet_modules = [WaveletLinearModule(input_size=input_size, output_size=mid_output_size,
                                                             kernel_size=kernel_size,
                                                             number_of_filters=number_of_filters,
                                                             number_of_layers=number_of_layers,
                                                             hidden_cells=hidden_cells,
                                                             dropout=dropout, lastActivation=None).to(device) for i in
                                         range(4)]
        # Create one unique module that take the 4 PIR data at once
        self.unique_wavelet_module = WaveletLinearModule(input_size=input_size * 2, output_size=mid_output_size * 4,
                                                         kernel_size=kernel_size,
                                                         number_of_filters=number_of_filters,
                                                         number_of_layers=number_of_layers,
                                                         hidden_cells=hidden_cells,
                                                         dropout=dropout, lastActivation=None).to(device)

        # In any case, the activation is a Dense of shape (4*output x 1)
        self.activation_function = nn.Linear(4 * mid_output_size, 1)

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            final_tensor = self.unique_wavelet_module(concat_pir)
        else:
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.multiple_wavelet_modules[i](pirs[i])
            final_tensor = torch.cat(tuple([tens for tens in pirs]), 1)

        final_tensor = self.activation_function(final_tensor)

        return final_tensor


class WaveletLinearModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1,
                 hidden_cells=None, dropout: Union[float, bool] = False, lastActivation=None, **useless):
        super(WaveletLinearModule, self).__init__()
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
                model_class: Type[nn.Module] = MultipleWaveletsModule,
                model_parameters=None,
                loss_function: Type[nn.Module] = nn.CrossEntropyLoss,
                training_for_autoencoder=False,
                optimizer: Any = None,
                learning_rate=0.010,
                num_epochs=5,
                early_stopping=False,
                display_precision: float = 100,
                verbose: Union[bool, int] = True):
    if model_parameters is None:
        raise Exception('No model parameters have been provided. Possible keys are: ' + '\n - '.join(
            model_class.__init__.__code__.co_varnames[1:]))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
    minibatch_size = 16
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
    net.to(device)  # move it to GPU if available

    # Defining loss function and optimizer
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
            pirs0, pirs1, pirs2, pirs3, concat_pir, labels = data
            # if possible, move them to GPU
            pirs0, pirs1, pirs2, pirs3, concat_pir, labels = pirs0.to(device), pirs1.to(device), \
                                                             pirs2.to(device), pirs3.to(device), \
                                                             concat_pir.to(device), labels.to(device)

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()

                _pirs0, _pirs1, _pirs2, _pirs3, _concat_pir, _labels = pirs0.clone(), pirs1.clone(), pirs2.clone(), pirs3.clone(), concat_pir.clone(), labels.clone()
                # forward + backward + optimize
                outputs = net(_pirs0, _pirs1, _pirs2, _pirs3, _concat_pir)
                if training_for_autoencoder:  # If we train an autoencoder, set the input also as output
                    raise Exception("No autoencoder allowed in this program")
                    # loss = criterion(outputs, inputs)
                else:
                    loss = criterion(outputs.reshape(-1).float(),
                                     _labels.reshape(-1).float())  # Otherwise, use the label
                loss.backward()
                return loss

            # if isinstance(optimizer, optim.LBFGS) or isinstance(optimizer, optim.):
            #     optimizer.step(closure)
            #     loss = 1.0  # No idea what to do here...
            # else:
            loss = closure()
            optimizer.step(closure)  # Need to be improved

            # count how many samples have been used during the training
            nb_used_sample += len(pirs0)
            running_loss += float(loss)  # .item()

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
                        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = data
                        # if possible, move them to GPU
                        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = pirs0.to(device), pirs1.to(device), \
                                                                         pirs2.to(device), pirs3.to(device), \
                                                                         concat_pir.to(device), labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = net(pirs0, pirs1, pirs2, pirs3, concat_pir)
                        if training_for_autoencoder:  # If we train an autoencoder, set the input also as output
                            raise Exception("No autoencoder allowed")
                            # loss = criterion(outputs, images)
                        else:
                            loss = criterion(outputs.float(), labels.float())  # Otherwise, use the label
                        totalValLoss += float(loss)  # .item()
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
def test_model(model, testloader):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
    if not isinstance(testloader, torch.utils.data.DataLoader):
        testloader = torch.utils.data.DataLoader(testloader, batch_size=16,
                                                 shuffle=False, num_workers=0)

    mae = 0
    mse = 0
    mpe = 0
    real = []
    predicted = []

    for i, data in enumerate(testloader, 0):
        # get the inputs
        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = data
        # if possible, move them to GPU
        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = pirs0.to(device), pirs1.to(device), \
                                                         pirs2.to(device), pirs3.to(device), \
                                                         concat_pir.to(device), labels.to(device)

        outputs = model(pirs0, pirs1, pirs2, pirs3, concat_pir)

        real.extend(labels.cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

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
    #
    # results = {'errors': [], 'mae': 0, 'mse': 0}
    # with torch.no_grad():
    #     for data in testloader:
    #         # get the inputs
    #         pirs0, pirs1, pirs2, pirs3, labels = data
    #         # if possible, move them to GPU
    #         pirs0, pirs1, pirs2, pirs3, labels = pirs0.to(device), pirs1.to(device), \
    #                                              pirs2.to(device), pirs3.to(device), labels.to(device)
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = model(pirs0, pirs1, pirs2, pirs3)
    #         _, predicted = torch.max(outputs, 1)  # Get the most probable prediction
    #         c = (predicted == labels).squeeze()  # Check with ground truth
    #         try:
    #             len(c)  # If "c" is an array, it's good
    #         except:
    #             c = c.unsqueeze(0)  # otherwise we need to add a dimension
    #
    #         for i in range(len(labels)):  # Update the confusion matrix
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #             confusion[label, predicted[i]] += 1
    #
    # # compute results
    # correct = 0
    # total = 0
    # for i in range(nb_classes):
    #     if class_total[i] > 0:
    #         correct += class_correct[i]
    #         total += class_total[i]
    #         results[i] = {
    #             'acc': class_correct[i] / class_total[i],
    #             'correct_samples': class_correct[i],
    #             'total_samples': class_total[i]}  # Store the accuracy and the sample size for each class
    #     else:
    #         results[i] = {'acc': 0, 'correct_samples': 0, 'total_samples': 0}
    #
    # # Store the global accuracy and the confusion matrix and return everything
    # results['global'] = {'acc': correct / total, 'correct_samples': correct, 'total_samples': total}
    # results['confusion'] = confusion
    # return results


# For this dataset, you can provide the root directory for the "symbols" and the root of the "junk" and
# it will label the symbols as 0 and the junk at 1
# Note that we do not randomize our "x" and targets, so be sure to apply a random_split or something like this :)
class PIR_DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_file, transform=None, data_inflate=1.0):
        self.data = data_file
        self.transform = transform
        self.inflate = data_inflate

    def __getitem__(self, index):
        # Get the file, load the image, apply transformation
        pir0, pir1, pir2, pir3, id_seq, body_speed = self.data[index % len(self.data)]
        if self.transform is not None:
            pir0 = self.transform(pir0)
            pir1 = self.transform(pir1)
            pir2 = self.transform(pir2)
            pir3 = self.transform(pir3)

        concat_pir = torch.cat((pir0, pir1, pir2, pir3), 0)

        return pir0, pir1, pir2, pir3, concat_pir, body_speed

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


def visualize_model(model, test_loader):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    # # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU
    if not isinstance(test_loader, torch.utils.data.DataLoader):
        test_loader = torch.utils.data.DataLoader(test_loader, batch_size=64,
                                                  shuffle=False, num_workers=0)

    real = []
    predicted = []
    for i, data in enumerate(test_loader, 0):
        # get the inputs
        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = data
        # if possible, move them to GPU
        pirs0, pirs1, pirs2, pirs3, concat_pir, labels = pirs0.to(device), pirs1.to(device), \
                                                         pirs2.to(device), pirs3.to(device), \
                                                         concat_pir.to(device), labels.to(device)

        outputs = model(pirs0, pirs1, pirs2, pirs3, concat_pir)
        real.extend(labels.cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

    real, predicted = np.array(real), np.array(predicted)[:, 0]
    plt.scatter(real, predicted)


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
                   max_number_of_models=math.inf):
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
                models = []  # save a little bit of memory

            if verbose > 1:
                print(len(combinations_done), "models have been retrieved and ignored")

    if len(combinations_done) >= min(max_number_of_models, len(combinations)):  # If there is nothing to train
        if verbose > 0:
            print("No new model to train")
        return models

    random.shuffle(combinations)  # Shuffle to train in a random order,
    # so that you can stop early and still have interesting data
    current_combination_number = 0
    for i, parameters in enumerate(combinations, 1):
        try:
            if len(combinations_done) + current_combination_number >= max_number_of_models:
                return models
            current_combination_number += 1
            models = []
            start_time = time.time()
            if verbose > 0:
                print(
                    F"Trying combination {current_combination_number} / {len(combinations) - len(combinations_done)} ...")
            if verbose > 1:
                for param in parameters:
                    print("    %15s" % param, " : ", parameters[param])
            if parameters in combinations_done:
                current_combination_number -= 1
                if verbose > 0:
                    print("Ignored (pretrained)")
                continue

            used_training_parameters = random.choice(parameters_combinations)
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

            # Save the training statistics (with the model itself), the testing statistics and the parameters used
            current_model = {'training': train_model(trainloader=copy.deepcopy(train_set),
                                                     validationloader=copy.deepcopy(validation_set),
                                                     **used_training_parameters)}
            if verbose:
                print("Training ended, testing the model")
            current_model['test'] = test_model(current_model['training']['model'], copy.deepcopy(testing_set))
            if verbose:
                print(current_model['test'])
            current_model['parameters'] = parameters
            current_model['training_parameters'] = used_training_parameters
            models.append(current_model)

            # if verbose > 1:
            #     print("Found an accuracy of ", current_model['test']['global']['acc'], "(",
            #           current_model['test']['global']['total_samples'], " used) from the ",
            #           current_model['training']['best_nb_samples'], "samples")

            # Save file at every iteration to by able to stop anytime
            if save_as:
                if len(models) > 0:
                    file = os.path.split(os.path.splitext(save_as)[0])[-1]
                    ext = os.path.splitext(save_as)[1]
                    directory = os.path.join(*os.path.split(os.path.splitext(save_as)[0])[:-1])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    pickle.dump(models, open(os.path.join(directory, file + "_" + str(time.time()) + ext), "wb+"))
                else:
                    print("No model to save")
            if verbose > 1:
                print("Trained in", round(time.time() - start_time), "seconds")
        except Exception as e:
            print('Exception occured : ', e)
    return models

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
                   max_y_val=0.20):
    models_results = []
    validation_results = []
    best_results = None

    plt.subplot(2, 1, 1)
    all_models = []
    for file in glob.glob(os.path.splitext(filename)[0] + "*" + os.path.splitext(filename)[1]):
        models = pickle.load(open(file, "rb"))
        all_models.extend(models)
        for model in models:
            plt.plot(np.array(model['training']['nb_samples']) / 1000, model['training']['val_err'])
            if model['training']['val_err'][-1] <= max_y_val:
                plt.text(model['training']['nb_samples'][-1] / 1000, model['training']['val_err'][-1],
                         str(round(model['test']['correlation'], 2)) + "%", horizontalalignment="left",
                         # str(round(model['test']['global']['acc'] * 100, 2)) + "%", horizontalalignment="left",
                         fontsize="x-small", path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
            _models_results = {**model['parameters'], 'number_of_samples': f"{model['training']['best_nb_samples']:,}",
                               'test_results': model['test']['correlation'], 'mae': model['test']['mae'],
                               'mse': model['test']['mse'], 'percent_error': model['test']['percent_error'],
                               **model['training_parameters']}
            # 'test_results': model['test']['global']['acc']}
            models_results.append(copy.deepcopy(_models_results))
            for attr in _models_results:
                if hasattr(_models_results[attr], 'dec') and hasattr(_models_results[attr], 'enc'):
                    models_results[-1]['number_of_layers'] = len(_models_results[attr].enc) // 3
                    models_results[-1]['kernel_size'] = str(
                        list(_models_results[attr].enc[0].parameters())[0][0].size(1))
                    models_results[-1] = dict(models_results[-1])
                    del models_results[-1][attr]

            # if best_results is None or best_results['test']['mse'] > model['test']['mse']:
            #     best_results = model

    df = pd.DataFrame(models_results)
    plt.ylim((0, min(max_y_val, plt.ylim()[1])))
    plt.xlabel("Number of samples (x1000)")
    plt.ylabel("Validation loss")

    # global device
    # device = cpu
    # best_results = all_models[1]
    # visualize_model(best_results['training']['model'], test_loader=test_loader)
    for i in range(10):
        try:
            plt.subplot(2, 10, i + 1)
            best_results = all_models[i]
            visualize_model(best_results['training']['model'], test_loader=test_loader)
        except Exception as e:
            print(e)
    return df

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
    result = waipy.cwt(data_norm, 1 / 3.2, 1, 1 / 100, 2, 4 / 0.25, alpha, 6, mother='Morlet', name='name')
    powers = np.array(result['power']).astype('uint8')
    res = Image.fromarray(powers).resize((512, 512))
    return res

def precompute_morlet_transforms(data, output_filename = None):
    start_time = time.time()
    sequence_ids = pd.unique(data['sequence_id'])
    all_pirs_transformed = []
    if os.path.exists('all_pirs_transformed.data'):
        all_pirs_transformed = pickle.load(open('all_pirs_transformed.data', 'rb'))
    all_pirs_transformed = list(filter(lambda seq: seq[5] < 10, all_pirs_transformed))
    preexisting_sequences = set([seq[4] for seq in all_pirs_transformed])

    for i, seq_id in enumerate(sequence_ids):
        print(f"{i + 1}/{len(sequence_ids)} ({seq_id}): {round(100 * i / len(sequence_ids), 1)}%")
        if seq_id in preexisting_sequences:
            print("Loaded from cache")
            continue

        with HiddenPrints():
            seq = data[data['sequence_id'] == seq_id]
            speed = seq['walking_speed'].median()
            pir0 = compute_morlet_transform(seq['pir0'])
            pir1 = compute_morlet_transform(seq['pir1'])
            pir2 = compute_morlet_transform(seq['pir2'])
            pir3 = compute_morlet_transform(seq['pir3'])
            if speed < 10:
                all_pirs_transformed.append([pir0, pir1, pir2, pir3, seq_id, speed])
        time_estimation = (time.time() - start_time) * len(sequence_ids) / (i + 1)
        print(
            f"\tTime estimation: {int(time_estimation // 60)}min{round(time_estimation % 60, 2)}s ({time_estimation}s)")

    if output_filename is not None:
        pickle.dump(all_pirs_transformed, open(output_filename, 'wb+'))
    return all_pirs_transformed

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # If possible, move to CUDA
    print(f"Using {device.type}")
    # device = torch.device("cpu")  # Until the CUDA problem remains, force the use of CPU

    # All data comes from the CSV file created through fill_csv() in create_pir_csv.py
    data = pd.read_csv("gait_values.csv")
    # From the PIR values, apply a wavelet transform using Morlet waves and save them in a binary file
    all_pirs_transformed = precompute_morlet_transforms(data, output_filename = 'all_pirs_transformed.data')

    # Required "resize" to fit the DNNs
    input_size = 224

    # Load the data through a DataLoader on which we can apply some transformation to create a more robust model
    # The "data_inflate" can increase the number of data by multiplying them and apply different transformations
    dataset = PIR_DataLoader(all_pirs_transformed, transform=transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation([-1.0, 1.0]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * (1 - torch.randn_like(x) * 0.1)),
        transforms.Lambda(lambda x: transforms.GaussianBlur(
            2 * int(random.random() * 20) + 1, (0.1, 10.0))(x)),
    ]),
                             data_inflate=10.0)
    # Train size = 70%, Validation = 10%, Testing = 20%
    ratio_train_set = 70 / 100
    ratio_valid_set = 10 / 100

    train_size = int(ratio_train_set * len(dataset))
    valid_size = int(ratio_valid_set * len(dataset))
    test_size = len(dataset) - (train_size + valid_size)
    training_set_PIR, validation_set_PIR, testing_set_PIR = torch.utils.data.random_split(dataset,
                                                                                          [train_size, valid_size,
                                                                                           test_size])
    # Number of epochs applied on our model during the training
    num_epochs = 50

    # Path to save the binary files containing the data about our trained models
    # root_to_cnn = "trained_models/" + device.type + "/"
    root_to_cnn = "trained_models/transfer_learning/"
    path_to_save_cnn = root_to_cnn + "convolution_network.model"

    # Parameters to vary on the models
    # Be careful to avoid having too much params as the number of models is greatly increasing
    models_hyperparameters = {
        # Those params are for the CNNs (MultipleWaveletsModule and MultipleWaveletsLinearModule)
        # 'input_size': input_size,
        # 'mid_output_size': [10, 20],
        # 'kernel_size': [3, 5],
        # 'number_of_filters': [2, 4, 8],
        # 'number_of_layers': [2, 3, 5, 6],
        # 'hidden_cells': [None, 100, (128, 256), (256, 128)],
        # 'dropout': [True, False],
        # 'use_concat_pir': [True, False],
        # These params are used for the transfer learning, it's the type of model used (AlexNet, SqueezeNet, etc...)
        'trained_model': list(all_pretrained_models.values())
    }

    # Those parameters modify the type of training used for each network
    # The parameters used for a model are chosen randomly and thus, does not increase the number of trainings
    # In a normal use, you should choose only one optimizer, one lr, one loss function, etc...
    training_parameters = {
        'optimizer': [optim.Rprop, optim.Adagrad, optim.ASGD, optim.LBFGS, optim.Adam, optim.SGD, optim.Adadelta,
                      optim.Adamax, optim.AdamW],
        'learning_rate': [0.0001, 0.00001, 0.00005],
        'loss_function': [nn.MSELoss, nn.L1Loss],
        'model_class': PretrainedModel,  # [MultipleWaveletsLinearModule, MultipleWaveletsModule]
    }

    # Function that starts the training for all the models
    # try_all_models(models_hyperparameters,
    #                training_set_PIR,
    #                validation_set_PIR,
    #                testing_set_PIR,
    #                training_parameters=training_parameters,
    #                model_class=MultipleWaveletsLinearModule,
    #                loss_function=nn.MSELoss,
    #                verbose=10,
    #                learning_rate=0.00005,
    #                training_for_autoencoder=False,
    #                save_as=path_to_save_cnn,
    #                start_from_beginning=False,
    #                display_precision=0.5,
    #                early_stopping=10,
    #                num_epochs=num_epochs)

    pd.set_option('display.max_columns', None)
    pd.set_option('expand_frame_repr', False)

    # Display the results with a learning curve (just to get an idea of the efficiency)
    # And display a recap table with your models, the training parameters, hyperparameters and results
    # Here, the "test_results" column is the coef of correlation between the estimation and ground truth
    #    "test_results" of 1 = perfect match, 0 = no significant match, -1 = what?, NaN = model is rubbish
    #    models that return a coef "NaN" = return same value, regardless of input
    # plt.title("Training of %d models for the speed estimator" % len(results_data))
    results_data = display_models(path_to_save_cnn, test_loader=testing_set_PIR, max_y_val=math.inf) \
        .sort_values(by="test_results", ascending=False)
    print(results_data)
    plt.show()


if __name__ == '__main__':
    main()
