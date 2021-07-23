"""! This script is used to train neural networks to predict gait characteristics from the sensors data"""

import copy
import glob
import itertools
import pickle
import random
import time
from typing import Union, Any, cast, Type, Tuple, List, Dict
import os

import matplotlib
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

from training_models import CustomModel, PretrainedModel, \
    MultipleWaveletsModule, MultipleWaveletsLinearModule, ResidualNet

## List of all "pretrained" models that can be used in the training
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

## List of walking parameters on which we can train a network.
# Additional data is used in order to display the results automatically
estimable_parameters = {
    'body_speed': {
        'index': 0,
        'name': 'speed',
        'unit': 'm/s',
        'multiplier': 1,  # /3.6,
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

## Device that handle the CPU usage
cpu = torch.device('cpu')
## Device that handle the GPU usage
cuda = torch.device('cuda')

## Current device used: CPU or GPU
device = cuda if torch.cuda.is_available() else cpu
## Keep track whever we are using the CPU or the GPU
use_cuda = True

def update_device(update_to_cuda: bool) -> torch.device:
    global cpu, cuda, device, use_cuda
    if update_to_cuda:
        device = cuda
    else:
        device = cpu
    use_cuda = update_to_cuda
    return device

def correlation_loss(x, y):
    """! A custom loss function that tries to optimize the correlation of the predictions with the ground truth
    Using this method can be interesting if you want your neural models to have better correlations in their results
    instead of lower errors
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    # compute the correlation
    cost = abs(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) - 1)
    # This is used to replace all NaN values to 1
    cost[cost != cost] = 1
    return cost


def train_model(trainloader,
                validationloader,
                model_class: Type[CustomModel] = MultipleWaveletsModule,
                model_parameters: Dict = None,
                loss_function: Type[nn.Module] = nn.CrossEntropyLoss,
                training_for_autoencoder: bool = False,
                optimizer: Any = None,
                learning_rate: float = 0.010,
                num_epochs: int = 5,
                early_stopping: Union[bool, int] = False,
                display_precision: float = 100,
                verbose: Union[bool, int] = True,
                estimated_parameter: str = 'body_speed'):
    """! Train function, give it your dataset, the parameters of your model, and there you go.
    @param trainloader the training part of your dataset
    @param validationloader the testing / validation part of your dataset
    @param model_class Specify the class of the model you want to train
    @param model_parameters parameters that will be given to the __init__() function of your model
    @param loss_function Specify the loss function you want to use. default is Cross Entropy
    @param training_for_autoencoder will simply change the target for the loss
    @param optimizer Specify your optimizer. default is SGD
    @param learning_rate parameter given to the optimizer
    @param num_epochs number of epochs for the training
    @param early_stopping if True, will stop if the validation loss starts increasing
    If a integer value, defines the "patience" (number of minibatch to study before officially stops for overfitting)
    @param display_precision if verbose, number of minibatch seen when printing results.
    And because we didn't work very well, it also defines the precision for early stopping...
    @param verbose level of verbosity
    @param estimated_parameter The parameter that you want to estimate with your model

    @return A dictionary with training results and the trained model
    """
    if model_parameters is None:
        raise Exception('No model parameters have been provided. Possible keys are: ' + '\n - '.join(
            model_class.__init__.__code__.co_varnames[1:]))

    # Get the index of this parameter to fetch the data
    estimated_parameter_index = estimable_parameters[estimated_parameter]['index']

    # Take a minibatch of size 1, that has usually better training results
    # Also use less GPU memory
    minibatch_size = 1

    # If not done, transform datasets in dataLoader
    if isinstance(trainloader, torch.utils.data.Dataset):
        trainloader = torch.utils.data.DataLoader(trainloader, batch_size=minibatch_size,
                                                  shuffle=False, num_workers=0)
    if isinstance(validationloader, torch.utils.data.Dataset):
        validationloader = torch.utils.data.DataLoader(validationloader, batch_size=2 * minibatch_size,
                                                       shuffle=False, num_workers=0)
    # Create the CNN from the given parameters :
    net = model_class(**model_parameters)
    # Update the device to switch automatically between CPU and GPU
    net.update_device()
    # net.to(device)  # move it to GPU if available
    if verbose:
        print(net)

    # Defining loss function and optimizer
    criterion = loss_function()
    optimizer = optimizer(net.parameters(), lr=learning_rate)

    # Definition of arrays to store the results and draw the learning curves
    val_err_array = np.array([])
    train_err_array = np.array([])
    nb_sample_array = np.array([])

    # store the best system results (early stopping)
    best_val_loss = math.inf
    best_nb_sample = 0
    best_model = copy.deepcopy(net)

    nb_used_sample = 0
    running_loss = 0.0
    display_precision = display_precision if display_precision > 1 else len(trainloader) * display_precision - 1
    print_every = display_precision

    # Stop the training if is_overfitting turns to True
    is_overfitting = False
    # number of iteration that are still allowed (depending on "early_stopping")
    left_before_early_stop = False

    # Know when you have to display losses
    next_print_in = display_precision

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # Stop the training if it's overfitting
        if is_overfitting:
            break

        # Track time
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):  # Iterate over the dataset
            if is_overfitting:
                break
            # get all the inputs and the ground truth
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
                """! Classic function to do a forward pass in the network and compute the loss
                with backpropagation"""
                # zero the parameter gradients
                optimizer.zero_grad()
                current_loss = 0

                outputs = net(pirs0, pirs1, pirs2, pirs3, concat_pir)
                if training_for_autoencoder:  # If we train an autoencoder, set the input also as output
                    # But in this code, no auto-encoding is allowed
                    raise Exception("No autoencoder allowed in this program")
                    # loss = criterion(outputs, inputs)
                else:
                    # Classic loss computation
                    current_loss = criterion(outputs.reshape(-1).float(),
                                             targets[estimated_parameter_index].reshape(
                                                 -1).float())
                try:
                    # I think there can be errors with the backpropagation with some specific optimizer/loss function
                    # So try anyway and don't bother if it throws an error
                    current_loss.backward()
                except:
                    pass
                return current_loss

            # Compute the loss value
            loss = closure()
            # Do the optimization step (update of the network). The closure function must be passed depending
            # on the chosen optimizer
            optimizer.step(closure)  # Need to be improved

            # count how many samples have been used during the training
            nb_used_sample += len(pirs0)
            running_loss += float(loss.item())

            # Delete the variable to avoid memory errors
            del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, footstep_width, \
                cycle_time, stride_length, corridor_distance

            next_print_in -= 1
            # Do a test on the validation set if we already trained a little bit the model
            if next_print_in <= 0:  # print every ""print_every"" mini-batches
                # Reset the counter
                next_print_in = display_precision
                # Compute the mean loss during the training
                train_err = (running_loss / (print_every * minibatch_size))
                running_loss = 0.0
                # evaluation on validation set
                totalValLoss = 0.0
                # Evaluation part, don't update the weights
                with torch.no_grad():
                    # Same process as the training part, but we don't update the model
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

                        # Compute the loss on the validation set
                        totalValLoss += float(closure().item())
                        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, footstep_width, \
                            cycle_time, stride_length, corridor_distance

                # Compute the mean loss during the validation
                val_err = (totalValLoss / len(validationloader))
                # Display the loss results
                if verbose > 2:
                    print('\tEpoch %d batch %5d ' % (epoch + 1, i + 1))
                    print('\tTrain loss : %.3f' % train_err)
                    print('\tValidation loss mean : %.3f' % val_err)
                # Store all results in the stats
                train_err_array = np.append(train_err_array, train_err)
                val_err_array = np.append(val_err_array, val_err)
                nb_sample_array = np.append(nb_sample_array, nb_used_sample)

                # save the model only when loss is better
                if val_err < best_val_loss:
                    best_val_loss = val_err
                    best_nb_sample = nb_used_sample
                    best_model = copy.deepcopy(net)
                    if early_stopping:  # If we want early stopping, we extend the remaining time
                        left_before_early_stop = int(early_stopping)  # Gives 1 if early_stopping === True
                    else:
                        left_before_early_stop = math.inf  # Just to be sure that we don't stop the learning

                else:  # If the validation loss is greater than before, we might be overfitting
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
    # Return statistics about the training
    return {
        'model': best_model,
        'model_name': best_model.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'train_err': train_err_array,
        'val_err': val_err_array,
        'nb_samples': nb_sample_array,
        'best_val_loss': best_val_loss,
        'best_nb_samples': best_nb_sample}


def test_model(model: CustomModel, testloader, estimated_parameter: str = 'body_speed', scalers=None):
    """! Function to test a model on a dataset
    The procedure is the same as the training, except we don't update the model.
    @see train_model()

    @param model The trained model you want to test
    @param testloader The dataset used for the testing
    @param estimated_parameter The name of the walking parameter you want to test on
    @param scalers The list of scalers that is used on the dataset, defined in the creation of dataset function

    @return A dictionary containing stats about the testing
    """
    if not isinstance(testloader, torch.utils.data.DataLoader):
        testloader = torch.utils.data.DataLoader(testloader, batch_size=16,
                                                 shuffle=False, num_workers=0)

    estimated_parameter_index = estimable_parameters[estimated_parameter]['index']

    # We will compute the Mean Absolute Error (MAE), Mean Square Error (MSE), Mean Percentage Error (MPE)
    mae = 0
    mse = 0
    mpe = 0
    # Store the ground truth values and predicted values to make more stats
    real = []
    predicted = []

    # Try to pass the model to GPU if possible
    model.update_device()
    # model.to(device)

    # Iterate over the dataset
    for i, data in enumerate(testloader, 0):
        # get the inputs (and desired outputs)
        pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, \
        _, _, footstep_width, cycle_time, stride_length, corridor_distance = data

        # if possible, move them to GPU
        body_speed, footstep_width, cycle_time, stride_length, corridor_distance = \
            body_speed.to(device), footstep_width.to(device), cycle_time.to(device), stride_length.to(device), \
            corridor_distance.to(device)
        # If we're using the concatenated PIR data, just pass this one on GPU
        if model.use_concat_pir:
            concat_pir = concat_pir.to(device)
        else:
            pirs0, pirs1, pirs2, pirs3 = pirs0.to(device), pirs1.to(device), \
                                         pirs2.to(device), pirs3.to(device)

        # Put all the possible outputs in an array, to then extract only the one that concern us
        targets = [body_speed, footstep_width, cycle_time, stride_length, corridor_distance]
        # Do a forward pass on the model
        outputs = model(pirs0, pirs1, pirs2, pirs3, concat_pir)

        # Store the ground truth and the predictions in an array
        real.extend(targets[estimated_parameter_index].cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

        # Delete all data to flush the memory
        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, outputs

    real, predicted = np.array(real), np.array(predicted)[:, 0]
    # Compute Pearsons correlation and p-value
    corr = np.corrcoef(np.array(real), np.array(predicted))[0, 1]

    # Get metadata for the concerned parameter
    parameter_data = estimable_parameters[estimated_parameter]
    # If a scaler was used on the data, inverse the process
    if scalers is not None:  # and parameter_data['scaler_name'] in scalers:
        predicted = scalers[parameter_data['scaler_name']].inverse_transform(predicted.reshape(1, -1)).reshape(-1)
        real = scalers[parameter_data['scaler_name']].inverse_transform(real.reshape(1, -1)).reshape(-1)

    # Compute the MAE, MSE and MPE
    mse = np.sqrt(((real - predicted) ** 2).mean())
    mae = abs(real - predicted).mean()
    mpe = abs((real - predicted) / real).mean() * 100

    # Return the stats
    return {'mae': mae, 'mse': mse, 'percent_error': mpe, 'correlation': corr}


class PIR_DataLoader(torch.utils.data.Dataset):
    """! Class that creates a dataloader that applies the desired transformations to the data
    before sending them as torch Tensors"""
    def __init__(self, data_file, transform=None, data_inflate: float = 1.0, apply_translation: bool = False):
        """! Constructor
        @param data_file The content of the file that contains the dataset
        @param transform The torchvision's transformation that will be made on the dataset
        @param data_inflate This variable can increase the number of samples. That means an epoch will be longer,
        but more transformations will be made on each sample
        @param apply_translation Apply a random translation on the X-axis for each sample.

        @return A dataloader for the dataset
        """
        ## The content of the file that contains the dataset
        self.data = data_file
        ## The torchvision's transformation that will be made on the dataset
        self.transform = transform
        ## This variable can increase the number of samples. That means an epoch will be longer,
        # but more transformations will be made on each sample
        self.inflate = int(data_inflate)
        ## Apply a random translation on the X-axis for each sample.
        self.apply_translation = apply_translation

    def __getitem__(self, index):
        """! Function called internally when iterating on the dataset
        This will extract the data sample and apply transformations on it

        @return The PIR images used as input for the models and the ground truth outputs"""
        body_speed = 1000
        # Just to check that we use data with not too high walking speed
        while body_speed > 3.0:
            pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction, \
            footstep_width, cycle_time, stride_length, corridor_distance, raw = self.data[
                index // self.inflate]
            # try:
            #     try:
            #         pass
            #         #
            #     except Exception as e:
            #         pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction, raw = self.data[
            #             index // self.inflate]
            # except:
            #     pir0, pir1, pir2, pir3, id_seq, body_speed, corridor, direction = self.data[index // self.inflate]
            index += 1

        # Apply each transformation on the images
        if self.transform is not None:
            pir0 = self.transform(pir0)
            pir1 = self.transform(pir1)
            pir2 = self.transform(pir2)
            pir3 = self.transform(pir3)
            # If we also want a translation on the X-axis, we apply a "roll" function on all our images
            if self.apply_translation:
                trans_force = int(random.random() * pir0.shape[2] / 4) * (-1 if random.random() < 0.5 else 1)
                trans = transforms.Lambda(lambda x: torch.roll(x, trans_force, 0))
                pir0 = trans(pir0)
                pir1 = trans(pir1)
                pir2 = trans(pir2)
                pir3 = trans(pir3)

        # The concatenate PIR image is just the 4 PIR images together
        concat_pir = torch.cat((pir0, pir1, pir2, pir3), 0)

        # Return inputs and outputs
        return pir0, pir1, pir2, pir3, concat_pir, body_speed, corridor, direction, footstep_width, cycle_time, stride_length, corridor_distance

    def __len__(self):
        # The length is the real length, but we multiply it if we have data inflation / augmentation
        return int(len(self.data) * self.inflate)


def plot_confusion_matrix(cm, classes: List):
    """! Function to display the confusion matrix (for models doing classification only)
    I leave this function here if you need to use it in the future (doing action classification for example)

    @param cm The computed confusion matrix
    @param classes The name of the classes for the classification task

    @note Code greatly inspired by https://deeplizard.com/learn/video/0LhiS6yu2qQ
    """
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


def display_classification_test_results(results: dict, classes: List):
    """! Function to display the results when testing a classifier
    This function cannot be used for regression, but might be useful for an action classifier for example

    @param results The results from a model testing
    @param classes The name of the classes for the classification task
    """
    # The result should contain an array for each class, for which the key is the label of the class
    # Should also contain the confusion matrix and a list of results with the key name "global"
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


def visualize_model(model: CustomModel, testloader, estimated_parameter: str = 'body_speed', scalers: dict = None,
                    results_with_color: bool = False):
    """! Function that display the results of a model on a dataset. This will print the outputs and the errors
    @param model The neural model you want to see
    @param testloader The dataset on which we test the model
    @param estimated_parameter The walking parameter we want to consider with the model
    @param scalers A dictionary containing the scalers used to transform the data in the dataset
    @param results_with_color Set this to True if you want to display the predictions with colors and shapes
    depending on the corridor/direction used.
    """
    # get metadata on the parameter we consider
    parameter_data = estimable_parameters[estimated_parameter]
    # Pass the model to the GPU if possible
    model.update_device()
    # model.to(device)
    # Put the model in "eval" mode to avoid to modify the model
    model.eval()
    print(model)
    if not isinstance(testloader, torch.utils.data.DataLoader):
        testloader = torch.utils.data.DataLoader(testloader, batch_size=1,
                                                 shuffle=False, num_workers=0)

    real = []
    predicted = []
    corridors = []
    directions = []
    # Same process as test_model() or train_model()
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

        # Store the ground truth and the predictions
        real.extend(targets[parameter_data['index']].cpu().detach().numpy().tolist())
        predicted.extend(outputs.cpu().detach().numpy().tolist())

        del pirs0, pirs1, pirs2, pirs3, concat_pir, body_speed, outputs

        # Store the positions in the corridor and the direction taken for stats
        corridors.extend(corridor.cpu().detach().numpy().tolist())
        directions.extend(direction.cpu().detach().numpy().tolist())

    real, predicted = np.array(real), np.array(predicted)[:, 0]

    # Easy fix for some errors when the "real" variable contains some NaN (Should be filtered from the beginning...)
    directions = np.array(directions)[real > 0]
    corridors = np.array(corridors)[real > 0]
    predicted = predicted[real > 0]
    real = real[real > 0]

    # Save the results in a CSV
    pd.DataFrame({'real': real, 'predicted': predicted}).to_csv("CNN_results.csv")
    errors = pd.Series(real - predicted)
    errors_percent = pd.Series(100 * (real - predicted) / real, name = "Real % error")
    real, predicted = np.array(real) * parameter_data['multiplier'], np.array(predicted) * parameter_data[
        'multiplier']
    # Inverse the transformation made on the data if possible
    if scalers is not None:  # and parameter_data['scaler_name'] in scalers:
        predicted = scalers[parameter_data['scaler_name']].inverse_transform(predicted.reshape(1, -1)).reshape(-1)
        real = scalers[parameter_data['scaler_name']].inverse_transform(real.reshape(1, -1)).reshape(-1)

    # Compute the absolute errors and relative errors to display them
    scaled_error = pd.Series(real - predicted, name = "Error after rescaling data")
    abs_errors = abs(real - predicted)
    abs_errors_percent = abs(100 * (real - predicted) / real)
    sorted_abs_errors = pd.Series([e for _, e in sorted(zip(real, abs_errors))], name='Abs Errors')
    sorted_abs_errors_percent = pd.Series([e for _, e in sorted(zip(real, abs_errors_percent))], name='Abs Error (%)')
    pearson_coef, p_value = pearsonr(real, predicted)

    # Display a scatter plot with the predictions and ground truth
    plt.subplot(3, 2, 1)
    colors = [(1, 0, 0, 1) if c == -1 else (0, 1, 0, 1) if c == 0 else (0, 0, 1, 1) for c in corridors]
    directions = np.array(directions)
    corridors = np.array(corridors)
    colors = np.array(colors)
    # If you want colors and shape, we need to do 2 scatter plots
    if results_with_color:
        plt.scatter(real[directions < 0], predicted[directions < 0], marker="<", c=colors[directions < 0])
        plt.scatter(real[directions > 0], predicted[directions > 0], marker=">", c=colors[directions > 0])
        plt.legend([(plt.plot([], 'b<')[0], plt.plot([], 'b>')[0]),
                    matplotlib.patches.Patch(color=(1, 0, 0)),
                    matplotlib.patches.Patch(color=(0, 1, 0)),
                    matplotlib.patches.Patch(color=(0, 0, 1))],
                   ['Direction', 'East', 'Center', 'West'], handler_map={tuple: HandlerTuple(ndivide=None)})
    # Otherwise a single scatter plot is enough
    else:
        plt.scatter(real, predicted, marker=".", c="black")

    # Display the unity line (y = x)
    plt.plot(list(plt.xlim()), list(plt.xlim()), '--', c=(0, 0, 0, .2))

    plt.ylabel(f"Measured {parameter_data['name']} ({parameter_data['unit']})")
    plt.xlabel(f"Reference {parameter_data['name']} ({parameter_data['unit']})")
    plt.title(f"Results from the CNN (correlation : {round(pearson_coef, 2)} | "
              f"{'p < 0.001' if p_value < 0.001 else 'p = ' + str(round(p_value, 4))})")

    # Display the absolute error as an histogram
    plt.subplot(3, 2, 2)
    plt.title('Error distribution')
    plt.hist(abs_errors, bins=30, color=(237 / 255, 190 / 255, 36 / 255, 1), density=1)
    # Calculate percentiles to display a mark at each quartile
    quant_25, quant_50, quant_75, quant_95 = np.percentile(abs_errors, [25, 50, 75, 95])
    quants = [[quant_25, 2, 25], [quant_50, 1.8, 50], [quant_75, 1.6, 75],
              [quant_95, 1.4, 95]]
    for i in quants:
        plt.plot([i[0], i[0]], [0, i[1]], ':', c='blue')
        plt.text(i[0], i[1] * 1.05, f"{i[2]}th", size=11, alpha=0.85, ha='center', va='bottom')

    plt.xlabel(f"Error in {parameter_data['unit']}")
    plt.ylabel('Frequency')

    # Display a scatter plot with the absolute error depending on the value of the ground truth
    plt.subplot(3, 2, 3)
    plt.axhline(np.median(abs_errors), color=(0, 0, 0, .5), linestyle='--')
    plt.text((min(real) + max(real)) / 2, np.median(abs_errors),
             f"median = {round(float(np.median(abs_errors)), 2)}{parameter_data['unit']}", va='bottom', ha='center')

    if results_with_color:
        plt.scatter(real[directions < 0], errors[directions < 0], marker='<', c=colors[directions < 0])
        plt.scatter(real[directions > 0], errors[directions > 0], marker='>', c=colors[directions > 0])
    else:
        plt.scatter(real, abs_errors, marker='.', c='black')
    plt.title(f"Error depending on the real {parameter_data['name']}")
    plt.xlabel(f"Real {parameter_data['name']}")
    plt.ylabel(f"Error in {parameter_data['unit']}")

    # Display the relative error depending on the value of the ground truth
    plt.subplot(3, 2, 4)
    plt.axhline(np.median(abs_errors_percent), color=(0, 0, 0, .5), linestyle='--')
    plt.text((min(real) + max(real)) / 2, np.median(abs_errors_percent),
             f"median = {round(float(np.median(abs_errors_percent)), 2)}%",
             va='bottom', ha='center')
    if results_with_color:
        plt.scatter(real[directions < 0], errors_percent[directions < 0], marker='<', c=colors[directions < 0])
        plt.scatter(real[directions > 0], errors_percent[directions > 0], marker='>', c=colors[directions > 0])
    else:
        plt.scatter(real, abs_errors_percent, marker='.', c='black')
    plt.title(f"% Error depending on the real {parameter_data['name']}")
    plt.xlabel(f"Real {parameter_data['name']}")
    plt.ylabel('% Error')

    # Display the absolute relative error as an histogram
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

    # Finally, compute the real error distribution to see the shape of the curve
    plt.subplot(3, 2, 6)
    plt.title('Error distribution')
    scaled_error.plot(kind='hist', density=1)
    scaled_error.plot(kind='kde')
    # Calculate percentiles
    quant_25, quant_50, quant_75, quant_95 = np.percentile(abs_errors, [25, 50, 75, 95])
    quants = [[quant_25, 2, 25], [quant_50, 1.8, 50], [quant_75, 1.6, 75],
              [quant_95, 1.4, 95]]
    # for i in quants:
    #     plt.plot([i[0], i[0]], [0, i[1]], ':', c='blue')
    #     plt.text(i[0], i[1] * 1.05, f"{i[2]}th", size=11, alpha=0.85, ha='center', va='bottom')

    plt.xlabel(f"Error in {parameter_data['unit']}")
    plt.ylabel('Frequency')

    # If the parameter can apply tight layout (labels not too long to write), do it
    if parameter_data['apply_tight_layout']:
        plt.tight_layout()

    # Display the error data
    print(pd.concat([scaled_error, errors_percent, sorted_abs_errors, sorted_abs_errors_percent,
                     pd.Series(sorted_abs_errors_percent.values[directions < 0], name='Error (%) South-North'),
                     pd.Series(sorted_abs_errors_percent.values[directions > 0], name='Error (%) North-South'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == -1], name='Error (%) East'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == 0], name='Error (%) Center'),
                     pd.Series(sorted_abs_errors_percent.values[corridors == 1], name='Error (%) West')],
                    axis=1).describe())


def create_all_combinaisons(hyperparameters: dict):
    """! From a set of parameters, create all the combinations possible
    @param hyperparameters A dictionary with all the variable you want to vary

    @return An array with all possible combination as dictionaries
    """
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


def try_all_models(hyperparameters: dict,
                   train_set,
                   validation_set,
                   testing_set,
                   training_parameters: dict = {},
                   model_class: Type[CustomModel] = MultipleWaveletsModule,
                   loss_function: Type[nn.modules.loss.Module] = nn.CrossEntropyLoss,
                   optimizer=False,
                   training_for_autoencoder: bool = False,
                   learning_rate: float = 0.001,
                   num_epochs: int = 5,
                   display_precision: Union[float, int] = 1.0,
                   early_stopping: Union[bool, int] = True,
                   save_as: Union[bool, str] = False,
                   start_from_beginning: bool = True,
                   verbose: Union[bool, int] = 3,
                   ignored_files: str = '_useless_',
                   scalers: List = None,
                   max_number_of_models: int = math.inf,
                   estimated_parameter: str = 'body_speed'):
    """! Launch the training function for every possible combination of neural network.
    @param hyperparameters All the parameters that can be changed in the model
    @param train_set The training set
    @param validation_set The validation set
    @param testing_set The testing set
    @param training_parameters A dictionary with possible parameters for the training
    @param model_class Specify the class of the models you want to train
    @param loss_function Specify the loss function you want to use. default is Cross Entropy (classification task)
    @param training_for_autoencoder Will simply change the target for the loss, but some small changes should be done in
    the training if you want to use it
    @param optimizer Specify your optimizer. default is SGD
    @param learning_rate Learning rate parameter given to the optimizer
    @param num_epochs Number of epochs for the training. This can be shorten if early_stopping is set to True
    @param display_precision The rate at which we do a validation pass, and display the current loss results
    of the training
    @param early_stopping if True, will stop if the validation loss starts increasing
    If an integer value is given, defines the "patience"
    (number of minibatch to study before officially stops because of overfitting)
    @param save_as Specify the name of the file where we save all the statistics (including the model itself)
    a timestamp is added to the file name
    @param start_from_beginning Remove all previously saved models if True, otherwise will ignore previously calculated
    combinations
    @param verbose level of verbosity
    @param ignored_files The prefix used for files that are considered "useless" (results really, really bad)
    @param scalers The list of scalers used to transform the dataset
    @param max_number_of_models The number of model you want to train
    @param estimated_parameter The walking parameter considered by the models
    """
    models = []
    # Compute all possible parameters of the networks (mainly: the structure)
    combinations = create_all_combinaisons(hyperparameters)  # Get all possible combinations
    combinations_done = []

    # Compute all possible training parameters (optimizers, learning rate, etc...)
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
    # Do a training for each possible network structure
    for i, parameters in enumerate(combinations, 1):
        # Empty the cuda cache to avoid memory issues
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

            # Configure the training parameters for this network
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

            # Store all parameters used for this training
            current_model = {}
            current_model['parameters'] = parameters
            current_model['training_parameters'] = used_training_parameters

            # Save the training statistics (with the model itself), the testing statistics and the parameters used
            current_model['training'] = train_model(trainloader=train_set,
                                                    validationloader=validation_set,
                                                    **used_training_parameters)
            if verbose:
                print("Training ended, testing the model")
            # Clear memory cache
            torch.cuda.empty_cache()
            current_model['test'] = test_model(current_model['training']['model'], testing_set,
                                               estimated_parameter=estimated_parameter,
                                               scalers=scalers)
            if verbose:
                print(current_model['test'])

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

def display_models(filename: str, test_loader, ignore_if_accuracy_lower: float = 0.0,
                   ignore_if_val_err_higher: float = math.inf, max_y_val: float = math.inf,
                   ignored_files: str = '_useless_', scalers: List = None, order_by: str = 'correlation',
                   bigger_is_better: bool = True, display_all_models: bool = False) -> Tuple[Any, pd.DataFrame]:
    """! Display all the trained models with the learning curve and the results of the best model
    @param filename Name of the file where trained model are stored
    @param test_loader Dataset containing test data
    @param ignore_if_accuracy_lower Unused parameter
    @param ignore_if_val_err_higher Unused parameter
    @param max_y_val Maximum loss value displayed on the learning curves
    @param ignored_files Prefix used for models that should be ignored (very bad results)
    @param scalers List of scalers used to transform the dataset
    @param order_by The criteria to find the best model. Criteria can be
    - correlation
    - MAE
    - MSE
    @param bigger_is_better Set to True if the "order_by" must be big to be a good model (ex: for "correlation")
    and set to False if "order_by" must be low (ex: "MSE" or "MAE")
    @param display_all_models Set to True to see the results of each model, and not just the best one
    """
    order_by = str(order_by).lower()
    models_results = []
    validation_results = []
    best_results = None
    best_index = 0

    # all_plots = []
    top_plots = []

    top_models = []
    min_score_of_tops = math.inf

    # Open each file containing models
    for i, file in enumerate(glob.glob(os.path.splitext(filename)[0] + "*" + os.path.splitext(filename)[1])):
        if ignored_files in file:
            # If it's a "useless" model (the results where very bad), just ignore it
            continue
        # Otherwise, unpickle the model and its training stats from the file
        models = pickle.load(open(file, "rb"))

        for model in models:
            try:
                # If the validation loss is lower than the limit you fixed, add the plot of the training and validation
                # learning curve
                # if model['training']['val_err'][-1] <= max_y_val:
                #     all_plots.append({'x': np.array(model['training']['nb_samples']) / 1000,
                #                       'y': model['training']['val_err'],
                #                       'y_train': model['training']['train_err'],
                #                       'result': model['test']['correlation'],
                #                       })
                # Store all the stats and training parameters in a variable
                _models_results = {**model['training_parameters']['model_parameters'],
                                   'number_of_samples': f"{model['training']['best_nb_samples']:,}",
                                   'test_results': model['test']['correlation'], 'mae': model['test']['mae'],
                                   'mse': model['test']['mse'], 'percent_error': model['test']['percent_error'],
                                   'id': i,
                                   **model['training_parameters']}
                # Store the results in an array
                models_results.append(copy.deepcopy(_models_results))
                # If you're working with an auto-encoder, change some value for display purpose
                # (I have to say, I did that a long time ago, I don't remember why it's useful...)
                # (But anyway, we're not working with auto-encoders)
                for attr in _models_results:
                    if hasattr(_models_results[attr], 'dec') and hasattr(_models_results[attr], 'enc'):
                        models_results[-1]['number_of_layers'] = len(_models_results[attr].enc) // 3
                        models_results[-1]['kernel_size'] = str(
                            list(_models_results[attr].enc[0].parameters())[0][0].size(1))
                        models_results[-1] = dict(models_results[-1])
                        del models_results[-1][attr]

                # If the model is having the best results (by comparing the "order_by" result), save it as "best model"
                if (best_results is None or np.isnan(best_results['test'][order_by])) or \
                        (not np.isnan(model['test'][order_by]) and
                         model['test'][order_by] * (-1 if bigger_is_better else 1) < best_results['test'][order_by] * (
                         -1 if bigger_is_better else 1)):
                    best_results = model
                    best_index = i

                try:
                    # Display briefly the results of each model
                    print(model['test'])
                    print(model['training_parameters']['estimated_parameter'])
                    # If desired, display the results of the model by plotting it's predictions on a dataset
                    if display_all_models:
                        torch.cuda.empty_cache()
                        visualize_model(model['training']['model'], testloader=test_loader,
                                        estimated_parameter=model['training_parameters']['estimated_parameter'],
                                        scalers=scalers)
                        plt.show()
                except Exception as e:
                    print(f"Whoops, Exception: {e}")
                    plt.close()
                    pass

                # Store the top 10 results to display their learning curves
                # If there is less than 10 models, add this one in the list
                if len(top_models) < 10 and not np.isnan(model['test'][order_by]):
                    top_models.append(model)
                    min_score_of_tops = min([m['test'][order_by] for m in top_models])
                # If there is more than 10 models, check if one of them is worse than this one
                elif not np.isnan(model['test'][order_by]) and model['test'][order_by] * (
                -1 if bigger_is_better else 1) > min_score_of_tops * (-1 if bigger_is_better else 1):
                    # Remove the worse model, add the new one and compute the score of the new worse model
                    top_models.remove([m for m in top_models if m['test'][order_by] == min_score_of_tops][0])
                    top_models.append(model)
                    min_score_of_tops = min([m['test'][order_by] for m in top_models])

            except Exception as e:
                print(e)
                pass

    for model in top_models:
        # Configure the plots for the learning curves of the best models
        if model['training']['val_err'][-1] <= max_y_val:
            top_plots.append({'x': np.array(model['training']['nb_samples']) / 1000,
                              'y': model['training']['val_err'],
                              'y_train': model['training']['train_err'],
                              'result': model['test'][order_by],
                              'name': model['training']['model'].__class__.__name__,
                              })
    # Sort the results by their result
    top_plots.sort(key=lambda x: x['result'], reverse=bigger_is_better)
    legend_train_val = plt.legend([plt.plot([], c=(1, 0, 0, 1))[0], plt.plot([], c=(1, 0, 0, .4))[0]],
                                  ['Validation', 'Training'], loc=3)
    # Display the learning curves
    for plot in top_plots:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        # Training curve
        plt.plot(plot['x'], plot['y_train'], color=color, alpha=.4)
        # Validation curve
        plt.plot(plot['x'], plot['y'], color=color, label=plot['name'])
        plt.text(plot['x'][-1], plot['y'][-1],
                 str(round(plot['result'], 2)), horizontalalignment="left",
                 fontsize="x-small", path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
        plt.xlabel('Number of samples (x1000)')
        plt.ylabel('L2 Loss')
        plt.title('Learning curve for training and validation\nWith resulting ' + order_by + ' on the test set')
    plt.yscale('log')
    plt.legend()
    plt.gca().add_artist(legend_train_val)
    plt.tight_layout()
    plt.show()

    # Put all statistics in a DataFrame
    df = pd.DataFrame(models_results)
    # Change a bit the visual for some values
    df[['test_results', 'mae', 'mse', 'percent_error']] = round(df[['test_results', 'mae', 'mse', 'percent_error']], 3)
    df['model_class'] = df['model_class'].apply(lambda model: model.__name__)
    df['trained_model'] = df.apply(axis=1, func=
    lambda row: row['trained_model'].__name__ if 'trained_model' in row
                                                 and row['model_class'] == 'PretrainedModel'
                                                 and hasattr(row['trained_model'], '__name__') else None)

    df.rename(columns={'kernel_size': 'kernel', 'number_of_filters': 'filters', 'number_of_layers': 'layers',
                       'test_results': 'correlation', 'number_of_conv_per_layer': 'conv/layer',
                       'reduce_last_conv': 'reduce',
                       'use_concat_pir': 'concat_pirs', 'percent_error': '%err'},
              inplace=True)
    # Keep a DataFrame with all data, and one with only some columns
    full_df = df
    df = df[['mid_output_size',
             'kernel',
             'filters',
             'layers',
             'conv/layer',
             'hidden_cells',
             # 'dropout',
             # 'concat_pirs',
             'model_class',
             'correlation',
             'mae',
             'mse',
             '%err',
             'pure_CNN',
             'pretrained',
             'trained_model'] +
            # (['reduce'] if 'reduce' in df.columns else []) +
            # (['use_exponential'] if 'use_exponential' in df.columns else []) +
            (['layer_relu'] if 'layer_relu' in df.columns else [])
            ]

    # Must delete all variables that may contain models on CUDA, to avoid falling out of memory!
    del models
    torch.cuda.empty_cache()
    print(best_results['test'])
    try:
        # Display the best model's results
        visualize_model(best_results['training']['model'], testloader=test_loader,
                        estimated_parameter=best_results['training_parameters']['estimated_parameter'],
                        scalers=scalers)
    except Exception as e:
        print(e)
    plt.show()

    return df, full_df

def compute_morlet_transform(pir_data) -> np.ndarray:
    """! Function that creates an image from the Continuous Wavelet Transform of a signal
    @param pir_data The signal values as a time series

    @return An image of the CWT of the signal
    """
    data_norm = waipy.normalize(pir_data)
    # No idea what this parameter alpha is, but they recommend to do that in the doc
    alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]
    ## The signal has data every 1/40 of a second
    dt = 1 / 40
    ## No idea what is this parameter
    dj = 1 / 100
    ## Default values of the CWT of the function in R language, so I did the same computations
    lowerPeriod = 2 * dt
    upperPeriod = len(data_norm) * dt / 3
    # Compute the CWT
    result = waipy.cwt(data_norm, dt, 1, dj, lowerPeriod, upperPeriod, alpha, 6, mother='Morlet', name='name')
    # Get the "power level". I took the square root to have the amplitude (low values and low variance),
    # but it might not be the best solution, maybe keep the power level
    powers = np.array(np.sqrt(result['power'])).astype('uint8')
    # Make it an image
    tmp_img = Image.fromarray(powers)  # , mode = 'I;16')
    # Create a BIG black image and paste the "power level image" on it
    res = Image.fromarray(np.zeros((1024, 2048)))  # , dtype=np.uint16), mode='I;16')
    res.paste(tmp_img, (0, 1024 - tmp_img.height))
    # Resize to keep low memory usage
    res = res.resize((128, 128), resample=Image.NEAREST)
    return res


def normalize(df: pd.DataFrame, concerned_columns: Union[list, str] = 'all', ignored_columns: list = None) \
        -> Tuple[pd.DataFrame, Dict]:
    """! Function to normalize the data from a DataFrame
    @param df The DataFrame containing all the data
    @param concerned_columns The names of the columns you want to normalize. Set to 'all' to normalize everything
    @param ignored_columns If "concerned_columns" is set to 'all', you can give the name of columns you
    actually DON'T want to normalize

    @return A tuple containing the new DataFrame with normalized values and a list of the scalers used in the
    normalization process for each column.
    """
    result = df.copy()
    if concerned_columns == 'all':
        concerned_columns = result.columns.tolist()
    if ignored_columns is not None:
        for col in ignored_columns:
            if col in concerned_columns:
                concerned_columns.remove(col)

    scalers = {}
    # For each column you want to normalize, create a scaler and process the data
    for feature_name in concerned_columns:
        scaler = sklearn.preprocessing.MinMaxScaler()
        df[feature_name] = scaler.fit_transform(df[feature_name].values.reshape(-1, 1))
        scalers[feature_name] = scaler
    return df, scalers


def precompute_morlet_transforms(raw_data: pd.DataFrame,
                                 pir_transforms_filename: str = 'all_pirs_transformed.data',
                                 output_filename: Union[str, bool] = True) -> Dict:
    """! Function that compute all the Continuous Wavelet Transforms of the dataset and store them in a Pickle file.
    @param raw_data A DataFrame containing all the data of the dataset. This might be the content of 'gait_values.csv'
    @param pir_transforms_filename The name of the file in which you want to save the CWT data
    @param output_filename Can be a boolean set to True to effectively save the CWT data or False to not save the file
    and just return the results in the script.\n Can also be a string if you want to save the CWT data in another file.

    @return The dataset values containing the Continuous Wavelet Transform of each sensor and the ground truth of all
    walking parameters.

    @note If the file "pir_transform_filename" already exists, none of the sequences contained inside is computed again.
    For a fresh start (your CWT computation has been modified, there was error on the data or else), you must delete
    the previous file in order to compute completely all the sequences
    """
    if isinstance(output_filename, bool) and output_filename is True:
        output_filename = pir_transforms_filename
    start_time = time.time()
    # Get all sequence ID
    sequence_ids = pd.unique(raw_data['sequence_id'])
    # Normalize the data from the dataset
    data, scalers = normalize(raw_data, 'all',
                              ignored_columns=['time', 'nb_steps_motion', 'nb_steps_distance', 'direction',
                                               'corridor', 'sequence_id', ])

    all_pirs_transformed = {'data': [], 'scalers': scalers}
    # If the output file already exists, take it to avoid some computations
    if os.path.exists(pir_transforms_filename):
        all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))
    # Filter sequences where the corridor wasn't found (probably data error),
    # where the direction is set to 0 (person did a half-turn)
    # (I think seq[5] is the walking speed, I think it was a quick fix for some corrupted data.
    # You can probably remove it now)
    all_pirs_transformed['data'] = list(filter(lambda seq: seq[5] < 10 and not np.isnan(seq[6]) and seq[7] != 0,
                                               all_pirs_transformed['data']))
    # keep the sequence ID of those that are not good (half-turn done or corrupted data) so you don't have to
    # do the computations on them
    ignored_sequences = set([seq[4] for seq in list(filter(lambda seq: seq[5] > 10 or np.isnan(seq[6]) or seq[7] == 0,
                                                           all_pirs_transformed['data']))])
    # Keep the sequence ID of those that are already in the previous file, you then don't have to compute them also
    preexisting_sequences = set([seq[4] for seq in all_pirs_transformed['data']])

    # For each sequence, check if a computation is needed. If yes, compute the CWT and the walking parameters values.
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
                with HiddenPrints(False):
                    # Compute the CWT image of each sensor
                    pir0 = compute_morlet_transform(seq['pir0'])
                    pir1 = compute_morlet_transform(seq['pir1'])
                    pir2 = compute_morlet_transform(seq['pir2'])
                    pir3 = compute_morlet_transform(seq['pir3'])
                    # Store the CWT images and the gait features in an array
                    all_pirs_transformed['data'].append([pir0, pir1, pir2, pir3, seq_id, speed, corridor, direction,
                                                         footstep_width, cycle_time, stride_length, corridor_position,
                                                         raw])
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

    # If we want to save the file, do it
    if output_filename is not None:
        print(f"Saving {len(all_pirs_transformed['data'])} sequences into {output_filename}")
        pickle.dump(all_pirs_transformed, open(output_filename, 'wb+'))
        print('Done.')
    return all_pirs_transformed


def training(path_to_save_cnn: str,
             pir_transforms_filename: str = 'all_pirs_transformed.data',
             estimated_parameter: str = 'body_speed',
             number_of_models_trained: int = math.inf) -> None:
    """! Main training function that make dataloader from the dataset, defines the model parameters
    and training parameters that will be used to train all the models
    @param path_to_save_cnn Path where the trained models should be saved
    @param pir_transforms_filename Path to the file containing the dataset (the one with Continuous Wavelet Transforms
    computed) to create the dataloader
    @param estimated_parameter The walking parameter that will be estimated by the models
    @param number_of_models_trained You can give a maximum number of models to train before the program stops
    """
    # Get the dataset
    all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))
    # Required "resize" to fit the CNNs
    input_size = 128

    # Load the data through a DataLoader on which we can apply some transformation to create a more robust model
    # The "data_inflate" can increase the number of data by multiplying them and apply different transformations
    dataset = PIR_DataLoader(all_pirs_transformed['data'], transform=transforms.Compose([
        # First, resize each image to fit the networks input size
        transforms.Resize((input_size, input_size), transforms.InterpolationMode.NEAREST),
        # Transform the images in torch Tensors, which is the "official" type that models work with
        transforms.ToTensor(),
        # Make sure the tensors are defined as floating values
        transforms.Lambda(lambda x: x.float()),
        # Decomment the next lines to add more noise to the dataset
        # Add some random noise on the images
        # transforms.Lambda(lambda x: x * (1 - torch.randn_like(x) * 0.01 + 0.005)),
        # Blur a little the image
        # transforms.Lambda(lambda x: transforms.GaussianBlur(
        #     2 * int(random.random() * 10) + 1, (0.1, 10.0))(x)),
    ]),
                             # For the training, increase the sample size x3 and allow translations of the images
                             # on the X-axis
                             data_inflate=3.0, apply_translation=True)

    # Train size = 70%, Validation = 10%, Testing = 20%
    ratio_train_set = 70 / 100
    ratio_valid_set = 10 / 100

    train_size = int(ratio_train_set * len(dataset))
    valid_size = int(ratio_valid_set * len(dataset))
    test_size = len(dataset) - (train_size + valid_size)
    # Split randomly the samples to have a training set, a validation set and a testing set
    training_set_PIR, validation_set_PIR, testing_set_PIR = torch.utils.data.random_split(dataset,
                                                                                          [train_size, valid_size,
                                                                                           test_size])
    # We've chose low minibatch size to be more efficient and be sure to avoid memory issues, but takes more time
    minibatch_size = 1
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
        'use_exponential': False,  # [True, False, 'pow'],
        'number_of_outputs': 1,

        # These params are used for the transfer learning, it's the type of model used (AlexNet, SqueezeNet, etc...)
        'trained_model': list(all_pretrained_models.values()),
        'pretrained': [True, False]
    }

    # Those parameters modify the type of training used for each network
    # The parameters used for a model are chosen randomly and thus, does not increase the number of trainings
    # In a normal use, you should choose only one optimizer, one lr, one loss function, etc...
    training_parameters = {
        'optimizer': [optim.Rprop, optim.Adagrad, optim.ASGD, optim.LBFGS, optim.Adam, optim.SGD, optim.Adadelta,
                      optim.Adamax, optim.AdamW],
        'learning_rate': [0.0001, 0.00001, 0.00005],
        'loss_function': [nn.MSELoss, nn.L1Loss],
        'model_class': [MultipleWaveletsModule, ResidualNet, PretrainedModel],
        # [Residualnet, MultipleWaveletsLinearModule, MultipleWaveletsModule, PretrainedModel],
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
                   scalers=all_pirs_transformed['scalers'],
                   max_number_of_models=number_of_models_trained)


def display(path_to_save_cnn: str,
            pir_transforms_filename: str = 'all_pirs_transformed.data',
            display_all_models: bool = False) -> None:
    """! Main function to display the results of your trained networks
    @param path_to_save_cnn The location where you stored your trained networks
    @param pir_transforms_filename The name of the file containing the dataset
    @param display_all_models If True, we will plot the the predictions and the errors of the model on the dataset,
    instead of just the statistics computed during the training part

    @see This can be done only after doing a training()
    """
    # Required "resize" to fit the CNNs
    input_size = 128

    # Open the dataset
    all_pirs_transformed = pickle.load(open(pir_transforms_filename, 'rb'))

    # Load the data through a DataLoader on which we can apply some transformation to create a more robust model
    # The "data_inflate" can increase the number of data by multiplying them and apply different transformations
    dataset = PIR_DataLoader(all_pirs_transformed['data'], transform=transforms.Compose([
        # Resize the images to fit the input size of the networks
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        # Transform the images in tensors
        transforms.ToTensor(),
        # Make sure to use floating values
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
    # But you can change it by commenting and decommenting some of the next lines:
    ordering = ('correlation', True)
    # ordering = ('mae', False)
    # ordering = ('mse', False)
    # ordering = ('percent_error', False)

    # Fetch the results of all your trained models
    short_results, full_results = display_models(path_to_save_cnn, test_loader=testing_set_PIR,
                                                 scalers=all_pirs_transformed['scalers'], order_by=ordering[0],
                                                 bigger_is_better=ordering[1], display_all_models=display_all_models)
    # The "full_results" contains all the information for each models (optimizer, loss function, parameters, etc...)
    # full_results.sort_values(by=ordering[0], ascending=not ordering[1], inplace=True)
    # print(full_results)
    print(short_results.sort_values(by=ordering[0], ascending=ordering[1]).head(500))


def merge_models_in_one_file(path_to_cnn: str, ignored_files: str = '_useless_') -> None:
    """! When doing a training, each model is stored in a different file. With this function you can merge all
    files into a single one. It might not always be a good idea as the file might be way too big at the end if
    lots of training has been done, and I'm not sure about the consequences on the GPU memory when unpickling the file.
    @param path_to_cnn The path where your models have been saved
    @param ignored_files The prefix given to models that had very bad results and can be ignored

    @see If you want to first remove useless files: see remove_useless_models()
    """
    filename = os.path.split(os.path.splitext(path_to_cnn)[0])[-1]
    ext = os.path.splitext(path_to_cnn)[1]
    directory = os.path.join(*os.path.split(os.path.splitext(path_to_cnn)[0])[:-1])

    models = []
    ignored_models = []
    files_to_delete = []
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + "*" + os.path.splitext(path_to_cnn)[1])
    # Iterate on each file containing models
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        if ignored_files in file:
            # If this file should be ignored, don't do anything
            continue
        try:
            # Extract the models from the file
            these_models = pickle.load(open(file, "rb"))
            # Take out the "useless" models that can be stored here and temporarily save them in another file
            # marked with the "useless" pattern
            ignored_models = list(filter(lambda model: np.isnan(model['test']['correlation']), these_models))
            pickle.dump(ignored_models, open(os.path.join(directory, filename + ignored_files + str(i) +
                                                          '_' + str(time.time()) + ext), "wb+"))
            these_models = list(filter(lambda model: not np.isnan(model['test']['correlation']), these_models))
            models.extend(these_models)
            # This file can be deleted at the end
            files_to_delete.append(file)
        except Exception as e:
            print(e)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Put all the models in a single file, and remove all other files
    print(f"Saving {len(models)} models...")
    pickle.dump(models, open(os.path.join(directory, filename + "_" + str(time.time()) + ext), "wb+"))
    for file in files_to_delete:
        os.remove(file)

    # Now we also merge the "useless" models, marked with a pattern in the filenames
    print('Merging useless models...')
    models = []
    files_to_delete = []
    all_files = glob.glob(os.path.splitext(path_to_cnn)[0] + ignored_files + "*" + os.path.splitext(path_to_cnn)[1])
    for i, file in enumerate(all_files):
        print(f"{round(100 * (i + 1) / len(all_files), 1)}% ({i + 1} / {len(all_files)})")
        try:
            # Extract them all and add them to a list
            these_models = pickle.load(open(file, "rb"))
            models.extend(these_models)
            files_to_delete.append(file)
        except Exception as e:
            print(e)
    # Put all of them in a single file
    pickle.dump(models, open(os.path.join(directory, filename + ignored_files + str(time.time()) + ext), "wb+"))

    # Remove all the other useless files
    print('Cleaning')
    for file in files_to_delete:
        os.remove(file)

    print('Done.')


def remove_useless_models(path_to_cnn: str, ignored_files: str = '_useless_') -> None:
    """! Function to store the models with really bad results in a file containing the "ignore_file" pattern, so
    it won't be taken in consideration during the display process of the training results
    @param path_to_cnn The path where your models have been saved
    @param ignored_files The prefix given to models that had very bad results and can be ignored

    @see The process is exactly the same as merge_models_in_one_file(), but without the merging of the good models.
    """
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


def main():
    # Example of usage: python deep_neural_training.py --training --parameter all
    #  or
    #                   python deep_neural_training.py --update-data --display --parameter body_speed
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
    parser.add_argument('--display-all', dest='display_all', help='Set this value to True if you want to see the '
                                                                  'predictions and errors of each trained model '
                                                                  'during the "display" part, or set to False to '
                                                                  'see only the "best one".',
                        action='store_true')

    args = parser.parse_args()
    do_training = args.train
    do_display = args.display
    update_data = args.update_data
    merge_files = args.merge_files
    clean_files = args.clean_files
    estimated_parameter = args.estimated_parameter
    display_all_models = args.display_all

    if (do_training or do_display or clean_files or merge_files) and estimated_parameter is None:
        parser.print_help()
        print("Please select a parameter to train/display your models")
        exit()

    # The random seed used so the data splits stays the same and results are reproducible
    torch.manual_seed(42)

    ## Change this variable to change the dataset used (for now values are "ankles" or "hips")
    sensors_height = 'ankles'


    # Path to save the binary files containing the data about our trained models
    root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{estimated_parameter}/"
    path_to_save_cnn = root_to_cnn + "convolution_network.model"
    pir_transforms_filename = f"all_pirs_transformed_{sensors_height}.data"

    ## Variable used when training all the walking parameters so it trains 5 models per parameter,
    # then increase to 10, etc...
    number_of_models_per_parameter = 5

    # If the --update-data is given, update the CWT data from the CSV
    if update_data:
        # All data comes from the CSV file created through fill_csv() in create_pir_csv.py
        data = pd.read_csv(f"gait_values_{sensors_height}.csv")
        # From the PIR values, apply a wavelet transform using Morlet waves and save them in a binary file
        precompute_morlet_transforms(data,
                                     pir_transforms_filename=pir_transforms_filename,
                                     output_filename=pir_transforms_filename)

    # If --train is given, we engage the training of our models
    if do_training:
        # If we want all walking parameters to be trained, we loop infinitely to train each parameter
        if estimated_parameter == 'all':
            while True:
                for param in estimable_parameters:
                    root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{param}/"
                    path_to_save_cnn = root_to_cnn + "convolution_network.model"
                    training(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                             estimated_parameter=param, number_of_models_trained=number_of_models_per_parameter)
                number_of_models_per_parameter += 5
        # Otherwise, train your models on the desired walking parameter
        else:
            training(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                     estimated_parameter=estimated_parameter)

    # If --display is given, display the learning curves, the predictions and the errors of the models
    if do_display:
        # If we want to see all walking parameters, loop on each of them to do the display
        if estimated_parameter == 'all':
            for param in estimable_parameters:
                root_to_cnn = f"trained_models/{device.type}_batch_normed_{sensors_height}_{param}/"
                path_to_save_cnn = root_to_cnn + "convolution_network.model"
                display(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                        display_all_models=display_all_models)
        # Otherwise just display the models trained on the desired one
        else:
            display(path_to_save_cnn, pir_transforms_filename=pir_transforms_filename,
                    display_all_models=display_all_models)

    # If --merge-file is given, do a clean and a merge of your models
    if merge_files:
        merge_models_in_one_file(path_to_save_cnn)

    # If --clean-files is given, do a clean on your models
    if clean_files:
        remove_useless_models(path_to_save_cnn)


if __name__ == '__main__':
    main()

