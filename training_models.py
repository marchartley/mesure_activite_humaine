import torch
from torch import nn
from typing import Union, List, Tuple
from collections import Iterable


class CustomModel(nn.Module):
    """! Main class of Neural networks trained here"""
    def __init__(self):
        """! Constructor"""
        super().__init__()
        ## Specify if this model can be used on the GPU or need the CPU
        self.use_cuda = True
        ## Specify if we have one neural module for each sensor or if we merge them together
        self.use_concat_pir = False
        ## Number of values the model has to predict. It's way easier for now to predict one single value at a time.
        self.number_of_outputs = 1

    def update_device(self):
        from deep_neural_training import cuda, cpu
        """! Update the device to move from the CPU to GPU or from GPU to CPU"""
        global device, use_cuda
        if 'use_cuda' not in self.__dict__:
            self.use_cuda = True
        use_cuda = self.use_cuda
        if self.use_cuda:
            device = cuda
        else:
            device = cpu
        # Move our model to ve desired memory part
        self.to(device)

    def __str__(self):
        """! The "name" of the networks briefly prints all its parameters"""
        var_name_length = 25
        out = f"{self.__class__.__name__}:" + "\n\t"
        out += "\n\t".join([
            f"{str(var_name).replace('_', ' ')}{' ' * (var_name_length - len(str(var_name)))} = {getattr(self, var_name)}"
            for var_name in self.__init__.__code__.co_varnames[1:] if var_name in self.__dict__])
        return out

    def __repr__(self):
        return str(self)


class PretrainedModel(CustomModel):
    """! Class used when training predefined neural networks like AlexNet, DenseNet, etc..."""
    def __init__(self, trained_model,
                 pretrained: bool = False,
                 use_exponential: Union[bool, str] = False,
                 number_of_outputs: int = 1,
                 hidden_cells: Union[List[int], int] = [],
                 dropout: bool = False,
                 lastActivation: nn.Module = None,
                 **useless):
        """! Constructor
        @param trained_model PyTorch model used to build this network. List of them can be found in the dictionary
        "list_of_predefined_models" defined in "deep_neural_training.py"
        @param pretrained Use the weights from the pretrained model or start the training from scratch
        @param use_exponential Apply an exponential activation at the end of each prediction.
        Can be "True", "False" or "pow" to apply a squaring computation.
        @param number_of_outputs The number of prediction the model has to output
        @param useless Garbage variable for any non-important parameters
        """
        super().__init__()
        from deep_neural_training import all_pretrained_models
        ## The number of prediction the model has to output
        self.number_of_outputs = number_of_outputs
        if not isinstance(use_exponential, Iterable):
            use_exponential = [use_exponential] * number_of_outputs
        ## Apply an exponential activation at the end of each prediction
        self.use_exponential = use_exponential
        ## Use the weights from the pretrained model or start the training from scratch
        self.pretrained = pretrained
        ## Apply a dropout on the linear layers
        self.dropout = dropout
        ## Activation function at the end of the network
        self.lastActivation = lastActivation

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]
        ## Structure of the hidden layers in the linear part of the model
        self.hidden_cells = hidden_cells

        try:
            ## The main part of the network, designed from the PyTorch model
            self.module = trained_model(pretrained=self.pretrained, init_weights=True)
        except:
            self.module = trained_model(pretrained=self.pretrained)
        for param in self.module.parameters():
            param.requires_grad = not self.pretrained

        ## Name of the model used
        self.trained_model = self.module._get_name()

        ## Number of features at the end of the predefined model.
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

        ## First convolution used for resizing the input to fit in the predefined module
        self.first_conv = nn.Conv2d(4, 3, kernel_size=(3, 3), padding=1)
        ## Head of the model that flatten the last convolutions and squeeze all in one neuron as output.
        # self.endings = nn.ModuleList([nn.Linear(self.num_ftrs, 1) for _ in range(number_of_outputs)])

        all_endings = []
        for _ in range(number_of_outputs):
            linear = []
            # If an hidden layer is asked, create them
            for i_hidden in range(len(hidden_cells)):
                if i_hidden == 0:
                    linear.append(nn.Linear(self.num_ftrs, hidden_cells[i_hidden]))
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
                linear.append(nn.Linear(hidden_cells[-1], number_of_outputs))
            else:
                linear.append(nn.Linear(self.num_ftrs, number_of_outputs))

            # Apply a final activation function on the output.
            if lastActivation:
                linear.append(lastActivation)
            # self.linear_layers = nn.Sequential(*linear)
            all_endings.append(nn.Sequential(*linear))

        ## Head of the model that flatten the last convolutions and squeeze all in one neuron as output.
        self.endings = nn.ModuleList(all_endings)

        self.use_concat_pir = True

        self.update_device()

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        """! Prediction process"""
        x = self.first_conv(concat_pir)
        x = self.module(x)

        final_tensor = [2 ** self.endings[i](x) if self.use_exponential[i] else self.endings[i](x) for i in
                        range(self.number_of_outputs)]
        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor


class MultipleWaveletsModule(CustomModel):
    """! Class that use convolutions and poolings"""
    def __init__(self, input_size: int = 512, mid_output_size: int = 10, kernel_size: Union[int, Tuple[int, int]] = 3,
                 number_of_filters: int = 2, number_of_layers: int = 8, hidden_cells: Union[int, List[int]]=[],
                 dropout: bool = False, use_concat_pir: bool = True, number_of_conv_per_layer: int = 1,
                 pure_CNN: bool = False, layer_relu: nn.Module = None, reduce_last_conv: int = 0,
                 use_exponential: Union[bool, str] = False, number_of_outputs: int = 1,
                 using_batch_norm: bool = True,
                 **useless):
        """! Constructor
        @param input_size Width (and height) of the images to process
        @param mid_output_size Number of neurons at the beginning of the linear part of the model
        """
        super().__init__()
        ## Width (and height) of the images to process
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
        if lastActivation:
            linear.append(lastActivation)
        self.linear_layers = nn.Sequential(*linear)

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
