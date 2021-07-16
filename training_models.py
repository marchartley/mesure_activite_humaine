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
        """! Update the device to move from the CPU to GPU or from GPU to CPU"""
        # from deep_neural_training import cuda, cpu, device
        from deep_neural_training import update_device
        # global device, use_cuda
        if 'use_cuda' not in self.__dict__:
            self.use_cuda = True
        # use_cuda = self.use_cuda
        # if self.use_cuda:
        #     device = cuda
        # else:
        #     device = cpu
        # Move our model to ve desired memory part
        self.to(update_device(self.use_cuda))

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

        # In the next part, we will just modify the head of the predefined models to fit our regression problem
        if trained_model is all_pretrained_models['resnet18'] or trained_model is all_pretrained_models['shufflenet']:
            self.num_ftrs = self.module.fc.in_features
            self.module.fc = nn.Identity()  # nn.Linear(self.num_ftrs, 1)
        elif trained_model is all_pretrained_models['alexnet'] \
                or trained_model is all_pretrained_models['mobilenet_v2'] \
                or trained_model is all_pretrained_models['mobilenet_v3_small'] \
                or trained_model is all_pretrained_models['mobilenet_v3_large'] \
                or trained_model is all_pretrained_models['mnasnet']:
            self.num_ftrs = self.module.classifier[-1].in_features
            self.module.classifier[-1] = nn.Identity()
        elif trained_model is all_pretrained_models['squeezenet']:
            self.num_ftrs = 13 * 13
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
        # Call the first convolution to fit the CNN requirements (3 channels on an image)
        x = self.first_conv(concat_pir)
        # Call the CNN process
        x = self.module(x)

        # Apply the ending process for each of the possible output if you asked for multiple ones
        final_tensor = [2 ** self.endings[i](x) if self.use_exponential[i] else self.endings[i](x) for i in
                        range(self.number_of_outputs)]

        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor


class MultipleWaveletsModule(CustomModel):
    """! Class that use convolutions and poolings"""

    def __init__(self, input_size: int = 512, mid_output_size: int = 10, kernel_size: Union[int, Tuple[int, int]] = 3,
                 number_of_filters: int = 2, number_of_layers: int = 8, hidden_cells: Union[int, List[int]] = [],
                 dropout: bool = False, use_concat_pir: bool = True, number_of_conv_per_layer: int = 1,
                 pure_CNN: bool = False, layer_relu: nn.Module = None, reduce_last_conv: int = 0,
                 use_exponential: Union[bool, str] = False, number_of_outputs: int = 1,
                 using_batch_norm: bool = True,
                 **useless):
        """! Constructor
        @param input_size Width (and height) of the images to process
        @param mid_output_size Number of neurons at the beginning of the linear part of the model
        @param kernel_size The size of the kernel used during the convolutions
        @param number_of_filters The number of filters at the end of a convolution. In our program this number doesn't
        vary after each convolution, but you could program to do so if you have some patience.
        @param number_of_layers Number of times we do a Convolution+Max pooling in the network
        @param hidden_cells The number of hidden neurons on the dense part of the network. If integer, it makes a
        unique hidden layer with this number of neurons but if you pass an array of integers, it will make multiple
        hidden layers with the desired number of neurons
        @param dropout Defines if a dropout is applied at the beginning of the dense part of the network. If True,
        it will apply a 0.5 dropout but you can pass a float instead to define yourself the desired value.
        If you want to take the time to put dropouts somewhere else, feel free to program it
        @param use_concat_pir Defines if you want to have one tensor containing the 4 sensors' data or whever you
        prefer having one unique process for each sensor. (I think it doesn't change much, maybe keep it to True)
        @param number_of_conv_per_layer Number of convolution done before each Max Pooling
        @param pure_CNN Defines if you prefer a classic CNN with a Dense Layer using each pixel of each filter at the
        end of the convolutional part as neurons, or if you want to apply a Global Average pooling to reduce greatly
        the size of the dense layers.
        (without "pure_CNN", the number of neurons is (input_size / 2^number_of_layers) * number_of_filters, while
        with "pure_CNN", the number of neurons is equal to number_of_filters.)
        With a bit more of programming, you can increase the number of filters during the process to improve
        the network.
        @param layer_relu Activation function used after each convolution layer.
        @param reduce_last_conv If greater than 0, apply a last convolution with "reduce_last_conv" filters to reduce
        the number of neurons in the dense part.
        @param use_exponential The artificial activation function. Values can be :
        - True for using exponential
        - False for not using anything
        - "pow" to apply a squaring operation
        But this parameter should stay to False for a real explainable model
        @param number_of_outputs Number of desired outputs. Can be set to >1 for doing multi-task, but that should
        require some changes in the training and displaying functions.
        @param using_batch_norm Set to True if you want a Batch Normalization before each convolution.

        @return An instance of MultipleWaveletModule
        """
        super().__init__()
        ## Width (and height) of the images to process
        self.input_size = input_size
        ## Number of neurons at the beginning of the linear part of the model
        self.mid_output_size = mid_output_size
        ## The size of the kernel used during the convolutions
        self.kernel_size = kernel_size
        ## The number of filters at the end of a convolution. In our program this number doesn't
        # vary after each convolution, but you could program to do so if you have some patience.
        self.number_of_filters = number_of_filters
        ## Number of times we do a Convolution+Max pooling in the network
        self.number_of_layers = number_of_layers
        ## The number of hidden neurons on the dense part of the network
        self.hidden_cells = hidden_cells
        ## Defines if a dropout is applied at the beginning of the dense part of the network
        self.dropout = dropout
        ## Defines if you want to have one tensor containing the 4 sensors' data or whever you
        # prefer having one unique process for each sensor
        self.use_concat_pir = use_concat_pir
        ## Number of convolution done before each Max Pooling
        self.number_of_conv_per_layer = number_of_conv_per_layer
        ## Activation function used after each convolution layer
        self.layer_relu = layer_relu
        ## Set to True if you want a Batch Normalization before each convolution
        self.using_batch_norm = using_batch_norm
        ## Defines if you prefer a classic CNN with a Dense Layer using each pixel of each filter at the
        # end of the convolutional part as neurons, or if you want to apply a Global Average pooling to reduce greatly
        # the size of the dense layers.
        self.pure_CNN = pure_CNN
        ## If greater than 0, apply a last convolution with "reduce_last_conv" filters to reduce
        # the number of neurons in the dense part.
        self.reduce_last_conv = reduce_last_conv
        ## Number of desired outputs
        self.number_of_outputs = number_of_outputs
        if not isinstance(use_exponential, Iterable):
            use_exponential = [use_exponential] * number_of_outputs
        ## The artificial activation function
        self.use_exponential = use_exponential

        ## Create one module per sensor signal (used when not using the concat_pir version)
        self.multiple_wavelet_modules = nn.ModuleList([WaveletModule(input_size=input_size, output_size=mid_output_size,
                                                                     kernel_size=kernel_size,
                                                                     number_of_filters=number_of_filters,
                                                                     number_of_layers=number_of_layers,
                                                                     hidden_cells=hidden_cells,
                                                                     dropout=dropout,
                                                                     number_of_conv_per_layer=number_of_conv_per_layer,
                                                                     use_concat=False,
                                                                     pure_CNN=pure_CNN,
                                                                     reduce_last_conv=reduce_last_conv,
                                                                     layer_relu=layer_relu,
                                                                     using_batch_norm=self.using_batch_norm,
                                                                     lastActivation=None) for _ in
                                                       range(4)])
        ## Create one unique module that take the 4 PIR data at once
        self.unique_wavelet_module = WaveletModule(input_size=input_size, output_size=4 * mid_output_size,
                                                   kernel_size=kernel_size,
                                                   number_of_filters=number_of_filters,
                                                   number_of_layers=number_of_layers,
                                                   hidden_cells=hidden_cells,
                                                   dropout=dropout,
                                                   number_of_conv_per_layer=number_of_conv_per_layer,
                                                   use_concat=True,
                                                   pure_CNN=pure_CNN,
                                                   reduce_last_conv=reduce_last_conv,
                                                   layer_relu=layer_relu,
                                                   lastActivation=None)

        ## In any case, the activation is a Dense of shape (4*output x 1)
        self.activation_functions = nn.ModuleList([nn.Linear(4 * mid_output_size, 1) for _ in range(number_of_outputs)])

        self.update_device()

    def forward(self, pir0, pir1, pir2, pir3, concat_pir):
        if self.use_concat_pir:
            # If using concat_pir, apply the CNN on the data
            x = self.unique_wavelet_module(concat_pir)
        else:
            # Otherwise, apply the CNN to each sensor individually
            pirs = [pir0, pir1, pir2, pir3]
            for i in range(4):
                pirs[i] = self.multiple_wavelet_modules[i](pirs[i])
            # Concat all results as one to flatten it after
            x = torch.cat(tuple([tens for tens in pirs]), 1)

        # Apply the artificial activation on each output if desired
        final_tensor = [
            self.activation_functions[i](x) ** 2 if self.use_exponential == 'pow' else \
                2 ** self.activation_functions[i](x) if self.use_exponential[i] else self.activation_functions[i](x)
            for i in range(self.number_of_outputs)]

        if len(final_tensor) == 1:
            return final_tensor[0]
        else:
            return final_tensor


class WaveletModule(nn.Module):
    """! Submodule used in the MultipleWaveletModule. This neural network treats a unique sensor's signal"""

    def __init__(self, input_size: int, output_size: int, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1, use_concat=True,
                 hidden_cells=None, dropout: Union[float, bool] = False, number_of_conv_per_layer=1,
                 layer_relu=None, using_batch_norm=True, pure_CNN=False,
                 lastActivation=None, reduce_last_conv=0, **useless):
        """! Constructor
        @param input_size Width (and height) of the images to process
        @param output_size Number of neurons at the end of the linear part of the model
        @param kernel_size The size of the kernel used during the convolutions
        @param number_of_filters The number of filters at the end of a convolution. In our program this number doesn't
        vary after each convolution, but you could program to do so if you have some patience.
        @param number_of_layers Number of times we do a Convolution+Max pooling in the network
        @param hidden_cells The number of hidden neurons on the dense part of the network. If integer, it makes a
        unique hidden layer with this number of neurons but if you pass an array of integers, it will make multiple
        hidden layers with the desired number of neurons
        @param dropout Defines if a dropout is applied at the beginning of the dense part of the network. If True,
        it will apply a 0.5 dropout but you can pass a float instead to define yourself the desired value.
        If you want to take the time to put dropouts somewhere else, feel free to program it
        @param use_concat_pir Defines if you want to have one tensor containing the 4 sensors' data or whever you
        prefer having one unique process for each sensor. (I think it doesn't change much, maybe keep it to True)
        @param number_of_conv_per_layer Number of convolution done before each Max Pooling
        @param pure_CNN Defines if you prefer a classic CNN with a Dense Layer using each pixel of each filter at the
        end of the convolutional part as neurons, or if you want to apply a Global Average pooling to reduce greatly
        the size of the dense layers.
        (without "pure_CNN", the number of neurons is (input_size / 2^number_of_layers) * number_of_filters, while
        with "pure_CNN", the number of neurons is equal to number_of_filters.)
        With a bit more of programming, you can increase the number of filters during the process to improve
        the network.
        @param layer_relu Activation function used after each convolution layer.
        @param reduce_last_conv If greater than 0, apply a last convolution with "reduce_last_conv" filters to reduce
        the number of neurons in the dense part.
        @param using_batch_norm Set to True if you want a Batch Normalization before each convolution.

        @return An instance of WaveletModule
        """
        super().__init__()

        if lastActivation is None:
            lastActivation = layer_relu

        ## Width (and height) of the images to process
        self.input_size = input_size
        ## Number of neurons at the end of the linear part of the model
        self.output_size = output_size
        ## The size of the kernel used during the convolutions
        self.kernel_size = kernel_size
        ## The number of filters at the end of a convolution
        self.number_of_filters = number_of_filters
        ## Number of times we do a Convolution+Max pooling in the network
        self.number_of_layers = number_of_layers
        ## The number of hidden neurons on the dense part of the network
        self.hidden_cells = hidden_cells
        ## Defines if a dropout is applied at the beginning of the dense part of the network
        self.dropout = dropout
        ## Defines if you want to have one tensor containing the 4 sensors' data or whever you
        # prefer having one unique process for each sensor
        self.use_concat = use_concat
        ## Number of convolution done before each Max Pooling
        self.number_of_conv_per_layer = number_of_conv_per_layer
        ## If greater than 0, apply a last convolution with "reduce_last_conv" filters to reduce
        # the number of neurons in the dense part.
        self.reduce_last_conv = reduce_last_conv
        ## Activation function used after each convolution layer
        self.layer_relu = layer_relu
        ## Set to True if you want a Batch Normalization before each convolution
        self.using_batch_norm = using_batch_norm
        ## Defines if you prefer a classic CNN with a Dense Layer using each pixel of each filter at the
        # end of the convolutional part as neurons, or if you want to apply a Global Average pooling to reduce greatly
        # the size of the dense layers.
        self.pure_CNN = pure_CNN

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]

        # If we want a binary output, we use 2 outputs and apply a Softmax function at the end
        is_binary = (output_size == 1)
        if output_size == 1:
            output_size = 2

        # The padding added at each convolution to keep the same input size
        padding = int(kernel_size / 2)
        # First, a convolution. Here we specified an image with a unique channel (no RGB)
        elems = [nn.Conv2d(4 if use_concat else 1, number_of_filters, kernel_size, padding=padding)]

        # For every new "layer" (or block), we add a MaxPooling and a Convolution
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

        # If we want "pure_CNN", we apply an average pooling on all the filters
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
        # Apply the convolutional part of the network
        x = self.cnn_layers(x)
        # Flatten the data to have one single row of neurons
        x = x.view(x.size(0), -1)
        # Apply the linear part of the model
        x = self.linear_layers(x)

        return x


class MultipleWaveletsLinearModule(CustomModel):
    """! A false convolutional neural network that takes each pixels of the sensors' signals directly into
    a Dense layer.
    @note Note
    This is absolutely not a good network in this case because it is not invariant in space and time at all.
    I'm not even going to comment this network, just avoid it...
    """
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
    """! Module used in the MultipleLinearWaveletModule.
    @note Note
    This is absolutely not a good network in this case because it is not invariant in space and time at all.
    I'm not even going to comment this network, just avoid it...
    """
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
    """! The ResNet adaptation

    This networks apply convolutions on the images but associate them with parts of the image without transformations.
    For more detail about the ResNet process, see papers about it / google it.
    """
    def __init__(self, input_size: int, output_size: int = 1, number_of_outputs: int = 1, kernel_size: int = 3,
                 number_of_filters: int = 2, number_of_layers: int = 1,
                 hidden_cells: Union[int, List[int]] = None, number_of_conv_per_layer: int = 1, layer_relu=None,
                 lastActivation=None, use_concat_pir: bool = True, **useless):
        """! Constructor
        @note Remark:
        The names of some attributes are not very explicit. I tried to use variable names that are used in other models
        because that reduces the size of the "training hyperparameters" from the training function in
        "deep_neural_training.py"

        @see deep_neural_training::training()

        @param input_size The width (and height) of the input images
        @param output_size The number of outputs desired. Can be useful for multi-task, but needs some adaptations in
        the code
        @param number_of_outputs Useless
        @param kernel_size The size of the kernels used in the convolutions
        @param number_of_filters Number of filters created by the convolutions
        @param number_of_layers Number of residual blocks that will be created.
        @param hidden_cells Number of hidden neurons to create in the dense part. If an integer is provided, a unique
        hidden layer is created, if an array is provided there will be multiple hidden layers.
        @param number_of_conv_per_layer Number of convolutions applied during each residual block
        @param layer_relu The activation function used before each convolution
        @param lastActivation The activation function applied at the end of the network
        @param use_concat_pir Defines if we are using one input containing the 4 sensors data or if we do 4 processes
        individually. Better to keep it to True, I guess.
        """
        super().__init__()
        ## Number of residual blocks that will be created
        self.number_of_layers = number_of_layers
        ## The size of the kernels used in the convolutions
        self.kernel_size = kernel_size
        ## Number of filters created by the convolutions
        self.number_of_filters = number_of_filters
        ## Number of hidden neurons to create in the dense part
        self.hidden_cells = hidden_cells
        ## Number of convolutions applied during each residual block
        self.number_of_conv_per_layer = number_of_conv_per_layer
        ## The activation function used before each convolution
        self.layer_relu = layer_relu
        ## Defines if we are using one input containing the 4 sensors data or if we do 4 processes
        # individually. Better to keep it to True, I guess.
        self.use_concat_pir = use_concat_pir

        if hidden_cells is None:
            hidden_cells = []
        if isinstance(hidden_cells, int):
            hidden_cells = [hidden_cells]

        padding = int(kernel_size / 2)
        # First, a convolution. Here we specified an image with a unique channel (no RGB)
        ## The initial convolution
        self.init_conv = nn.Conv2d(4 if self.use_concat_pir else 1, number_of_filters, kernel_size, padding=padding)

        residual_blocks = []
        for i_block in range(number_of_layers):
            residual_blocks.append([])
            for _ in range(number_of_conv_per_layer):
                residual_blocks[i_block].append(nn.BatchNorm2d(number_of_filters))
                if layer_relu:
                    residual_blocks[i_block].append(layer_relu)
                residual_blocks[i_block].append(
                    nn.Conv2d(number_of_filters, number_of_filters, kernel_size, padding=padding))
            # residual_blocks[i_block].append(nn.MaxPool2d(2))
            residual_blocks[i_block] = nn.Sequential(*residual_blocks[i_block])
            input_size = input_size / 2
        ## List of residual blocks being used
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
        ## Linear part of the network
        self.linear_layers = nn.Sequential(*linear)

        self.update_device()

    def forward_process(self, x: torch.Tensor):
        """! The forward pass that applies the residual blocks"""
        # Apply the first convolution to the tensor
        x = self.init_conv(x)
        # Repeat the process of residual blocks
        for i in range(len(self.residual_blocks)):
            # Keep the original tensor
            original_x = torch.clone(x)
            # Add the result of a convolution on the original tensor
            x = original_x + self.residual_blocks[i](original_x)
            # Apply a max pooling
            x = nn.MaxPool2d(2)(x)

        # Flatten the neurons to get one row
        x = x.view(x.size(0), -1)
        # Apply the linear/dense layer
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
