import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_classes import Localization, Classifier

afn = nn.ReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


### LOCALIZATION ###

class Small_localization(Localization):
    default_parameters = [32]

    def init_model(self, in_shape):
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(in_shape), self.param[0]),
            afn()
        )


class FCN_localization(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        self.l1 = nn.Linear(np.prod(in_shape), self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])

    def model(self, x):
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

class FCN_localization_batchnorm(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.l1 = nn.Linear(np.prod(in_shape), self.param[0])
        self.b1 = nn.BatchNorm1d(self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = self.b1(F.relu(self.l1(x.view(x.size(0), -1))))
        x = self.b2(F.relu(self.l2(x)))
        x = self.b3(F.relu(self.l3(x)))
        return x

class FCN_localization_maxpool(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1 = nn.BatchNorm1d(np.prod(in_shape)//4)
        self.l1 = nn.Linear(np.prod(in_shape)//4, self.param[0])
        self.b2 = nn.BatchNorm1d(self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.b3 = nn.BatchNorm1d(self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])
        self.b4 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = self.mp(x)
        x = F.relu(self.l1(self.b1(x.view(x.size(0), -1))))
        x = F.relu(self.l2(self.b2(x)))
        x = F.relu(self.l3(self.b3(x)))
        return self.b4(x)

class FCN_maxpool_nobatchnorm(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l1 = nn.Linear(np.prod(in_shape)//4, self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])

    def model(self, x):
        x = self.mp(x)
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

class CNNFCN_localization(Localization):
    '''MNIST localization consisting of jaderberg's initial convolution
    followed by the three layers from the FCN localization
    '''

    default_parameters = [64, 32, 32, 32]

    def init_model(self, in_shape):
        self.c = nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9))
        self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2)
        flattened = self.param[0] * ((in_shape[1] - 8)//2)**2
        self.l1 = nn.Linear(flattened, self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.l3 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = F.relu(self.mp(self.c(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

class CNNFCN_batchnorm(Localization):
    '''MNIST localization consisting of jaderberg's initial convolution
    followed by the three layers from the FCN localization, including the
    extra maxpool used in FCNmp.
    '''

    default_parameters = [64, 32, 32, 32]

    def init_model(self, in_shape):
        self.c = nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9))
        self.mp = nn.MaxPool2d(kernel_size = 4, stride = 4)
        flattened = self.param[0] * ((in_shape[1] - 8)//4)**2
        self.b1 = nn.BatchNorm1d(flattened)
        self.l1 = nn.Linear(flattened, self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])
        self.l3 = nn.Linear(self.param[2], self.param[3])
        self.b4 = nn.BatchNorm1d(self.param[3])

    def model(self, x):
        x = F.relu(self.mp(self.c(x)))
        x = F.relu(self.l1(self.b1(x.view(x.size(0), -1))))
        x = F.relu(self.l2(self.b2(x)))
        x = F.relu(self.l3(self.b3(x)))
        return self.b4(x)

class CNNFCN_maxpool(Localization):
    '''MNIST localization consisting of jaderberg's initial convolution
    followed by the three layers from the FCN localization, including the
    extra maxpool used in FCNmp.
    '''

    default_parameters = [64, 32, 32, 32]

    def init_model(self, in_shape):
        self.c = nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9))
        self.mp = nn.MaxPool2d(kernel_size = 4, stride = 4)
        flattened = self.param[0] * ((in_shape[1] - 8)//4)**2
        self.l1 = nn.Linear(flattened, self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.l3 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = F.relu(self.mp(self.c(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class CNN_localization(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        side = int((in_shape[-1]/2 - 4)/2 - 4)
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x

class CNN_mp(Localization):
    default_parameters = [20,20,20]
    
    def init_model(self, in_shape):
        # maybe interpolation
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        # second mp
        side = in_shape[-1]//2 if in_shape[-1] == 112 else in_shape[-1]
        side = ((side - 4)//2 - 4)//2
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        if x.size(-1) == 112:  # either 112 or 52
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.mp(F.relu(self.c1(x)))
        x = self.mp(F.relu(self.c2(x)))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x

class CNNCNN_mp(Localization):
    default_parameters = [64,20,20,20]
    
    def init_model(self, in_shape):
        self.c0 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(9,9))
        self.c1 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[1], self.param[2], kernel_size=(5,5))
        # second mp
        side = (((in_shape[-1] - 8)//2 - 4)//2 - 4)//2
        self.l = nn.Linear(self.param[2] * side**2, self.param[3])

    def model(self, x):
        x = self.mp(F.relu(self.c0(x)))
        x = self.mp(F.relu(self.c1(x)))
        x = self.mp(F.relu(self.c2(x)))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x


class CNN_localization_batchnorm(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1 = nn.BatchNorm2d(self.param[0])
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        flattened = self.param[1] * int((in_shape[-1]/2 - 4)/2 - 4)**2
        self.b2 = nn.BatchNorm1d(flattened)
        self.l = nn.Linear(flattened, self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.b1(self.mp(F.relu(self.c1(x))))
        x = self.b2(F.relu(self.c2(x)))
        x = self.b3(F.relu(self.l(x.view(x.size(0), -1))))
        return x

class CNN_middleloc_batchnorm(Localization):
    default_parameters = [24,48,48]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        flattened = self.param[0] * (3 if in_shape[-1] == 10 else 5)**2
        self.b1 = nn.BatchNorm1d(flattened)
        self.l1 = nn.Linear(flattened, self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        if x.size(-1) == 28:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.b1(self.mp(F.relu(self.c1(x))))
        x = self.b2(F.relu(self.l1(x.view(x.size(0), -1))))
        x = self.b3(F.relu(self.l2(x)))
        return x

class CNN_middleloc(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        if in_shape[-1] > 24:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        side = 8 if in_shape[1] == 28 else 2
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = self.c1(x)
        if x.size(-1) > 20:
            x = self.mp(x)
        x = F.relu(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x

class CNN_translate(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        side = (in_shape[-1]-4)//2 - 4
        side = side//2 if side > 20 else side
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        if x.size(-1) > 20: # the size should be 7 or 24
            x = self.mp(x)
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x

class CNNCNN(Localization):
    default_parameters = [64,20,20,20]

    def init_model(self, in_shape):
        self.c0 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(9,9))
        self.c1 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[1], self.param[2], kernel_size=(5,5))
        side = int(((in_shape[-1] - 8)/2 - 4)/2 - 4)
        self.l = nn.Linear(self.param[2] * side**2, self.param[3])

    def model(self, x):
        x = self.mp(F.relu(self.c0(x)))
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x


### CLASSIFICATION ###

class FCN(Classifier):
    default_parameters = [256,200]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape):
        return nn.ModuleList([
            Flatten(),
            nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            nn.Linear(self.param[0], self.param[1]),
            afn(),
        ])

class CNN(Classifier): # original for mnist, works for cifar
    default_parameters = [64,64]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape):
        return nn.ModuleList([
            nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
        ])

class CNN_batchnorm(Classifier):
    default_parameters = [64,64]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape):
        # Changed to put batchnorm after each layer instead of before
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                afn(),
                nn.BatchNorm2d(self.param[0]),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                afn(),
                nn.BatchNorm2d(self.param[1]),
            ),
        ])
