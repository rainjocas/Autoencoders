"""
The task we’ll explore today is that of Denoising (note that we covered the UNet
for segmentation in class, which means that things will be slightly different for
this UNet implementation) We’ll use the MNIST dataset for this purpose. In this
problem, you’ll artificially introduce noise to the data during training and then
you should try to predict the clean image using the noisy data as the input. In
PyTorch, you can introduce noise to the images by adding matrices whose values
are normally distributed to them (you can use randn() or randn_like() Pytorch
functions for that). As you add the noise, make sure to consider how much of that
noise should be in the image. For example, if I is the image (whose values should
be normalized to be in [0, 1]) and N is the noise matrix, the noisy image Y
should be Y = (1 - mu)*I + mu*N, where mu is a value in [0, 1]. After you add the
noise, make sure that the final image has its values in [0, 1]. Try to add this
whole procedure in a Pytorch Dataset class.

"""
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#import MNIST dataset

data_folder = '~/data/MNIST'
mnist_train = datasets.MNIST(data_folder, download=True, train=True)
x_train, y_train = mnist_train.data, mnist_train.targets


# if I is the image (whose values should be normalized to be in [0, 1]) and N is
# the noise matrix, the noisy image Y should be Y = (1 - mu)*I + mu*N, where mu
# is a value in [0, 1]. After you add the noise, make sure that the final image
# has its values in [0, 1]. Try to add this whole procedure in a Pytorch Dataset
# class.
#NOTE: HELP HOW DO I ADD NOISE

class MNISTDataset(Dataset):
    """
    Creates a dataset that processes the data and adds noise
    """
    def __init__(self, x, y):
        x = x.float()/255 # Data rescaling
        x = x.view(-1,28*28) # Data reshaping
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        #x = 1 - x * img + img + N
        return x.to(device), y.to(device)

def loadMNISTData():
    """
    Loads the MNIST training and testing data
    """
    mnist_train = datasets.MNIST('~/data/MNIST', download=True, train=True)
    mnist_test = datasets.MNIST('~/data/MNIST', download=True, train=False)
    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    return mnist_train, mnist_test, x_train, y_train, x_test, y_test

def defineDataLoaders(x_train, y_train, x_test, y_test):
    train_dataset = MNISTDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = MNISTDataset(x_test, y_test)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataset, train_dl, test_dataset, test_dl


#Loads the MNIST training and testing data
mnist_train, mnist_test, x_train, y_train, x_test, y_test = loadMNISTData()

#Define the Dataloaders, that coordinate how the data will be read
train_dataset, train_dl, test_dataset, test_dl = defineDataLoaders(x_train, y_train, x_test, y_test)

#Visualize the dataset
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)