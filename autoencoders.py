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
print(device)

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
    def __init__(self, x, y, mu):
        x = x.float()/255 #rescale because its an image to make values between [0,1]
        x = x.unsqueeze(1)
        N = torch.randn_like(x)
        self.x, self.y, self.mu, self.N = x, y, mu, N
    def __len__(self):
        return len(self.x)
    def __getitem__(self, ix):
        x_clean, mu, N = self.x[ix], self.mu, self.N[ix]
        x_noisy = ((1- mu) * x_clean) + (mu * N)
        return x_noisy.to(device), x_clean.to(device)

def loadMNISTData():
    """
    Loads the MNIST training and testing data
    """
    mnist_train = datasets.MNIST('~/data/MNIST', download=True, train=True)
    mnist_test = datasets.MNIST('~/data/MNIST', download=True, train=False)
    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    return mnist_train, mnist_test, x_train, y_train, x_test, y_test

def defineDataLoaders(x_train, y_train, x_test, y_test, mu):
    train_dataset = MNISTDataset(x_train, y_train, mu)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = MNISTDataset(x_test, y_test, mu)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataset, train_dl, test_dataset, test_dl

class U_NetModel(nn.Module):
    """
    NOTE: ADD DOCUMENTATION
    """
    def __init__(self):
        super().__init__()
        self.downsample_one = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.downsample_two = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.downsample_three = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2))

        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)

        self.upsample_one = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.upsample_two = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        
        self.conv_Transpose_one = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),)
        self.conv_Transpose_two = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),)
        
        self.conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.downsample_one(x)
        x2 = self.max_pool(x1)
        x3 = self.downsample_two(x2)
        x4 = self.max_pool(x3)
        x5 = self.downsample_three(x4)

        x6 = self.middle(x5)

        x7 = self.conv_Transpose_one(x6)
        x8 = self.upsample_one(torch.cat((x7, x3), dim=1))
        x9 = self.conv_Transpose_two(x8)
        x10 = self.upsample_two(torch.cat((x9, x1), dim=1))
        x11 = self.conv(x10)

        return x11

class AutoencoderModel(nn.Module):
    """
    NOTE: ADD DOCUMENTATION
    """
    def __init__(self):
        super().__init__()
        self.downsample_one = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.downsample_two = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.downsample_three = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2))

        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)

        self.upsample_one = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        self.upsample_two = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = (3,3), padding = 1),
            nn.ReLU(),)
        
        self.conv_Transpose_one = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),)
        self.conv_Transpose_two = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),)
        
        self.conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.downsample_one(x)
        x = self.max_pool(x)
        x = self.downsample_two(x)
        x = self.max_pool(x)
        x = self.downsample_three(x)

        x = self.middle(x)

        x = self.conv_Transpose_one(x)
        x = self.upsample_one(x)
        x = self.conv_Transpose_two(x)
        x = self.upsample_two(x)
        x = self.conv(x)

        return x

@torch.no_grad()
def accuracy(x, y, model):
    """
    Calculates a model's accuracy (Taken from Slides)
    """
    model.eval()
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float())/len(y)
    return s.cpu().numpy()

def train_batch(x, y, model, opt, loss_fn):
    """
    Trains a batch (Taken from Slides and edited)
    """
    model.train()
    opt.zero_grad() # Flush memory
    batch_loss = loss_fn(model(x), y) # Compute loss
    batch_loss.backward() # Compute gradients
    opt.step() # Make a GD step
    return batch_loss.detach().cpu().numpy() # Removes grad, sends data to mps, converts tensor to array

def train_model(model, opt, loss_fn, train_dl):
    """
    Trains a model for 10 epochs
    """
    train_losses, train_accuracies, n_epochs = [], [], 1 #NOTE: change back to 10 batches
    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")
        train_epoch_losses, train_epoch_accuracies = [], []
        train_epoch_losses = [train_batch(x, y, model, opt, loss_fn) for x, y in train_dl]
        train_epoch_loss = np.mean(train_epoch_losses)
        train_epoch_accuracies = [accuracy(x, y, model) for x, y in train_dl]
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
    return train_losses, train_accuracies

def run_U_Net(mu):
    """
    Preprocess data, and runs and returns metrics of the VGG16 TL model
    """
    print("ELEPHANT")

    #Loading and data pre-processing
    mnist_train, mnist_test, x_train, y_train, x_test, y_test = loadMNISTData()
    train_dataset, train_dl, test_dataset, test_dl = defineDataLoaders(x_train, y_train, x_test, y_test, mu)

    #Visualize the dataset
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train[0])

    #get model
    model = U_NetModel().to(device)

    #model.train(train_dataset)

    #loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, train_accuracies = train_model(model, optimizer, loss_fn, train_dl)

    print(train_losses, train_accuracies)

def run_AutoEncoder(mu):
    """
    Preprocess data, and runs and returns metrics of the VGG16 TL model
    """
    print("ELEPHANT")

    #Loading and data pre-processing
    mnist_train, mnist_test, x_train, y_train, x_test, y_test = loadMNISTData()
    train_dataset, train_dl, test_dataset, test_dl = defineDataLoaders(x_train, y_train, x_test, y_test, mu)

    #Visualize the dataset
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train[0])

    #get model
    model = AutoencoderModel().to(device)

    #Plot noisy images
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(x_train[i])
        plt.title(f"Noisy {y_train[i]}")
        plt.show()

    #model.train(train_dataset)

    #loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, train_accuracies = train_model(model, optimizer, loss_fn, train_dl)

    print(train_losses, train_accuracies)

    #Plot denoised images
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(x_train[i])
        plt.title(f"Label {y_train[i]}")
        plt.show()


def runAll():
    run_U_Net(1)
    run_AutoEncoder(1)
    

runAll()

#NOTE: How to export images?
    