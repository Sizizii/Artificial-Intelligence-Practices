# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 1
class2 = 2

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # raise NotImplementedError("You need to write this part!")
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size

        self.hidden_size = 32

        self.model = nn.Sequential(
            nn.Linear(self.in_size, 128),
            nn.ReLU(),
            nn.Linear(128,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size)
        )

        self.optimizer = optim.Adam(self.model.parameters(), self.lrate)

    # def set_parameters(self, params):
    #     """ Sets the parameters of your network.

    #     @param params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    # def get_parameters(self):
    #     """ Gets the parameters of your network.

    #     @return params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")
        return self.model(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        #raise NotImplementedError("You need to write this part!")

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = self.forward(x)

        # Compute and print loss.
        loss_value = self.loss_fn(y_pred, y)
        loss = loss_value.item()

        self.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss_value.backward()
        self.optimizer.step()

        return loss


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #raise NotImplementedError("You need to write this part!")

    #data standardization
    train_mean = train_set.mean(dim=1,keepdim=True)
    train_std = train_set.std(dim=1,keepdim=True)
    train_set = (train_set-train_mean)/train_std

    dev_mean = dev_set.mean(dim=1,keepdim=True)
    dev_std = dev_set.std(dim=1,keepdim=True)
    dev_set = (dev_set-dev_mean)/dev_std

    lrate = 0.0005
    in_size = train_set.shape[1]

    net = NeuralNet(lrate, nn.CrossEntropyLoss(), in_size, 2)

    losses = []

    data_batches = torch.split(train_set, batch_size, dim=0)
    label_batches = torch.split(train_labels, batch_size, dim=0)
    batch_len = len(data_batches)

    for i in range(n_iter):

        while i >= batch_len:
            j = np.random.randint(0,batch_len,1)
            i = j[0]
        data_batch = data_batches[i]
        label_batch = label_batches[i]

        loss = net.step(data_batch, label_batch)
        losses.append(loss)

    yhats = np.argmax(net.forward(dev_set).detach().numpy(),axis=1)

    return losses, yhats, net
