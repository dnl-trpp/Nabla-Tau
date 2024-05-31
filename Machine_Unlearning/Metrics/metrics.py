import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import random

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    DEVICE = next(net.parameters()).device
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    DEVICE = next(net.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def compute_moments(net,loader,criterion=nn.CrossEntropyLoss(reduction='none')):

    total_samples = 0
    first_test_moment = 0
    second_test_moment = 0
    val_loss = 0
    DEVICE = next(net.parameters()).device

    for x,y in loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      out = net(x)
      losses = criterion(out,y)
      total_samples += y.size(dim=0)

      val_loss += torch.sum(losses).item()


    val_loss = val_loss / total_samples

    for x,y in loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      out = net(x)
      losses = criterion(out,y)

      first_test_moment += torch.sum((losses-val_loss)**2).item()


    first_test_moment = first_test_moment / total_samples
    test_std = first_test_moment ** 0.5

    for x,y in loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      out = net(x)
      losses = criterion(out,y)

      second_test_moment += torch.sum((losses-val_loss)**3).item()

    second_test_moment = second_test_moment / total_samples
    second_test_moment = second_test_moment / (test_std**3)

    return val_loss,first_test_moment, second_test_moment, test_std


def compute_entropy(net, loader):
    """Auxiliary function to compute per-sample losses"""

    #data_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers = 2, prefetch_factor = 10)
    prob = []
    DEVICE = next(net.parameters()).device
    with torch.no_grad():
        for data,target in loader:
            #batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = net(data)
            prob.append(torch.nn.functional.softmax(output, dim=-1).data)

    p = torch.cat(prob)
    return (-torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=-1, keepdim=False)).numpy(force=True)

