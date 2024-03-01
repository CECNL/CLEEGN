import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import numpy as np
import math
import json
import time
import mne
import sys
import os

def val(val_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # switch to evaluation mode

    epoch_loss = np.zeros((len(val_loader), ))
    for i, (x_batch, y_batch) in enumerate(val_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)
        with torch.no_grad():
            output = model(x_batch)
        loss = criterion(output, y_batch)

        epoch_loss[i] = loss.item()
    return epoch_loss.mean(axis=0)


def train(tra_loader, model, criterion, optimizer, verbose=1):
    max_iter = len(tra_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # switch to train mode

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((max_iter, ))
    for i, (x_batch, y_batch) in enumerate(tra_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss[i] = loss.item()
        if verbose:
            print("\r{}".format(" " * len(log)), end="")
            log = "\r{}/{} - {:.4f} s - loss: {:.4f} - acc: nan".format(
                i + 1, max_iter, time.time() - ep_time0, epoch_loss[i]
            )
            print(log, end="")
    return epoch_loss.mean(axis=0)
