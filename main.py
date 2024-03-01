from utils.cleegn import CLEEGN
from utils.tv_epoch import train
from utils.tv_epoch import val
from utils.data_process import *
from utils.tools import *

import torch
import torch.nn as nn
from torchinfo import summary

from scipy.io import loadmat
import numpy as np
import math
import time
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAG_LOAD_SAVED = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("config_file", type=str, help="path to configuration file")
    args = parser.parse_args()

    k = 0
    cfg = read_json(args.config_file)
    n_epoch       = cfg["epochs"]
    learning_rate = cfg["lr"]
    ava_path = "weights/{}_{}".format(cfg["model_name"], cfg["trial_name"])
    os.makedirs(ava_path, exist_ok=True)

    tra_fmt_terms = cfg["fmt_terms"]
    val_fmt_terms = cfg["fmt_terms2"]
    print(f"{tra_fmt_terms}\n{val_fmt_terms}\n")

    x_train, y_train = create_dataset(tra_fmt_terms, cfg["range_t"], cfg)
    x_valid, y_valid = create_dataset(val_fmt_terms, cfg["range_t"], cfg)

    x_train = torch.from_numpy(x_train).unsqueeze(1)
    y_train = torch.from_numpy(y_train).unsqueeze(1)
    x_valid = torch.from_numpy(x_valid).unsqueeze(1)
    y_valid = torch.from_numpy(y_valid).unsqueeze(1)
    print(x_train.size(), y_train.size())
    print(x_valid.size(), y_valid.size())

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    tra_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(
        validset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )

    struct_args = dict(n_chan=x_train.size(2), fs=cfg["sfreq"], N_F=x_train.size(2))
    model = CLEEGN(**struct_args).to(device)
    summary(model, input_size=(cfg["batch_size"], *list(x_train.size())[1:]))

    if FLAG_LOAD_SAVED:
        state_path = f"{ava_path}/epoch-minloss-{k}.pth"
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        print("[ NOTICE ] load pre-trained weight ...ok")

    loss_fn = nn.MSELoss()
    if cfg["optim"] == "adam":
        opt_fn = torch.optim.Adam(model.parameters(), lr=learning_rate, **cfg["optim_args"])
    elif cfg["optim"] == "sgd":
        opt_fn = torch.optim.SGD(model.parameters(), lr=learning_rate, **cfg["optim_args"])
    else:
        raise NotImplementedError("`{}` not implement".format(cfg["optim"]))
    print(cfg["optim_args"])

    """ should be added to conf """
    sche_fn = None
    #sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fn, mode="min", factor=0.8, patience=40)
    #sche_fn = torch.optim.lr_scheduler.MultiStepLR(opt_fn, milestones=[20, 40, 80, 160, 320], gamma=0.8)
    #sche_fn = torch.optim.lr_scheduler.ExponentialLR(opt_fn, gamma=0.8)

    lr_now = learning_rate
    loss_curve  = np.zeros((n_epoch, ), dtype=np.float32) + np.inf
    vloss_curve = np.zeros((n_epoch, ), dtype=np.float32) + np.inf
    for ep in range(n_epoch):
        ep_time0 = time.time()
        loss = train(tra_loader, model, loss_fn, opt_fn)
        val_loss = val(val_loader, model, loss_fn)

        FLAG_MIN_LOSS = loss < loss_curve.min()
        FLAG_MIN_VLOSS = val_loss < vloss_curve.min()
        loss_curve[ep], vloss_curve[ep] = loss, val_loss
        
        if FLAG_MIN_LOSS or FLAG_MIN_VLOSS:
            checkpoint = dict(
                epoch=ep, state_dict=model.state_dict(), loss_curve=(loss_curve, vloss_curve),
                ini_lr=learning_rate, struct_args=struct_args,
                train_terms=tra_fmt_terms
            )
            if FLAG_MIN_LOSS:
                torch.save(checkpoint, f"{ava_path}/train-{k}.pth")
            if FLAG_MIN_VLOSS:
                torch.save(checkpoint, f"{ava_path}/valid-{k}.pth")
            ##torch.save(checkpoint, f"{ava_path}/epoch-{ep:03d}-{k}.pth")

        print("\rEp_{:03d} - {:.1f} - loss: {:.4f}{} - val_loss: {:.4f}{} - lr: {:.1e}".format(
            ep, time.time() - ep_time0,
            loss, "*" if FLAG_MIN_LOSS else "",
            val_loss, "*" if FLAG_MIN_VLOSS else "",
            lr_now
        ))
        for param_group in opt_fn.param_groups:
            lr_now = param_group["lr"]
            break

        if not (sche_fn is None):
            sche_fn.step()
    ## END_OF_TRAIN_LOOP ##
## END_OF_MAIN_FUNC ##
