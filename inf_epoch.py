from utils.cleegn import CLEEGN
from utils.data_process import *
from utils.tools import *

import torch
import torch.nn as nn
from torchinfo import summary

from scipy.io import loadmat
from scipy import signal
import numpy as np
import math
import time
import mne
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("set_path", type=str, help="path to eeglab .set file")
    parser.add_argument("model_path", type=str, help="path to .pth file")
    parser.add_argument("--prefix", type=str, default="/tmp/result", help="write path prefix")
    parser.add_argument("--nn", type=str, help="specify nerual network structure")
    parser.add_argument("--tmin", type=float, default=-4.0, help="start time, for epoching")
    parser.add_argument("--tmax", type=float, default=10.0, help="stop  time, for epoching")
    parser.add_argument("--baseline", type=float, default=5.0, help="time of baseline")
    args = parser.parse_args()

    raw = mne.io.read_raw_eeglab(args.set_path, verbose=0)
    pick_channels = None # []: list, ToBeModify
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id=event_id, picks=pick_channels,
        tmin=args.tmin, tmax=args.tmax, baseline=(args.tmin, args.tmin + args.baseline)
    )

    if not (args.nn is None):
        state_path = args.model_path
        state = torch.load(state_path, map_location="cpu")

        if args.nn == "cleegn":
            # KeyError: build directory manaully
            model = CLEEGN(**state["struct_args"]).to(device)
        else:
            raise NotImplementedError("`{}` not implement".format(cfg["model_name"]))
        model.load_state_dict(state["state_dict"])
        model.eval()

        idx = 0
        rArr = np.zeros(
            (len(epochs.events), epochs.info["nchan"], epochs.times.size),
            dtype=np.float32
        )
        for epoch in epochs:
            X = epoch.copy()

            """ EXperimental, channel normalization """
            for i in range(X.shape[0]):
                s = X[i, :].std()
                s = X[i, (-s < X[i]) & (X[i] < s)].std()
                X[i] /= s

            fs = epochs.info["sfreq"]
            step_ratio = 0.125
            width = math.ceil(4.0 * fs)
            step = math.ceil(width * step_ratio)

            BREAK_FLAG = False
            hwin = signal.hann(width) + 1e-9
            hcoef = np.zeros(X.shape[-1], dtype=np.float32)
            Yr_hat = np.zeros(X.shape, dtype=np.float32)
            for i in range(0, Yr_hat.shape[-1], step):
                if BREAK_FLAG:
                    break

                X_ = X[..., i: i + width]
                if X_.shape[-1] < width:
                    X_ = X[:, -width:]
                    BREAK_FLAG = True

                with torch.no_grad():
                    X_ = torch.from_numpy(X_).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
                    X_ = model(X_).detach().cpu().squeeze().numpy()

                ii = Yr_hat.shape[-1] - width if BREAK_FLAG else i
                Yr_hat[:, ii: ii + width] += X_ * hwin
                hcoef[ii: ii + width] += hwin
            Yr_hat /= hcoef

            rArr[idx] = Yr_hat
            idx += 1
        ## END_OF_LOOP_EPOCHS ##
        epochs = mne.EpochsArray(rArr, epochs.info, events=events, tmin=args.tmin, event_id=event_id)
    epochs.save(f"{args.prefix}-epo.fif", overwrite=True)
## END_OF_MAIN_FUNC ##
