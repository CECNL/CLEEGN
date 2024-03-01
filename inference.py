from utils.cleegn import CLEEGN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy as np2TT
from torchinfo import summary

from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
import numpy as np
import math
import json
import time
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
electrode = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'PO8', 'O1', 'O2']

""" pyplot waveform visualization """
def plotEEG(tstmps, data_colle, ref_i, electrode, titles=None, colors=None, alphas=None, ax=None):
    n_data = len(data_colle)
    titles = ["" for di in range(n_data)] if titles is None else titles
    alphas = [0.5 for di in range(n_data)] if alphas is None else alphas
    if colors is None:
        cmap_ = plt.cm.get_cmap("tab20", n_data)
        colors = [rgb2hex(cmap_(di)) for di in range(n_data)]

    picks_chs = ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Fz", "Pz"]
    picks = [electrode.index(c) for c in picks_chs]
    for di in range(n_data):
        data_colle[di] = data_colle[di][picks, :]
    if ax is None:
        ax = plt.subplot()
    for ii, ch_name in enumerate(picks_chs):
        offset = len(picks) - ii - 1
        norm_coef = 0.25 / np.abs(data_colle[ref_i][ii]).max()
        for di in range(n_data):
            eeg_dt = data_colle[di]
            ax.plot(tstmps, eeg_dt[ii] * norm_coef + offset,
                label=None if ii else titles[di], color=colors[di], alpha=alphas[di],
                linewidth=3 if alphas[di] > 0.6 else 1.5, # default=1.5
            )
    ax.set_xlim(tstmps[0], tstmps[-1])
    ax.set_ylim(-0.5, len(picks) - 0.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks(np.arange(len(picks)))
    ax.set_yticklabels(picks_chs[::-1], fontsize=20)
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower right", borderaxespad=0, ncol=3, fontsize=20
    )

def ar_through_model(eeg_data, model, window_size, stride):
    model.eval()

    noiseless_eeg = np.zeros(eeg_data.shape, dtype=np.float32)
    hcoef = np.zeros(eeg_data.shape[1], dtype=np.float32)

    hwin = signal.hann(window_size) + 1e-9
    for i in range(0, noiseless_eeg.shape[1], stride):
        tstap, LAST_FRAME = i, False
        segment = eeg_data[:, tstap: tstap + window_size]
        if segment.shape[1] != window_size:
            tstap = noiseless_eeg.shape[1] - window_size
            segment = eeg_data[:, tstap:]
            LAST_FRAME = True
        with torch.no_grad():
            segment = np.expand_dims(segment, axis=0)
            data = np2TT(np.expand_dims(segment, axis=0))
            data = data.to(device, dtype=torch.float)
            pred_segment = model(data)
            pred_segment = np.array(pred_segment.cpu()).astype(np.float32)
        noiseless_eeg[:, tstap: tstap + window_size] += \
            pred_segment.squeeze() * hwin
        hcoef[tstap: tstap + window_size] += hwin

        if LAST_FRAME:
            break
    noiseless_eeg /= hcoef

    return noiseless_eeg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="removal artifact from multi-channel EEG data")
    parser.add_argument("--mat-path", required=True, type=str, help="path to EEG data (.mat)")
    parser.add_argument("--model-path", required=True, type=str, help="path to pre-trained model (.pth)")
    args = parser.parse_args()

    mat = loadmat(args.mat_path)
    dt_polluted, dt_ref = mat["x_test"], mat["y_test"]

    ### temporary fixed mode
    state_path = os.path.join(args.model_path)
    state = torch.load(state_path, map_location="cpu")
    model = CLEEGN(n_chan=56, fs=128.0, N_F=56).to(device)
    model.load_state_dict(state["state_dict"])
    dt_cleegn = ar_through_model(
        dt_polluted, model, math.ceil(4.0 * 128.0), math.ceil(1.0 * 128.0)
    )

    x_min, x_max = 1500, 1500 + 1000
    x_data = dt_polluted[:, x_min: x_max]
    y_data = dt_ref[:, x_min: x_max]
    p_data = dt_cleegn[:, x_min: x_max]
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plotEEG(
        np.linspace(0, math.ceil(x_data.shape[-1] / 128.0), x_data.shape[-1]),
        [x_data, y_data, y_data, p_data], 1, electrode,
        titles=["Original", "", "Reference", "CLEEGN"], colors=["gray", "gray", "red", "blue"], alphas=[0.5, 0, 0.8, 0.8], ax=ax
    )
    plt.show()
