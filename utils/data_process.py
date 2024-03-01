from .tools import assert_
import numpy as np
import time
import math
import mne

def sliding_epoch(A, width, step):
    runList = range(width, A.shape[-1] + 1, step)
    epochs = np.zeros((len(runList), *A.shape[:-1], width), dtype=np.float32)
    for i, k in enumerate(runList):
        epochs[i] = A[..., k - width: k]
    return epochs

def create_dataset(fmt_terms, range_t, cfg):
    # setting up variables
    sfreq = cfg["sfreq"]
    assert_(cfg["duration_unit"] in ["sample", "sec", "min"]), "undefine str"
    width, step = cfg["window"], cfg["step"]
    tmin, tmax = range_t
    if cfg["duration_unit"] != "sample":
        width *= sfreq; step *= sfreq; tmin *= sfreq; tmax *= sfreq
    if cfg["duration_unit"] == "min":
        width *= 60; step *= 60; tmin *= 60; tmax *= 60
    width, step = math.ceil(width), math.ceil(step)
    tmin, tmax = math.ceil(tmin), math.ceil(tmax)

    # read eeglab .set and generate dataset
    x, y = None, None
    for ft in fmt_terms:
        x_raw = mne.io.read_raw_eeglab(cfg["x_fpath"].format(*ft), verbose=0)
        y_raw = mne.io.read_raw_eeglab(cfg["y_fpath"].format(*ft), verbose=0)

        assert_(x_raw.info["sfreq"] == y_raw.info["sfreq"], "unpair sfreq")
        assert_(x_raw.info["sfreq"] == sfreq, "invailid sfreq")
        for _1, _2 in zip(x_raw.ch_names, y_raw.ch_names):
            assert_(_1 == _2, "unpair channel name")
    
        x_data = x_raw[:, tmin: tmax][0]
        y_data = y_raw[:, tmin: tmax][0]
        if cfg["ch_names"] is not None:
            picks = [x_raw.ch_names.index(ch) for ch in cfg["ch_names"]]
            x_data, y_data = x_data[picks, :], y_data[picks, :]

        """ EXperimental, channel normalization """
        for i in range(x_data.shape[0]):
            s = x_data[i, :].std()
            s = x_data[i, (-s < x_data[i]) & (x_data[i] < s)].std()
            x_data[i] /= s

            s = y_data[i, :].std()
            s = y_data[i, (-s < y_data[i]) & (y_data[i] < s)].std()
            y_data[i] /= s

        x_epochs = sliding_epoch(x_data, width, step)
        y_epochs = sliding_epoch(y_data, width, step)

        """ EXperimental, ban extreme large std sample, usually is non-physical artifact """
        aErr = np.abs(x_epochs - y_epochs).sum(axis=1).mean(axis=-1)
        mask = aErr < (aErr.mean() + 3 * aErr.std())
        x_epochs = x_epochs[mask, ...]
        y_epochs = y_epochs[mask, ...]

        if (x is None) and (y is None):
            x, y = x_epochs, y_epochs
        else:
            x = np.append(x, x_epochs, axis=0)
            y = np.append(y, y_epochs, axis=0)
    return x, y

def Kfold_split(A, k: int, shuffle=True, seed=None):
    assert_(isinstance(k, int), "`k` should be `int`")
    indices = np.arange(0, len(A), dtype=int)
    if shuffle:
        np.random.shuffle(indices)
    groups = np.array_split(indices, k)

    B = [[A[i] for i in g] for g in groups]
    return tuple(B)
