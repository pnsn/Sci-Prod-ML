"""
:module: benchmark_stream2array.py
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:purpose:

"""
import torch
import prediction_methods as pm
import numpy as np
import pandas as pd
from obspy import read
from time import time
from tqdm import tqdm

# As a default, run on a single CPU (2 threads)
torch.set_num_threads(2)

# Document "random" value generator seed
rnd_seed = np.random.default_rng(62323)


def _startup(testdata='GNW.UW.2017.131.ms'):
    """
    Convenience
    """
    # Load waveform data
    stream = read(testdata, fmt='MSEED')
    # Shorten to 2 hours
    stream = stream.trim(endtime=stream[0].stats.starttime + 6.*3600.)
    # Convert data to array
    data = pm.stream2array(stream, band='?', inst='N', chanorder='ZNE')
    # Initialize model
    model, device = pm.initialize_EQT_model()

    return stream, data, model, device


def benchmark_windowing(data, model, ndraw=100):
    d_meth = ['median', 'mean']
    n_meth = ['max', 'std']
    vals = np.zeros(shape=(ndraw, 4), dtype=np.float32)
    cols = []
    for _i in tqdm(range(ndraw)):
        _j = 0
        for _d in d_meth:
            for _n in n_meth:
                tick = time()
                outs = pm.prepare_windows(data, model,
                                          detrend_method=_d,
                                          norm_method=_n,
                                          verbose=False)
                tock = time()
                vals[_i, _j] = tock - tick
                _j += 1
                del outs
                if _i == 0:
                    cols.append(f'{_d}-{_n}')
    df = pd.DataFrame(vals, columns=cols)
    return df


def prep_data_for_pred(data, model):
    # Window data
    windows, windex, _ = pm.prepare_windows(data, model,
                                            detrend_method='mean',
                                            norm_method='max')
    # Taper data
    windows = pm.taper(windows, ntaper=6)
    # Convert into torch.Tensor
    windows_tt = torch.Tensor(windows)
    return windows_tt, windex


def benchmark_window_count_logscale(model, windows_tt, device,
                                    nscales=10, maxscale=300, ndraw=10):
    """
    Run predictions on a randomly selected block of windows with progressively
    decreasing block size (log10 scaled)
    """
    print(f'PyTorch is predicting using {torch.get_num_threads()} threads')
    # Generate a log-scaling for amount of windows to predict
    if maxscale is None:
        scales = np.unique(np.logspace(0, np.log10(windows_tt.shape[0])//1,
                                       nscales, dtype=np.int32))
    else:
        scales = np.unique(np.logspace(0, np.log10(maxscale)//1,
                                       nscales, dtype=np.int32))
    vals = np.zeros(shape=(len(scales), ndraw), dtype=np.float32)
    for _i in tqdm(range(len(scales))):
        _s = scales[-_i - 1]
        for _j, _d in enumerate(range(ndraw)):
            # Get random, valid starting point
            _ind = np.random.randint(0, windows_tt.shape[0] - _s, 1)[0]
            wtt = windows_tt[_ind:_ind+_s]
            tick = time()
            preds = pm.run_prediction(wtt, model, device)
            tock = time()
            del preds
            vals[_i, _j] = (tock - tick)/_s
    try:
        df = pd.DataFrame(vals.T, columns=scales)
    except ValueError:
        breakpoint()
    return df


def benchmark_window_count_finescale(model, windows_tt, device,
                                     scales=np.arange(50, 160, 10), ndraw=10):
    """
    Run predictions on a randomly selected block of windows
    """
    print(f'PyTorch is predicting using {torch.get_num_threads()} thread(s)')
    # Generate a log-scaling for amount of windows to predict

    vals = np.zeros(shape=(len(scales), ndraw), dtype=np.float32)
    for _i in tqdm(range(len(scales))):
        _s = scales[-_i - 1]
        for _j, _d in enumerate(range(ndraw)):
            # Get random, valid starting point
            _ind = np.random.randint(0, windows_tt.shape[0] - _s, 1)[0]
            wtt = windows_tt[_ind:_ind+_s]
            tick = time()
            preds = pm.run_prediction(wtt, model, device)
            tock = time()
            del preds
            vals[_i, _j] = (tock - tick)/_s
    try:
        df = pd.DataFrame(vals.T, columns=scales)
    except ValueError:
        breakpoint()
    return df