import gc
import obspy
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pd.set_option('display.max_columns', None)

from matplotlib.patches import Rectangle
import seisbench.models as sbm
from obspy.signal.filter import bandpass
from obspy.core.utcdatetime import UTCDateTime
from tqdm import tqdm


import prediction_methods as pm



# cpu, or cuda
device = torch.device("cpu")
# starttime = UTCDateTime("2022-02-06T00:00:00.000000")
# endtime   = UTCDateTime("2022-02-06T03:59:59.999000")
# # s = client.get_waveforms("PB", "B204", location = "*", channel = "EH?", 
#                          starttime = starttime, endtime = endtime)
print(f'PyTorch is using {torch.get_num_threads()} threads')

# s0 = obspy.read("https://github.com/congcy/ELEP/raw/main/docs/tutorials/data/PB.B204..EH.ms")
s0 = obspy.read('./GNW.UW.2017.131.ms',fmt='MSEED').select(channel='?N?')
s0.trim(endtime=s0[0].stats.starttime + 4.*3600.)
s0.detrend("spline", order = 3, dspline=1500).normalize()
print(s0)
try:
    s1 = obspy.Stream([s0.select(channel=f'??{x}')[0].copy() for x in 'ZNE'])
except IndexError:
    s1 = obspy.Stream([s0.select(channel=f'??{x}')[0].copy() for x in 'Z12'])
print(s1)
npts = s1[0].stats.npts
delta = s1[0].stats.delta
starttime = s1[0].stats.starttime

# cut continuous data into one-minute time window (6000 sample for EqTransformer)
# 3000 (50%, 30 seconds) sample overlap 
# mute 500 samples of prediction on both end of time window

twin = 6000     # length of time window
step = 3000     # step length
l_blnd, r_blnd = 500, 500
nseg = int(np.ceil((npts - twin) / step))

eqt = sbm.EQTransformer.from_pretrained('pnw')
eqt.to(device);
eqt._annotate_args['overlap'] = ('Overlap between prediction windows in samples \
                                 (only for window prediction models)', step)
eqt._annotate_args['blinding'] = ('Number of prediction samples to discard on \
                                  each side of each window prediction', (l_blnd, r_blnd))

eqt.filter_args = None
eqt.filter_kwargs = None
eqt.eval();

# Apply pre-processing to stream specified by model
s2 = eqt.annotate_stream_pre(s1.copy(), argdict=eqt._annotate_args)
try:
    sdata = np.array([s2.select(channel=f'??{x}')[0].data for x in 'ZNE'], dtype=np.float32) #s2)[[2,0,1], :] #12Z
except IndexError:
    sdata = np.array([s2.select(channel=f'??{x}')[0].data for x in 'Z12'], dtype=np.float32) #s2)[[2,0,1], :] #12Z
# sdata: Z12

windows_std = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
windows_max = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
_windows = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
windows_idx = np.zeros(nseg, dtype=np.int32)
tap = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 6)))

for iseg in range(nseg):
    idx = iseg * step
    _windows[iseg, :] = sdata[:, idx:idx + twin]
    _windows[iseg, :] -= np.mean(_windows[iseg, :], axis=-1, keepdims=True)
    # original use std norm
    windows_std[iseg, :] = _windows[iseg, :] / np.std(_windows[iseg, :]) + 1e-10
    # others use max norm
    windows_max[iseg, :] = _windows[iseg, :] / (np.max(np.abs(_windows[iseg, :]), axis=-1, keepdims=True))
    windows_idx[iseg] = idx

# taper
windows_std[:, :, :6] *= tap; windows_std[:, :, -6:] *= tap[::-1]; 
windows_max[:, :, :6] *= tap; windows_max[:, :, -6:] *= tap[::-1];
del _windows

print(f"Window data shape: {windows_std.shape}")



# dim 0: 0 = P, 1 = S
pred = np.zeros([nseg, 3, twin], dtype = np.float32) 

t0 = time.time()
windows_max_tt = torch.Tensor(windows_max)
_torch_pred = eqt(windows_max_tt.to(device))
pred[:, 0, :] = _torch_pred[0].detach().cpu().numpy()
pred[:, 1, :] = _torch_pred[1].detach().cpu().numpy()
pred[:, 2, :] = _torch_pred[2].detach().cpu().numpy()
stack = pm.restructure_predictions(pred, windows_idx, eqt)
bare_stream = pm.stack2stream(stack, s2, eqt)
t1 = time.time()
print(f"picking using BareMetal method: {t1 - t0: .3f} second")
    
# clean up memory
del _torch_pred, windows_max_tt
del windows_max
gc.collect()
torch.cuda.empty_cache()

t2 = time.time()
annotate_stream = eqt.annotate(s2)
t3 = time.time()

print(f"model.annotate runtime: {t3 - t2}")