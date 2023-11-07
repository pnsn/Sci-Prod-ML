"""
:module: benchmark_predict.py
:author: Nathan T. Stevens
:email: ntsteven at uw.edu
:org: Pacific Northwest Seismic Network
:purpose: Provide a set of benchmark tests for PyTorch facilitated 
          earthquake detection, body wave arrival time, and body wave
          classification probability modeling enforcing a thread count
          for an AWS EC2 t2.medium+ instance (i.e., 1 cpu [2 threads]
          and 4+ Gb memory*). This example uses 1 day of data from the 
          Green Mountain 6-channel PNSN site (UW.GNW), focusing on the 
          accelerometer records (UW.GNW..?N?), that coincides with the 
          ml 3.5 main-shock of the nearby 11 May 2017 Bremerton earthquake
          swarm (year 2017, julian day 131). The provided example model
          via the `_startup` method uses the EQTransformer method with
          retrained weights based on the STEAD model (Mosavi et al., 2020)
          using 21 years of PNSN broadband (velocimiter) data described
          in Ni et al. (2023). 
"""
import torch
import prediction_methods as pm
import numpy as np
import pandas as pd
from obspy import read
from time import time
from tqdm import tqdm

# As a default, run on a single CPU (1 threads)
# torch.set_num_threads(1)

# Document "random" value generator seed
rnd_seed = np.random.default_rng(62323)


def _startup(testdata='GNW.UW.2017.131.ms', icode='N'):
    """
    Convenience method for instantiating the example
    dataset, model, and device objects. The default
    `testdata` kwarg assumes a local copy of the 
    velocimeter (BH?) and accelerometer (EN?) data from
    network = 'UW', station = 'GNW', location = '*', channel = '*'
    which can be downloaded and saved using the 
    obspy.clients.fdsn class-method with the following code:
        from obspy.clients.fdsn import client
        from obspy import UTCDateTime
        client = Client("IRIS")
        st = client.get_waveforms(network='UW', station='GNE',
                                  location='*', channel='*',
                                  starttime=UTCDateTime(2017,5,11),
                                  endtime=UTCDateTime(2017,5,12))
        st.write('GNW.UW.2017.131.ms',fmt='MSEED')

    At present (Oct. 2023) this instantiation also assumes that 
    the `pnw` EQTransformer model from Ni et al. (2023) has been placed
    in the following location:

    The pnw.json.v1 and pnw.pt.v1 files for this model are downloadable from:
    https://github.com/congcy/ELEP/tree/main/docs/tutorials/data
    installation via `wget` documented here:
    https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb
    
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
    """
    Wrapper to window data as specified in the model object
    and convert the windowed data array into a pytorch
    tensor. Return the tensor and the window index vector
    `windex`
    """

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


def compare_prediction_pipeines(testdata='GNW.UW.2017.131.ms', nhours=4, icode='?N?', nruns=100):
    """
    Run something
    """
    # Initialize common model
    model, device = pm.initialize_EQT_model()
    # Manually suppress prefilter
    model.filter_args = None
    model.filter_kwargs = None
    # Load common dataset
    st = read(testdata, fmt='MSEED').select(channel='?N?')
    # Trim to specified length
    st = st.trim(endtime=st[0].stats.starttime + nhours*3600.)
    # Enforce sequencing
    stream = st.select(channel='ENZ')
    stream += st.select(channel='ENN')
    stream += st.select(channel='ENE')
    print(st)
    # Apply preprocessing to all data being considered
    # s2 = model.annotate_stream_pre(stream.copy(), argdict=model._annotate_args)
    s2 = stream.copy()
    print(s2)
    print(s2[0].stats)
    # Preallocate timing array
    vals = np.zeros(shape=(4, nruns), dtype=np.float32)
    # Iterate across specified  numer of iterations
    for _i in tqdm(range(nruns)):
       

        # Do bare-metal prediction
        # Wrapped start
        t2 = time()
        data = pm.stream2array(s2, band='?', inst='N', chanorder='ZNE')
        # Core start
        t2a = time()
        windows, windex, _ = pm.prepare_windows(data, model,
                                                detrend_method='mean',
                                                norm_method='max',
                                                verbose=False)
        windows = pm.taper(windows)
        windows_tt = torch.Tensor(windows)
        preds = np.zeros(shape=windows_tt.shape, dtype=np.float32)
        # PredOnly start
        t2b = time()
        _torch_preds = model(windows_tt.to(device))
        t3b = time()
        # PredOnly stop
        for _a, _p in enumerate(_torch_preds):
            preds[:, _a, :] = _p.detach().cpu().numpy()
        stack = pm.restructure_predictions(preds, windex, model)
        t3a = time()
        # Core stop
        bm_stream = pm.stack2stream(stack, stream, model)
        t3 = time()
        # Wrapped stop
        # Record runtimes
        vals[1, _i] = t3 - t2
        vals[2, _i] = t3a - t2a
        vals[3, _i] = t3b - t2b
        # Clear generated parameters
        if _i < nruns - 1:
            del data
            del windows
            del windex
            del windows_tt
            del preds
            del stack
            del bm_stream
        else:
            print('preserving last baremetal outputs')

     # Do seisbench-facilitated run
        t0 = time()
        annotation_stream = pm.run_seisbench_pred(s2, model)
        t1 = time()
        # Record runtime
        vals[0, _i] = t1 - t0
        # Clear generated parameters
        if _i < nruns - 1:
            del annotation_stream
        else:
            print('preserving last annotation')
    # Compose output dataframe
    df = pd.DataFrame(vals.T, columns=['SeisBench', 'BareMetal_Wrapped', 'BareMetal_Core', 'Baremetal_PredOnly'])
    return df, annotation_stream, bm_stream

