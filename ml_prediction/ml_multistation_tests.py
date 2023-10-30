"""
:module: ml_multistation_tests.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:purpose: Test cases for running predictions on arbitrary assemblages of
          pre-processed, pre-windowed data from multiple seismic stations
          using PyTorch implementations from SeisBench (Woollam et al., 2022).
"""
import os
import numpy as np
from obspy import read, Stream, UTCDateTime
from obspy.clients.fdsn import Client
import prediction_methods as ml
from time import time


def _download_example_waveforms(chanlist=['BH?','EH?','EN?','HH?','HN?'],
                                reftime=UTCDateTime(2017, 5, 11, 2, 31),
                                pad=120):
    client = Client("IRIS")
    st_dict = {}
    try:
        os.mkdir(f'data_{pad/30:d}min')
    except FileExistsError:
        pass
    t0 = time()
    for _ch in chanlist:
        print(f'Processing {_ch} -- elapsed time {time() - t0: .3f} sec')
        _st = client.get_waveforms(network='UW', station='*',
                                  location='*', channel=_ch,
                                  starttime=reftime - pad, endtime=reftime + pad)
        _st_dict = _get_unique_NSBI_from_stream(_st)
        for _k in st_dict.keys():
            __st = _st_dict[_k]
            pok = _k.split('.')
            if len(__st) > 0:
                fname = os.path.join(f'data_{pad/30:d}min',
                                     f'{pok[0]}.{pok[1]}.{pok[3][:2]}.mseed')
                __st.write(fname, fmt='MSEED')
                st_dict.update({_k:__st})
        print(f'Saving {_ch} ------ elapsed time {time() - t0: .3f} sec')
    return st_dict


def _get_model():
    """
    Convenience method for loading PNW-trained EQTransformer model and
    specifying a 'cpu' device for completeness of workflow
    """
    model, device = ml.initialize_EQT_model()
    return model, device


def _get_testdata(ROOT='data_5min', kwargs={'fmt': 'MSEED'}):
    """
    Convenience method for loading 5 minutes of seismic data from
    broadband and strong-motion sites within 50 km of Bremerton, WA
    on May 11th 2017 (start of an EQ sequence) from a local
    data repository. UTC Times 2017-05-11T02:30:00 to 2017-05-11T02:34:59
    for channels fitting unix wildcard notation: `[BEH][NH]?` for the `UW`
    network of seismic stations.

    This encompasses a clear local event on UW.GNW at approximately 02:31

    This collection includes stations with sampling rate of 40, 100, & 200 sps
    that trigger all resampling cases in _preprocess_testdata()

    :: INPUT ::
    :param ROOT: directory path (with OS-specific syntax) to example data
                 directory that conforms to the following glob.glob search:
                    os.path.join(ROOT,'*.mseed')
    :param kwargs: kwargs to pass to obspy.core.read

    :: OUTPUT ::
    :return st: [obspy.core.stream.Stream]
                Stream composed of Trace objects
    """
    from glob import glob
    flist = glob(os.path.join(ROOT, '*.mseed'))
    st = Stream()
    for _f in flist:
        st += read(_f, **kwargs)
    return st


def _get_unique_NSBI_from_stream(stream):
    """
    Get unique combinations of network, station,
    and Band/Instrument codes of traces (characters
    1 and 2 in a SEED channel name) from input Stream
    and use these to split traces into subgroups structured as

    {'UW.BABE..EN?': 3 Trace(s) in Stream:
    UW.BABE..ENE | 2017-05-11T02:00:00.000000Z - 2017-05-11T02:05:00.000000Z | 100.0 Hz, 30001 samples
    UW.BABE..ENN | 2017-05-11T02:00:00.000000Z - 2017-05-11T02:05:00.000000Z | 100.0 Hz, 30001 samples
    UW.BABE..ENZ | 2017-05-11T02:00:00.000000Z - 2017-05-11T02:05:00.000000Z | 100.0 Hz, 30001 samples,
    ...
    }

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream] - Stream object containing traces
    :param chan_order: [list of strings]

    :: OUTPUT ::
    :return st_dict: [dictionary of Streams]
    """
    # Get unique Network.Station..BandInst? codes
    NSBI_list = []
    for _tr in stream.copy():
        _net = _tr.stats.network
        _sta = _tr.stats.station
        _cha = _tr.stats.channel
        _cbi = _cha[:2]
        _nsbi = f'{_net}.{_sta}..{_cbi}?'
        if _nsbi not in NSBI_list:
            NSBI_list.append(_nsbi)
    # Filter input Stream to split of subset streams
    st_list = []
    for _nsbi in NSBI_list:
        kwargs = dict(zip(['network', 'station',
                           'location', 'channel'],
                          _nsbi.split('.')))
        st_list.append(stream.copy().select(**kwargs))
    # Zip everything together for output
    st_dict = dict(zip(NSBI_list, st_list))
    return st_dict


def _resample_testdata(st_dict, samp_rate=100,
                       interp_kwargs={'method': 'weighted_average_slopes',
                                      'no_filter': False},
                       resamp_kwargs={'window': 'hann'}):
    """
    Wrapper for resampling and filtering test data with a basic
    decision tree for resampling method (and additional filtering)
    based on inputs of desired sampling rate

    :: INPUTS ::
    :param st_dict: [dictionary of obspy.core.stream.Stream]
                    Dictionary keys are N.S.L.BI codes, with BI
                    meaning the first 2 characters of a SEED channel name
                    --> the Band and Instrument character codes
    :param samp_rate: [int-like]
                    Desired uniform sampling rate for output data. This
                    is used for the `sampling_rate` positional argument
                    in the Trace.resample() and Trace.interpolate() methods
                    used in this method.
    :param interp_kwargs: [dict]
                    Dictionary for passing key-word arguments to the
                    Stream.interpolate(samp_rate, **interp_kwargs)
                    method. This method is used for data UPSAMPLING.
    :param resamp_kwargs: [dict]
                    Dictionary for passing key-word arguments to the
                    Stream.resample(samp_rate, **resamp_kwargs)
                    method. This method is used for data DOWNSAMPLING.

    :: OUTPUT ::
    :return st_dict: [dictionary of streams]
                    resampled/filtered stream data
                    NOTE: Operations are run IN PLACE, meaning that contents
                    of `st_dict` are altered. Use the copy.deepcopy method (not
                    imported in the current version of this module) to create
                    a copy of the dictionary and its contents.
    """
    # Iterate across unique NSBI codes
    for _k in st_dict.keys():
        _st = st_dict[_k].copy()
        _sr = _st[0].stats.sampling_rate
        # If upsampling data, use 'interpolate'
        if _sr < samp_rate:
            _st.interpolate(samp_rate, **interp_kwargs)
        # If _sr equals samp_rate, do nothing
        elif _sr == samp_rate:
            pass
        # If downsampling data, lowpass then resample()
        elif _sr > samp_rate:
            _st.resample(samp_rate, **resamp_kwargs)

        st_dict.update({_k: _st})

    return st_dict


def _build_windowed_array(st_dict, model, method_1C='ZP'):
    """
    TEST METHOD

    Take an arbitrary set of streams contained in a "stream dictionary"
    and convert their data into a numpy.ndarray scaled for input to a
    PyTorch model. Also provide an indexing array to document index-ownership
    for reorganizing ML prediction outputs.

    :: INPUTS ::
    :param st_dict: [dictionary of streams]
                    see _get_unique_NSBI_from_streams() for structure
    """
    dict_swindex = {}
    last_sindex = 0
    for _k in st_dict.keys():
        # Get subset stream
        _st = st_dict[_k]
        # Convert into an ordered array (pre-tensor)
        if len(_st) == 3:
            try:
                _data = ml.stream2array(_st, band='?', inst='?',
                                        chanorder='ZNE')
            except IndexError:
                _data = ml.stream2array(_st, band='?', inst='?',
                                        chanorder='Z12')
        # In event of a single-channel station
        # ZP: use 0-padded (Ni et al., 2023)
        elif len(_st) == 1 and method_1C == 'ZP':
            _data = np.c_[_st[0].data,
                          np.zeros(_st[0].stats.npts),
                          np.zeros(_st[0].stats.npts)].T
        # DUP: use duplicated Z data (Retailleau et al., 2021)
        elif len(_st) == 1 and method_1C == 'DUP':
            _data = np.c_[_st[0].data, _st[0].data, _st[0].data].T
        # Break _data into windows as specified by model
        _windows, _windex, _last_idx = ml.prepare_windows(_data, model,
                                                          verbose=False)
        # Concatenate windows
        if last_sindex == 0:
            windows = _windows
            _sindex = [last_sindex, windows.shape[0]]
            last_sindex = windows.shape[0]
        else:
            windows = np.concatenate([windows, _windows], axis=0)
            _sindex = [last_sindex, windows.shape[0]]
            last_sindex = windows.shape[0]

        dict_swindex.update({_k: (_sindex, _windex)})

    return windows, dict_swindex


def _apply_taper(windows, ntaper=6):
    """
    Convenience wrapper on ml.taper() method to document
    full workflow.
    """
    return ml.taper(windows, ntaper=ntaper)


def _run_prediction(windows, model, device):
    """
    Convenience wrapper on ml.run_prediction() method to
    document full workflow.
    """
    windows_tt = ml.torch.Tensor(windows)
    preds = ml.run_prediction(windows_tt, model, device)
    return preds


def _reassemble_multistation(preds, swindex, model, st_dict,
                             mod_wt_code='EQ', pred_codes=['D', 'P', 'S']):
    pred_stream = Stream()
    # Iterate across each station/instrument
    for _k in swindex.keys():
        # Get subset stream
        _st_src = st_dict[_k]
        # Get subset swindex
        _swindex = swindex[_k]
        # Get subset of predictions
        _preds = preds[_swindex[0][0]:_swindex[0][1], :, :]
        # Make substack
        _stack = ml.restructure_predictions(_preds, _swindex[1], model)
        # Convert substack to stream
        _stream = ml.stack2stream(_stack, _st_src, model,
                                  mod_wt_code=mod_wt_code,
                                  pred_codes=pred_codes)
        # Update pred_streams
        pred_stream += _stream

    return pred_stream


## FULL RUNTHROUGH ##
times = [time()]
print(f'starting processing {times[0]}')
st = _get_testdata(ROOT='data_2min')
times.append(time())
print(f'data loaded: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

model, device = _get_model()
times.append(time())
print(f'model loaded: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

st_dict = _get_unique_NSBI_from_stream(st)
times.append(time())
print(f'data split by NSBI: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

st_dict = _resample_testdata(st_dict)
times.append(time())
print(f'data resampled: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

windows, swindex = _build_windowed_array(st_dict, model)
times.append(time())
print(f'data windowed: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

windows = _apply_taper(windows)
times.append(time())
print(f'taper applied: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

pred = _run_prediction(windows, model, device)
times.append(time())
print(f'prediction complete: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

pred_st = _reassemble_multistation(pred, swindex, model, st_dict)
times.append(time())
print(f'output streams composed: {times[-1] - times[-2]: .2f} sec (elapsed: {times[-1] - times[0]: .2f} sec)')

