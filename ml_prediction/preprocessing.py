"""
:module: ml_prediction.preprocessing
:auth: Nathan T. Stevens
:email: ntsteven at uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT License (2023)
:purpose: A series of pre-processing methods for converting an
          arbitrary obspy.core.stream.Stream of waveform data
          into a set of preprocessed data windows ready to 
          convert into a PyTorch Tensor for prediction.

          Methods herein use a `model` object that is currently
          sourced from the SeisBench API and we provide an example
          model and device loading script for completeness
          
:attribution:
          Approaches herein build on code developed by Yiyu Ni
          for the ELEP project and conversations with Yiyu:
          https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb
"""
import numpy as np
from tqdm import tqdm
from obspy import UTCDateTime
import torch
import seisbench.models as sbm


####################################
# MODEL LOADING CONVENIENCE METHOD #
####################################


def initialize_EQT_model(sbm_model=sbm.EQTransformer.from_pretrained('pnw'),
                         nstep=1800, nlb=500, nrb=500,
                         device=torch.device('cpu'),
                         filter_args_overwrite=False,
                         filter_kwargs_overwrite=False):
    """
    Convenience method documenting a "standard" model for PNSN
    detection/phase picking prediction using EQTransformer

    :: INPUTS ::
    :param sbm_model: [seisbench.model.<model_subclass>]
            specified PyTorch model built with the SeisBench
            WaveformModel abstraction and weights loaded
    :param nstep: [int]
            Number of datapoints to advance an annotation window by for each
            prediction
    :param nlb: [int]
            Number of datapoints to blind on the left side of prediction output
            vectors
    :param nrb: [int]
            Number of datapoints to blind on the right side of predction output
            vectors
    :param filter_args_overwrite: [bool], [None], or [dict]
            Arguments for the optional prefilter built into the sbm_model class
            methods.
            Default `False` preserves the default settings associated with
            the pretrained model arguments
            `None` disables any pre-specified filter (also have to do this to
            filter_kwargs_overwrite)
            A dict can overwrite pre-specified filter arguments. See
            obspy.core.trace.Trace.filter() for formatting.
    :param filter_kwargs_overwrite: [bool], [None], or [dict]
            See documentation for `filter_args_overwrite`

    :: OUTPUT ::
    :return model: initialized seisbench.models.<model_subclass> object placed
                into evaluate mode.

    ** References **
    EQTransformer algorithm -               Mousavi et al. (2020)
    SeisBench/PyTorch EQT implementation -  Woollam et al. (2022)
                                            Münchmeyer et al. (2022)
    'pnw' EQT model weights -               Ni et al. (2023)
    """
    # Define local variable
    model = sbm_model
    # Proivide options for overwriting filter args/kwargs
    if filter_args_overwrite:
        model.filter_args = filter_args_overwrite
    if filter_kwargs_overwrite:
        model.filter_kwargs = filter_kwargs_overwrite
    # Assign what hardware the model is running predictions on
    model.to(device)
    # Append annotate() arguments to model
    model._annotate_args['overlap'] = ('Overlap between prediction windows in\
                                        samples (only for window prediction \
                                        models)', nstep)
    model._annotate_args['blinding'] = ('Number of prediction samples to \
                                        discard on each side of aeach \
                                        window prediction', (nlb, nrb))
    # Place model into evaluate mode
    model.eval()
    return model, device




def prepare_windows_from_stream(stream, model, 
                                fill_value=np.nan,
                                merge_kwargs={'method': 1,
                                              'interpolation_values': 5},
                                resample_rate=100, 
                                interp_kwargs={'method': 'weighted_average_slopes',
                                               'no_filter': False},
                                resamp_kwargs={'window': 'hann',
                                               'no_filter': False},
                                method_1C='ZP',
                                detrend_method='mean',
                                norm_method='max',
                                ntaper=6):
    """
    Wrapper method to convert an arbitrary stream of 3-component (and 1-C)
    seismic data into resampled, detrended, normalized data windows ready to
    convert into PyTorch tensors for prediction purposes.

    Unlike methods provided in the SeisBench API, which tends to focus on
    predicting on a single, continuous 3-C seismic trace (via the annotate()
    class method), this wrapper method converts data from several seismic
    stations into a 3-D numpy.ndarray of windowed, pre-processed data for
    prediction via the PyTorch API directly. This allows for easier
    job-batching control and is better posed for ingesting near-real-time
    streaming data from across a regional seismic array.

    Processing Steps:
    1) Convert stream into a dictionary of streams split by NSLC codes
       with an option to merge traces
        see: _NSLBI_dict_from_stream()
    2) Resample streams to common sampling rate
        see: _resample_NSLBI_dict()
    3) Form data windows of specified length, stride, detrending,
       and normalization and keep track of station-window ownership
        see: _NSLBI_dict_to_windows()
    4) Apply cosine taper to windowed data
        see: _taper()

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
                Stream containing data traces (obspy.core.trace.Trace)
    :param model: [seisbench.models.<model_subclass>]
                PyTorch model wrapped with SeisBench API
    :param fill_value: [nan] or [float]
                Value to fill no-data entries in the event of gappy data
                or unequal channel start times for a given instrument recording
                see subroutine _stream_to_array()
    :param merge_kwargs: [dict]
                key-word-arguments to pass to Stream.merge()
                    via stream_to_NSLBI_dict()
    :param resample_rate: [int]
                sampling rate to unify all data to via homogenize_NSLBI_dict()
    :param interp_kwargs: [dict]
                interpolation kwargs used during data up-sampling
                    via homogenize_NSLBI_dict()
                    see documentation of ObsPy's Trace.interpolate()
                    defaults:
                        method='weighted_average_slopes' (SAC default)
                        no_filter=False (do apply a filter automatically)
                                Note: This shouldn't trigger as interp is used
                                in this case to upsample data
    :param resamp_kwargs: [dict]
                resampling kwargs used during data down-sampling
                    via homogenize_NSLBI_dict()
                    see documentation of ObsPy's Trace.resample()
                    defaults: 
                        window='hann' (ObsPy default)
                        no_filter=False (different from ObsPy default)
                            A lowpass prefilter should be applied to data
                            being downsampled to prevent aliasing. Use
                            in-built routines in this class-method to do so.
    :param method_1C: [str]
                method name for handling 1-Component streams
                    see: _NSLBI_dict_to_windows()
    :param detrend_method: [str]
                method name for detrending windowed data
                    see: _NSLBI_dict_to_windows()
    :param norm_method: [str]
                method name for normalizing windowed data
                    see: _NSLBI_dict_to_windows()
    :param ntaper: [int]
                integer length of cosine taper to apply to each side of every
                data window
    
    :: OUTPUTS ::
    :return NSLBI_dict: [dict of streams]
                This is passed to reassemble_multistation_preds()
                    see full description in _NSLBI_dict_from_stream()
    :return windows: [(m_, c_, n_) numpy.ndarray]
                windowed data preprocessed for conversion into a torch.Tensor()
                i.e., windows_tt = torch.Tensor(windows)
    :return swindex: [dict]
                Station-Window-Index
                a dictionary keyed identically to NSLBI_dict with following tuple elements:
                    Element 0: the first and last indices, _i:_j,  of windows[_i:_j, :, :] that
                               contain data from station _k for _k in NSLBI_dict.keys()
                               i.e., windows' ownership by a given station/instrument
                    Element 1: the indices of continuous waveform data, _d, in stream[:].data[_d]
                               that correspond to the first sample for windows[_i:_j, :, :]
                               i.e., the position of a given window's start in an input data vector (trace)

                tl;dr: this gets passed to reassemble_multistation_preds() 
                        to facilitate data book-keeping
    """
    # Split single stream into dictionary of streams split by
    # unique combinations of Network, Station, BandCode & InstCode
    NSLBI_dict = stream_to_NSLBI_dict(stream, merge_kwargs=merge_kwargs)
    # Conduct homogenization
    NSLBI_dict = homogenize_NSLBI_dict(NSLBI_dict, samp_rate=resample_rate,
                                       interp_kwargs=interp_kwargs,
                                       resamp_kwargs=resamp_kwargs)
    # Form data windows from NSLBI_dict data
    windows, swindex = NSLBI_dict_to_windows(NSLBI_dict, model,
                                             method_1C=method_1C,
                                             detrend_method=detrend_method,
                                             norm_method=norm_method,
                                             fill_value=fill_value)
    # Apply tapering
    windows = taper_windows(windows, ntaper=ntaper)
    return NSLBI_dict, windows, swindex

########################################################
# Supporting methods for prepare_windows_from_stream() #
########################################################


def stream_to_NSLBI_dict(stream,
                         merge_kwargs={'method': 1, 'interpolation_samples': 5},
                         verbose=True):
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
    :param stream: [obspy.core.stream.Stream]
                    Stream object containing traces
    :param merge_kwargs: [dict]
                    Dictionary of key-word-arguments sent to
                    obspy.core.stream.Stream.merge()

    :: OUTPUT ::
    :return NSLBI_dict: [dictionary of Streams]
                    Dictionary of streams keyed by Network.Station.Location.BI?
                    with
                    B = SEED Band Code
                    I = SEED Instrument Code
                    ? = Wildcard SEED Channel Code

    Note: the wildcard used is compliant with Stream.select(channel=...)

    """
    # Get unique Network.Station.Location.BandInst? codes
    NSLBI_list = []
    for _tr in stream.copy():
        _net = _tr.stats.network
        _sta = _tr.stats.station
        _cha = _tr.stats.channel
        _loc = _tr.stats.location
        _cbi = _cha[:2]
        _nslbi = f'{_net}.{_sta}.{_loc}.{_cbi}?'
        if _nslbi not in NSLBI_list:
            NSLBI_list.append(_nslbi)
    # Filter input Stream to split of subset streams
    st_list = []
    for _nslbi in tqdm(NSLBI_list, disable=~verbose):
        select_kwargs = dict(zip(['network', 'station',
                                  'location', 'channel'],
                                 _nslbi.split('.')))
        _stream = stream.copy().select(**select_kwargs)
        if merge_kwargs:
            try:
                _stream = _stream.merge(**merge_kwargs)
            except:
                breakpoint()
        st_list.append(_stream)
    # Zip everything together for output
    NSLBI_dict = dict(zip(NSLBI_list, st_list))
    return NSLBI_dict


def homogenize_NSLBI_dict(NSLBI_dict, samp_rate=100,
                          interp_kwargs={'method': 'weighted_average_slopes',
                                         'no_filter': False},
                          resamp_kwargs={'window': 'hann'},
                          trim_bound='median'):
    """
    Resample and pad data contained in an NSLBI dictionary to have uniform:
    1. sampling rates across NSLBI keyed entries
        accomplished with Stream.interpolate() [upsampling]
                            and 
                          Stream.resample() [downsampling]
    2. starttime and endtime values within each keyed entry
        accomplished with Stream.trim() via the 'trim_bound' kwarg

        
    :: INPUTS ::
    :param NSLBI_dict: [dictionary of obspy.core.stream.Stream]
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
    :param trim_method: [str] or None
                    'med' or 'median':  use the median starttime and endtime
                                        of a given stream as the trim bounds
                                        Defaults .trim( ... , pad=True)
                    'max' or 'maximum': use the earliest starttime and latest
                                        endtime of a given stream as the trim
                                        bounds. Defaults .trim( ... , pad=True)
                    'min' or 'minimum': use the latest starttime and earliest
                                        endtime of a given stream as teh trim
                                        bounds. Defaults .trim( ... , pad=True)
                    None:               Do not apply .trim()
    :: OUTPUT ::
    :return NSLBI_dict: [dictionary of streams]
                    resampled/filtered stream data
                    NOTE: Operations are run IN PLACE, meaning that contents
                    of `NSLBI_dict` are altered. One can use the copy.deepcopy 
                    method to create a copy of a dictionary and its contents.
    
    Gappy data and uneven time bounds are handled via masked numpy arrays housed in obspy.Trace.data:
    e.g., input entry of 
    NSLBI_dict
    {'UW.KDK..EN?: 3 Trace(s) in Stream:
    UW.KDK..ENE  | 2017-05-11T13:45:00.770000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 119924 samples
    UW.KDK..ENN  | 2017-05-11T13:45:00.000000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 120001 samples
    UW.KDK..ENZ  | 2017-05-11T13:45:00.000000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 120001 samples
    'UW.SP2..BH?: 3 Trace(s) in Stream:
    UW.SP2..BHE  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 40.0 Hz, 48001 samples
    UW.SP2..BHN  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 40.0 Hz, 48001 samples
    UW.SP2..BHZ  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 40.0 Hz, 48001 samples
    }

    results in:
    NSLBI_dict
    {'UW.KDK..EN?: 3 Trace(s) in Stream:
    UW.KDK..ENE  | 2017-05-11T13:45:00.770000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 120001 samples (masked)
    UW.KDK..ENN  | 2017-05-11T13:45:00.000000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 120001 samples
    UW.KDK..ENZ  | 2017-05-11T13:45:00.000000Z - 2017-05-11T14:05:00.000000Z | 100.0 Hz, 120001 samples
    'UW.SP2..BH?: 3 Trace(s) in Stream:
    UW.SP2..BHE  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 100.0 Hz, 120001 samples
    UW.SP2..BHN  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 100.0 Hz, 120001 samples
    UW.SP2..BHZ  | 2017-05-11T13:45:00.010000Z - 2017-05-11T14:05:00.010000Z | 100.0 Hz, 120001 samples
    }
    """
    # Iterate across unique NSBI codes
    for _k in NSLBI_dict.keys():
        _st = NSLBI_dict[_k].copy()
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

        # Enforce identical start & stop times, masking added entries
        if trim_bound in ['min', 'minimum', 'med', 'median', 'max', 'maximum']:
            ts_list = [_tr.stats.starttime.timestamp for _tr in _st]
            te_list = [_tr.stats.endtime.timestamp for _tr in _st]
            ts_min, ts_med, ts_max = np.nanmin(ts_list), np.nanmedian(ts_list), np.nanmax(ts_list)
            te_min, te_med, te_max = np.nanmin(te_list), np.nanmedian(te_list), np.nanmax(te_list)
            if trim_bound in ['med','median']:
                _st = _st.trim(starttime=UTCDateTime(ts_med), endtime=UTCDateTime(te_med), pad=True)
            elif trim_bound in ['max', 'maximum']:
                _st = _st.trim(starttime=UTCDateTime(ts_min), endtime=UTCDateTime(te_max), pad=True)
            elif trim_bound == ['min', 'minimum']:
                _st = _st.trim(starttime=UTCDateTime(ts_max), endtime=UTCDateTime(te_min), pad=True)
        else:
            pass
        NSLBI_dict.update({_k: _st})

    return NSLBI_dict


def NSLBI_dict_to_windows(NSLBI_dict, model, method_1C='ZP', detrend_method='mean', norm_method='max', fill_value=np.nan):
    """
    Take an arbitrary set of streams contained in a "NSLBI dictionary"
    and convert their data into a numpy.ndarray scaled for input to a
    PyTorch model. Also provide an indexing array to document index-ownership
    for reorganizing ML prediction outputs.

    This method wraps the following subroutines
    _stream_to_array()
    _array_to_windows()

    :: INPUTS ::
    :param NSLBI_dict: [dictionary of streams]
                    see _NSLBI_dict_from_streams() for structure
    :param model: [seisbench.WaveformModel.<model_subclass>]
                    SeisBench API hosted ML model architecture & weights
    :param method_1C: [str]
                    'ZP':   fill-out windows[_m, 1:, _n] data with 0-values
                            if source Stream is 1-component
                    'DUP':  duplicate windows[_m, 0, _n] data to 
                            windows[_m, 1, _n] and windows[_m, 2, _n] 
                            if source stream is 1-component
    :param detrend_method: [str]
                    'mean' or 'average': remove the windowed mean of data
                    'median':            remove hte windowed median of data
    :param norm_method: [str]
                    'max': use the maximum amplitude of windowed data to normalize
                        e.g., approach used in Ni et al. (2023)
                    'std': use the standard deviation of windowed data to normalize
                        e.g., approach used in Mousavi et al. (2020)
    
    :: OUTPUTS ::
    :return windows: [(m, c, n) numpy.ndarray]
                    array of m-windows, c-channels, and n-samples sampled from
                    the contents of NSLBI_dict as specified by parameters
                    contained within model._annotate_kwargs
    :return swindex: [dictionary of lists]
                    Station-Window-INDEX. Dictionary keyed with NSLBI keys
                    (see stream_to_NSLBI_dict()) with the following two lists
                    of index
                    swindex[<key>][0] = 0-axis indices of the first and last 
                                        windows belonging to <key> in `windows`
                    swindex[<key>][1] = indices of the first sample in each
                                        window belonging to the input stream for
                                        a given NSLBI_dict[<key>]
    """
    swindex = {}
    last_sindex = 0
    for _k in NSLBI_dict.keys():
        # Get subset stream
        _st = NSLBI_dict[_k]
        # Convert into an ordered array (pre-tensor)
        if len(_st) == 3:
            try:
                _data = _stream_to_array(_st, band='?', inst='?',
                                         chanorder='ZNE',
                                         fill_value=fill_value)
            except IndexError:
                _data = _stream_to_array(_st, band='?', inst='?',
                                         chanorder='Z12',
                                         fill_value=fill_value)
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
        args = [_data, model]
        kwargs = {'detrend_method': detrend_method,
                  'norm_method': norm_method}
        _windows, _windex, _last_idx = _array_to_windows(*args,
                                                         **kwargs)

        # Concatenate windows
        if last_sindex == 0:
            windows = _windows
            _sindex = [last_sindex, windows.shape[0]]
            last_sindex = windows.shape[0]
        else:
            windows = np.concatenate([windows, _windows], axis=0)
            _sindex = [last_sindex, windows.shape[0]]
            last_sindex = windows.shape[0]

        swindex.update({_k: (_sindex, _windex)})

    return windows, swindex


#########################################
# Subroutines for NSLBI_dict_to_windows #
#########################################
def _stream_to_array(stream, band='?', inst='N', chanorder='ZNE', fill_value=np.nan):
    """
    Convert an arbitrary stream into an ordered numpy.ndarray `data`
    Uses the v1 conversion method tested in `benchmark_stream2array.py`

    This also provides methods for filling masked element values with a
    specified value

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
    :param band: [character]
                Character to use for the SEED band code on desired traces
    :param inst: [character]
                Character to use for the SEED instrument code on desired traces
    :param chanorder: [string or list of characters]
                Sequence of SEED channel codes for data vector sequencing
    :param fill_value: [nan] or [float] or [int]
                Value to replace mask=True values with to convert
                numpy.ma.core.MaskedArray into a simple numpy.ndarray
    :: OUTPUT ::
    :return data: [(3, n) numpy.ndarray]
                float32 formatted numpy array of data
    """
    # Do sanity check that data are of the right size
    if len(stream) == 3:
        if stream[0].stats.npts == stream[1].stats.npts == stream[2].stats.npts:
            # Checks if any traces are masked
            masked = False
            for _tr in stream:
                masked += np.ma.is_masked(_tr.data)
            # If traces are not masked, use obspy + list comprehension
            if ~masked:
                data = np.array([stream.select(channel=f'{band}{inst}{_x}')[0].data for _x in chanorder], dtype=np.float32)
            # Otherwise, do more nuanced checks & abandon list comprehension for easier user comprehension
            else:
                # Preallocate space for `data`
                data = np.zeros(shape=(3, stream[0].stats.npts), dtype=np.float32)
                # Iterate across channels in specified order
                for _i, _x in enumerate(chanorder):
                    _tr = stream.select(channel=f'{band}{inst}{_x}')[0]
                    # If trace is masked
                    if np.ma.is_masked(_tr.data):
                        # Get data vector
                        _data = _tr.data.data
                        # Apply specified fill value to masked elements
                        _data[_tr.data.mask] = fill_value
                    # Otherwise, just grab the data vector
                    else:
                        _data = _tr.data
                    # Regardless of masked status, pass updated _data to output
                    data[_i, :] = _data
            # Output (3, n)
            return data
        
        # Layered warning/quit messages
        else:
            print(f'Stream elements are not of the same length\
                    [{stream[0].stats.npts}, {stream[1].stats.npts}, \
                    {stream[2].stats.npts}, fix this with pre-processing')
    else:
        print(f'Stream has {len(stream)} elements, must have 3 elements')


def _array_to_windows(data, model, detrend_method='mean', norm_method='max'):
    """
    Do minimal processing to make uniform dimension, windowed, normalized
    data based on the input data array dimensions and the expected feature 
    dimensionality for the input layer of the model.

    :: INPUTS ::
    :param data: [(c, d) numpy.ndarray]
                data array with c-channels and d-data
    :param model: [seisbench.models.<model_subclass>]
                prediction ML model object with the following attributes:
                    nstep = model._annotate_args['overlap']
                    nwin = model.in_samples
    :param detrend_method: [str]
                windowed data expected value calculation method for simple
                detrending of each segment `m` and channel `c`
                    'median': subtract numpy.median(x)
                        Higher compute cost, but more resistant to outliers
                (D) 'mean' or 'average': subtract numpy.mean(x)
                        Lower compute cost, but more sensitive to outliers
                Benchmarking shows that 'mean' is about 5x faster.
    :param norm_method: [str]
                windowed data normalization done for each segment `m` and 
                channel `c`
                (D) 'max': divide by numpy.max(numpy.abs(x))
                    'std': divide by numpy.std(x)
                normalization method is dependent on the model being used
                'max' is slightly more computationally efficient than 'std'

    :: OUTPUTS ::
    :param windows: [(m, c, nwin) numpy.ndarray: numpy.float32]
                data array with c-channels split into m-segments
                witn nwin-data length
    :param windex: [(m,) numpy.ndarray: numpy.int32]
                data array containing the starting indices of each
                window in `windows` inherited from `data`
    :param last_didx: [numpy.int32]
                index of last included sample from `data` along the
                d-axis, which corresponds to values in windows[:,-1,-1]
                data deficit scale can be calculated as:
                    data.shape[1] - last_didx - 1
                    or
                    windex[-1] + nwin - 1
    """
    # Get number of data in each channel
    npts = data.shape[1]
    nwin = model.in_samples
    nstep = model._annotate_args['overlap'][-1]
    
    # Get the number of complete windows
    mseg = (npts - nwin)//nstep + 1

    # Preallocate space for windows
    windows = np.zeros(shape=(mseg, 3, nwin), dtype=np.float32)
    windex = np.zeros(mseg, dtype=np.int32)

    for _s in range(mseg):
        # Get indices of window
        _i0 = _s*nstep
        _i1 = _i0 + nwin

        # Window data
        windows[_s, :, :] = data[:, _i0:_i1]

        # Detrend
        if detrend_method.lower() in ['mean', 'average']:
            windows[_s, :, :] -= np.mean(windows[_s, :, :], axis=-1, keepdims=True)
        elif detrend_method.lower() == 'median':
            windows[_s, :, :] -= np.median(windows[_s, :, :], axis=-1, keepdims=True)

        # Normalize
        if norm_method.lower() == 'max':
            windows[_s, :, :] /= np.max(np.abs(windows[_s, :, :]), axis=-1, keepdims=True)
        elif norm_method.lower() == 'std':
            windows[_s, :, :] /= np.std(windows[_s, :, :], axis=-1, keepdims=True)

        # Add start index to windex
        windex[_s] = _i0

    return windows, windex, _i1

#####################
# Window operations #
#####################


def detrend_windows(windows, method='mean', nan_replace=np.float32(0)):
    """
    Detrend windows based on statistics calculated
    for each window, assuming the 0-axis is the
    array indexing axis

    :: INPUTS ::
    :param windows: [(m, c, n) numpy.ndarray]
                    windowed data arrays with axes
                    0: window_number
                    1: data_component
                    2: data_values
    :param method: [str] or None
                'mean' or 'average' - remove the np.nanmean()
                    value calculated along the 2-axis
                'median' - remove the np.nanmedian()
                    values calculated along the 2-axis
                None - do nothing, return input windows
    :nan_replace: [False] or [numpy.float32] 
                specify a value to replace np.nan values with
                None - keep np.nan values in-place
                float - replace with float value
    :: OUTPUT ::
    :return windows: [(m, c, n) numpy.ndarray]
                in-place operation on input windows
    """
    for _m in range(windows.shape[0]):
        if method in ['mean','average']:
            windows[_m, :, :] -= np.nanmean(windows[_m, :, :],
                                            axis=-1, keepdims=True)
        elif method == 'median':
            windows[_m, :, :] -= np.nanmedian(windows[_m, :, :],
                                              axis=-1, keepdims=True)
        else:
            pass

        if nan_replace is None:
            pass
        elif nan_replace:
            windows[~np.isfinite(windows)] = np.float32(nan_replace)

    return windows


def normalize_windows(windows, method='max'):
    """
    Normalize windows based on statistics calculated
    for each window, assuming the 0-axis is the
    array indexing axis

    :: INPUTS :: 
    :param windows: [(m, c, n) numpy.ndarray]
                    windowed data arrays with axes
                    0: window_number
                    1: data_component
                    2: data_values
    :param method: [str] or None
                'max' - remove the np.nanmax()
                    value calculated along the 2-axis
                'std' - remove the np.nanstd()
                    values calculated along the 2-axis
                None - do nothing, return input windows
    :: OUTPUT ::
    :return windows: [(m, c, n) numpy.ndarray]
                in-place operation on input windows
    """
    for _m in range(windows.shape[0]):
        if method == 'max':
            windows[_m, :, :] /= np.nanmean(windows[_m, :, :],
                                            axis=-1, keepdims=True)
        elif method == 'std':
            windows[_m, :, :] /= np.nanmedian(windows[_m, :, :],
                                              axis=-1, keepdims=True)
        else:
            pass

    return windows


def taper_windows(windows, ntaper=6):
    """
    Apply a cosine (Tukey) taper of a specified
    sample length to either end of a series of windows
    
    :: INPUTS ::
    :param windows: [(m, 3, n) numpy.ndarray]
                    windowed data arrays with axes
                    0: window_number
                    1: data_component
                    2: data_values
    :param ntaper: [int]
                    number of data_value samples
                    on either end 
    :: OUTPUT ::
    :return windows: [(m, 3, n) numpy.ndarray]
                    in-place tapered, windowed data 
    """
    taper = 0.5*(1. + np.cos(np.linspace(np.pi, 2.*np.pi, ntaper)))
    windows[:, :, :ntaper] *= taper
    windows[:, :, -ntaper:] *= taper[::-1]

    return windows
