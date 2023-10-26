"""
:module: prediction_methods.py
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:purpose: Provide methods for running detection/picking predictions on arbirary length,
          somewhat pre-processed 3-C seismic data arrays*.

*Assumed pre-processing:
    Data form a (3,n) numpy.ndarray with:
            [0,:] as vertical data
            [1,:] as north / horizontal channel 1 data
            [2,:] as east / horizontal channel 2 data
    
    Data only contain finite sample values (i.e., no NaN or inf)

"""
import os
import time
import torch
import numpy as np
import seisbench.models as sbm


def initialize_EQT_model(sbm_model=sbm.EQTransformer.from_pretrained('pnw'),
                         nwin=6000, nstep=1800, nlb=500, nrb=500,
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
    :param nwin: [int]
            Number of datapoints the model expects for prediction inputs
    :param nstep: [int]
            Number of datapoints to advance an annotation window by for each
            prediction
    :param nlb: [int]
            Number of datapoints to blind on the left side of prediction output
            vectors
    :param nrb: [int]
            Number of datapoints oto blind on the right side of predction output
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
    model._annotate_args['overlap'] = ('Overlap between prediction windows in samples\
                                        (only for window prediction models)', nstep)
    model._annotate_args['blinding'] = ('Number of prediction samples to discard on \
                                         each side of aeach window prediction', (nlb, nrb))
    # Place model into evaluate mode
    model.eval();
    return model, device


def prepare_windows(data, model, detrend_method='mean', norm_method='max', verbose=True):
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
                windowed data expected value calculation method for simple detrending
                of each segment `m` and channel `c`
                    'median': subtract numpy.median(x)
                        Higher compute cost, but more resistant to outliers
                (D) 'mean' or 'average': subtract numpy.mean(x)
                        Lower compute cost, but more sensitive to outliers
                Benchmarking shows that 'mean' is about 5x faster.
    :param norm_method: [str]
                windowed data normalization done for each segment `m` and channel `c`
                (D) 'max': divide by numpy.max(numpy.abs(x)) 
                    'std': divide by numpy.std(x)
                normalization method is dependent on the model being used
                'max' is slightly more computationally efficient than 'std'

    :param verbose: [bool]
                Should print() messages be executed?

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
    if verbose:
        tick = time.time()
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

    if verbose:
        tock = time.time()
        mm =   '=== Applied processing ===\n'
        mm += f'  detrending: {detrend_method}\n'
        mm += f'  normalization: {norm_method}\n'
        mm += f'  dropped trailing datapoints: {npts - _i1}\n'
        mm += f'  elapsed time: {tock - tick:.2f} sec'
        print(mm)
    return windows, windex, _i1


def taper(windows, ntaper=6):
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
    windows[:, :, -ntaper:] *= taper

    return windows


def run_prediction(windows_tt, model, device):
    """
    Run predictions on windowed data and return a prediction
    vector of the same scale
    :: INPUTS ::
    :param windows_tt: [(m, 3, n) torch.Tensor]
                preprocessed data windows
    :param model: [seisbench.models.<model_subclass>]
                model object with which to conduct prediction
    :param device: [torch.device]
                hardware specification on which to conduct prediction
    """
    # Preallocate space for predictions
    preds = np.zeros(shape=windows_tt.shape, dtype=np.float32)
    # Do prediction
    _torch_preds = model(windows_tt.to(device))
    # Unpack predictions onto CPU-hosted memory as numpy arrays
    for _i, _p in enumerate(_torch_preds):
        preds[:, _i, :] = _p.detach().cpu().numpy()
    
    return preds


# def run_seisbench_pred(stream, model):


def restructure_predictions(preds, windex, model, merge_method=np.nanmax):
    """
    Reshape windowed predictions back into an array of prediction vectors,
    applying blinding to prediction windows and merging overlapping samples
    with a specified `merge_method`.

    :: INPUTS ::
    :param preds: [(nwin, cpred, mdata) numpy.ndarray]
                array of prediction outputs from the ML model
                (Hint: this can also be done with windowed data!)
    :param windex: [numpy.ndarray]
                vector of window start indices, produced by `prepare_windows()`
    :param model: [seisbench.models.<model_subclass>]
                model object used for predictions. Used to recover blinding and
                window step size
    :param merge_method: [numpy.nanmax, numpy.nanmean, or similar]
                method to use for merging overlapping samples.
                MUST have handling for np.nan entries to prevent bleed of blinded
                samples.
    
    :: OUTPUT ::
    :return stack: [(cpred, npts) numpy.ndarray]
                Array composed of row-vectors for each predicted parameter with 
                blinded samples on either end of the entire section reassigned as
                numpy.nan values.
    """
    # Get scale of prediction
    nwind, cpred, mdata = preds.shape
    # Recover moving window & blinding parameters from model
    nstep = model._annotate_args['overlap'][-1]
    nlb, nrb = model._annotate_args['blinding'][-1]
    # Calculate numer of points for each prediction curve
    npts = mdata + nstep*(nwind - 1)
    # Preallocate space for stacked results
    stack = np.full(shape=(cpred, npts), fill_value=np.nan, dtype=np.float32)

    # Iterate across windows
    for _i, _wi in enumerate(windex):
        # Copy windowed prediction 
        _data = preds[_i, :, :].copy()
        # Apply blinding
        _data[:, :nlb] = np.nan
        _data[:, -nrb:] = np.nan
        # Attach predictions to stack
        stack[:, _wi:_wi+mdata] = merge_method([stack[:, _wi:_wi+mdata], _data], axis=0)
     
    return stack


def stream2array(stream, band='?', inst='N', chanorder='ZNE'):
    """
    Convert an arbitrary stream into an ordered numpy.ndarray `data`
    Uses the v1 conversion method tested in `benchmark_stream2array.py`
    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
    :param band: [character]
                Character to use for the SEED band code on desired traces
    :param inst: [character]
                Character to use for the SEED instrument code on desired traces
    :param chanorder: [string or list of characters]
                Sequence of SEED channel codes for data vector sequencing
    :: OUTPUT ::
    :return data: [(3, n) numpy.ndarray]
                float32 formatted numpy array of data
    """
    data = np.array([stream.select(channel=f'{band}{inst}{x}')[0].data for x in chanorder], dtype=np.float32)
    return data


def stack2stream(stack, source_stream, model, mod_wt_code='EW', pred_codes=['D','P','S'], trim=True):
    """
    Convert a prediction stack (or windowed data from `prepare_windows()`) into a 
    formatted obspy.core.stream.Stream, using the metadata from the initial stream 
    the data were sourced from and model metadata.

    :: INPUTS ::
    :param stack: [(3, m) numpy.ndarray] 
                    Array of re-assembled prediction values
    :param source_stream: [3-element obspy.core.stream.Stream]
                    Initial 3-C stream from which data were sourced for prediction
    :param model: [seisbench.models.<model_subclass>]
                    Model used for prediction
    :param mod_wt_code: [2-character str]
                    Model and weight codes to assign to the Trace.stats.location attribute
                    
                    Model Codes, e.g.
                    E = EQTransformer (Mousavi et al., 2020; Woollam et al., 2022)
                    P = PhaseNet (Zhu & Beroza, 2018)
                    G = GPD    (Ross et al., 2018)

                    Weight Codes, e.g.,
                    D = STEAD (Mousavi et al., 2019)
                    W = PNW (Ni et al., 2023)

    :param pred_codes: [list of char]
                    Codes for prediction channel that take the place of the 
                    SEED component code on the Trace.stats.channel attribute.
                    
                    e.g.,
                    P = P probability
                    S = S probability
                    D = Detection probability (EQTransformer specific)
                    N = Noise probability (PhaseNet specific)
    
                    Note: The band and insturment codes for the SEED channel
                    name are coppied from the source_stream
    
    :param trim: [bool]
                    should the blinded samples be trimmed off the output stream?
    
    :: OUTPUT ::
    :return pred_stream: [obspy.core.stream.Stream]
                    Output stream containing prediction traces
    """
    # Get blinding samples
    nlb, nrb = model._annotate_args['blinding'][-1]
    # Copy input stream to host prediction traces
    pred_stream = source_stream.copy()[:stack.shape[0]]
    for _i, _tr in enumerate(pred_stream):
        # Get sample spacing
        _dt = _tr.stats.delta
        # Get blinding edge UTCDateTimes 
        _ts = _tr.stats.starttime + _dt*nlb
        _te = _tr.stats.endtime - _dt*nrb
        # Change channel code to prediction code
        _tr.stats.channel = _tr.stats.channel[:2] + pred_codes[_i]
        # Change location to model & training weight code
        _tr.stats.location = mod_wt_code
        # Reassign data
        _tr.data = stack[_i,:]
        # Trim off first and last blinding
        if trim:
            _tr = _tr.trim(starttime=_ts, endtime=_te)
        
    return pred_stream
