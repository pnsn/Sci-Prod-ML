"""
:module: ml_prediction.postprocessing
:auth: Nathan T. Stevens
:email: ntsteven at uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT License (2023)
:purpose: A series of post-processing methods for converting 
          outputs from a PyTorch prediction on windowed data
          back into the form of the source, arbitrary 
          obspy.core.stream.Stream of waveform data.

          Inputs to methods in this module are described 
          in the following modules:
          ml_prediction.preprocessing
          ml_prediction.prediction
          
:attribution:
          Approaches herein build on code developed by Yiyu Ni
          for the ELEP project and conversations with Yiyu:
          https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb
"""
import numpy as np
from obspy import Stream
from tqdm import tqdm

##########################
# Postprocessing Methods #
##########################


def reassemble_multistation_preds(preds, swindex, model, NSLBI_dict,
                                  mod_wt_code='EW', pred_codes=['D', 'P', 'S'],
                                  merge_method=np.nanmax, trim=True,
                                  tqdm_disable=True):
    """
    Reassemble predictions from PyTorch model from multiple stations
    into an obspy.core.stream.Stream using indexing from pre-processing 
    data objects and the model.

    This wraps subroutines:
    _restructure_predictions()
    _stack_to_stream()

    :: INPUTS ::
    :param preds: [(n, p, m) numpy.ndarray]
                array of n-windows of p-prediction-types with m-modeled values
                output by the PyTorch model prediction
    :param swindex: [dict of tuples]
                dictionary keyed with NSLBI codes that correspond to window
                ownership indices. 
                See: ml_prediction.preprocessing.NSLBI_dict_to_windows()
    :param model: [seisbench.models.<model_subclass>]
                Model object used for prediction
    :param NSLBI_dict: [dict of obspy.core.stream.Stream's]
                Network-Station-Location-Band-Instrument sorted streams of
                pre-processed data used to generate windowed data for
                prediction (See stream_to_NSLBI_dict() for more info)
    :param mod_wt_code: [2-character str]
                Model and weight codes to assign to the
                    Trace.stats.location attribute
                
                Model Codes, e.g.
                E = EQTransformer (Mousavi et al., 2020;
                                    Woollam et al., 2022)
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
    :param merge_method: [method that operates on numpy.ndarrays]
                Method for merging overlapping modeled values for a given 
                prediction type
                Suggested:  np.nanmax (default)
                            np.nanmean
    :param trim: [bool]
                trim blinded samples off the output stream?

    :: OUTPUT ::
    :return pred_stream: [obspy.core.stream.Stream]
                Stream object containing modeled values (predictions)
                referenced to source-data timing leveraging metadata
                contained in entries of NSLBI_dict. 

                NOTE:
                Use of the obspy.Stream class allows for ease of use
                alongside source waveform data for visualization and
                metadata-data coupling.

                Sub-routines in this wrapper script might be used
                as a code-base for faster I/O formatting (e.g., *.npy)
                if metadata are handled in some other clear format in
                your workflow.
"""
    pred_stream = Stream()
    # Iterate across each station/instrument
    for _k in tqdm(swindex.keys(), disable=tqdm_disable):
        # Get subset stream
        _st_src = NSLBI_dict[_k]
        # Get subset swindex
        _swindex = swindex[_k]
        # Get subset of predictions
        _preds = preds[_swindex[0][0]:_swindex[0][1], :, :]
        # Make substack
        _stack = _restructure_predictions(_preds, _swindex[1], model,
                                          merge_method=merge_method)
        # Convert substack to stream
        _stream = _stack_to_stream(_stack, _st_src, model,
                                   mod_wt_code=mod_wt_code,
                                   pred_codes=pred_codes, trim=trim)
        # Update pred_streams
        pred_stream += _stream

    return pred_stream


def _restructure_predictions(preds, windex, model, merge_method=np.nanmax):
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
                MUST have handling for np.nan entries to prevent bleed of
                blinded samples.

    :: OUTPUT ::
    :return stack: [(cpred, npts) numpy.ndarray]
                Array composed of row-vectors for each predicted parameter with
                blinded samples on either end of the entire section reassigned
                as numpy.nan values.
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
    for _i, _w in enumerate(windex):
        # Copy windowed prediction
        _data = preds[_i, :, :].copy()
        # Apply blinding
        _data[:, :nlb] = np.nan
        _data[:, -nrb:] = np.nan
        # Attach predictions to stack
        stack[:, _w:_w+mdata] = merge_method([stack[:, _w:_w+mdata], _data],
                                             axis=0)

    return stack


def _stack_to_stream(stack, source_stream, model, mod_wt_code='EW',
                     pred_codes=['D', 'P', 'S'], trim=True):
    """
    Convert a prediction stack (or windowed data from `prepare_windows()`)
    into a formatted obspy.core.stream.Stream, using the metadata from the
    initial stream the data were sourced from and model metadata.

    :: INPUTS ::
    :param stack: [(3, m) numpy.ndarray]
                    Array of re-assembled prediction values
    :param source_stream: [3-element obspy.core.stream.Stream]
                    Initial 3-C stream from which data were sourced
                        for prediction
    :param model: [seisbench.models.<model_subclass>]
                    Model used for prediction
    :param mod_wt_code: [2-character str]
                    Model and weight codes to assign to the
                        Trace.stats.location attribute
                    
                    Model Codes, e.g.
                    E = EQTransformer (Mousavi et al., 2020;
                                        Woollam et al., 2022)
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
                    trim blinded samples off the output stream?
    
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
        _tr.data = stack[_i, :]
        # Trim off first and last blinding
        if trim:
            _tr = _tr.trim(starttime=_ts, endtime=_te)
    return pred_stream
