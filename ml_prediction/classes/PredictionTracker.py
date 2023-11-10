"""
:module: PredictionTracker.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT (2023)
:purpose: This module contains the InstrumentPredictionTracker and PredictioTracker 
          class used to wrap data ingestion, pre-processing, ML prediction, and post
          processing while keeping track of 
"""
import os
import sys
import torch
import numpy as np
from copy import deepcopy
from obspy import Stream
sys.path.append('..')
import core.preprocessing as prep
import core.postprocessing as post


class InstrumentPredictionTracker:
    """
    This class provides a structured pre-processing and post-processing object for 
    waveform data and continuous prediction outputs from a machine learning model
    at the granularity of a single seismic instrument: i.e., 1-C or 3-C data
    """



    def __init__(self, raw_stream=None, model=None, winlen=6000, winstep=3000, resamp_rate=100):
        # Metadata attributes
        self.stats = None
        # Stream attributes
        self.raw_stream = raw_stream
        self.last_windowed_raw = None
        self.prep_stream = None
        self.pred_stream = None
        # Model/hyper-parameter attributes
        self.model = model
        self.winlen = winlen
        self.winstep = winstep
        self.resamp_rate = resamp_rate
        self.nwinds = None
        self.processing_log = []
        # Numpy ndarray attributes
        self.windows = None
        self.last_window = None
        self.preds = None
        self.last_pred = None

    
    def __repr__(self):
        repr =  f'=== windowing ===\n'
        repr += f'winlen: {self.winlen} samples\n'
        repr += f'winstep: {self.winstep} samples\n'
        repr += f'=== raw stream ===\n{self.raw_stream}\n'
        repr += f'=== pre-processeed stream ===\n{self.prep_stream}\n'
        repr += f'=== windows ===\nshape: {np.shape(self.windows)}\n'
        repr += f'=== last window ===\n{type(self.last_window)}\n'
        repr += f'=== model ===\n{self.model}\n'
        repr += f'=== predictions ===\nshape: {np.shape(self.preds)}\n'
        repr += f'=== last pred ===\n{type(self.last_pred.shape)}'
        return repr


    def copy(self):
        """
        Return a deepcopy of the PredictionTracker
        """
        return deepcopy(self)
    

    def group(self, from_raw=True, order='Z3N1E2', kwargs={'method':1}):
        """
        Merge and order traces into a vertical, horizontal1, horizontal2 order,
        with specified kwargs for obspy.stream.Stream.merge() and put streams
        writing to self.prep_stream.
        
        :: INPUTS ::
        :param from_raw: [bool] 
                        Should the method use the self.raw_stream data as 
                        an input?
                            True  = Try to use self.raw_stream
                            False = Try to use self.prep_stream
        :param order: [str]
                        Order of channel codes. Generally shouldn't change
                        from the default
        :param kwargs: [dict]
                        key-word-arguments for obspy.stream.Stream.merge()
        
        :: OUTPUT ::
        No output, results in populating self.prep_stream if successful
        """
        if from_raw:
            self.prep_stream = self.raw_stream.merge(**kwargs)
        elif ~from_raw and isinstance(self.prep_stream, Stream):
            self.prep_stream = self.prep_stream.merge(**kwargs)
        else:
            print(f'self.prep_stream is {type(self.prep_stream)} -- invalid')
            return None
        
        _stream = Stream()

        for _c in order:
            _st = self.prep_stream.select(channel=f'??{_c}')
            if len(_st) >= 1:
                _stream += _st
        self.prep_stream = _stream
        return None


    def resample(self, samp_rate,
                 interp_kwargs={'method': 'weighted_average_slopes','no_filter': False},
                 resamp_kwargs={'window': 'hann', 'no_filter': False},
                 from_raw=False):
        """
        Resample waveform data using ObsPy methods for a specified
        target `samp_rate` using Trace.interpolate() to upsample and
        Trace.resample() to downsample.

        :: INPUTS ::
        :param samp_rate: [int-like] target sampling rate
        :param interp_kwargs: [dict]
                        key-word-arguments to pass to Trace.interpolate()
                        for upsampling
        :param resamp_kwargs: [dict]
                        key-word-arguments to pass to Trace.resample()
                        for downsampling
        :param from_raw: [BOOL]
                        Should data be sourced from self.raw_stream?
                        False --> source from self.prep_stream
        :: OUTPUT ::
        No explicit outputs. Updated data are written to self.prep_stream
        """
        if from_raw:
            _st = self.raw_stream.copy()
        elif ~from_raw and isinstance(self.prep_stream, Stream):
            _st = self.prep_stream
        else:
            print(f'self.prep_stream is {type(self.prep_stream)} -- invalid')
            return None

        for _tr in _st:
            if _tr.stats.sampling_rate < samp_rate:
                _tr.interpolate(samp_rate, **interp_kwargs)
            elif _tr.stats.sampling_rate > samp_rate:
                _tr.resample(samp_rate, **resamp_kwargs)
        
        self.prep_stream = _st
        return None

    def 


    def __fill_masked_trace_data__(self, fill_value, dtype = np.float32, from_raw=False):
        """
        If a given trace is masked, fill masked values with a specified value
        and return the data array of that trace

        :: INPUTS ::
        :param fill_value: [dtype-like]
                    Value to fill in places where 
        :param dtype: [numpy.float32], [float-like], or [None]
                    output data format specification
                    If specified, must be compliant with the numpy.ndarray kwarg `dtype`.
                    If None, do not enforce a set dtype for output
                    Default is numpy.float32
        :param from_raw: [bool]
                    Pull initial data from raw?
                    False = pull data from self.prep_stream

        :: OUTPUTS ::
        No direct outputs. Result is written to self.prep_stream
        """
        if from_raw:
            _st = self.raw_stream.copy()
        else:
            _st = self.pred_stream
        
        for _tr in _st:
            if np.ma.
        

    def output_windows(self, astensor=True):
        """
        Return self.windows with the option as formatting
        as a torch.Tensor and write last last window to 
        the self.last_window attribute

        :: INPUTS ::
        :param astensor: [BOOL]
                should the output be a torch.Tensor?
                False = output numpy.ndarray
        :: OUTPUT ::
        :return self.windows: 
        """
        if isinstance(self.windows, np.array)
    
            self.last_window = self.windows[-1, :, :]
            if astensor:
                return torch.Tensor(self.windows)
            else:
                return self.windows
        else:
            return None

    ################################
    # POSTPROCESSING CLASS-METHODS #
    ################################

    def ingest_preds(self, preds, merge_method='max'):
        """
        Receive numpy.ndarray of windowed predictions and merge into 
        """
        _widx = self.windex
        _mod = self.model
        # Conduct merge from predictions to stack
        stack = post._restructure_predictions(
            preds,
            _widx,
            _mod,
            merge_method=merge_method
        )

        # 

        # Get index of last prediction (from a temporal standpoint)
        _lw = np.argmax(windex)
        # (over)write self.last_pred 
        self.last_pred = preds[_lw, :, :]


    def preps_to_disk(self, loc, fmt='MSEED'):
        """
        Write pre-procesed stream to disk
        """

    def preds_to_disk(self, loc, fmt='MSEED'):
        """
        Write prediction traces to disk
        """

    #########################
    # PRIVATE CLASS-METHODS #
    #########################
    
    def __update_lasts_prep__(self):
        """
        Update self.last_window_raw and self.last_window
        """

    def __update_last_window_raw__(self):
        """

        """

    def __update_last_window__(self):
    
    def __update_last_pred__(self):


    def __flush_waveforms__(self):
        """
        Clear out self.raw_waveforms and move
        self.last_window_raw to self.raw_waveforms
        to preserve 
        """
        if isinstance(self.last_window_raw, Stream):

    def __flush_predictions__(self):


    
        

def default_pp_kwargs():
    interp_kwargs = {'method': 'average_weighted_slopes', 'no_filter': False}
    resamp_kwargs = {'window': 'hann', 'no_filter': False}



class PredictionTracker:

    def __init__(self, instruments=None, model=None, device=None, waveform_pp_kwargs=default_pp_kwargs(), ):
        
        self.instruments = dict(zip([f'{x.stats.network}.{x.stats.station}.{x.stats.location}.{x.stats.bandinst}' for x in stations],
                                    instruments))
        self.model = model
        self.device = device
        self.windows = None
        self.swindex = None
        self.resample_kwargs = resample_kwargs
        self.windowing_kwargs = windowing_kwargs

    def __repr__(self):
        return list(self.stations.keys())


    # def __add__(self, other):
    #     """
    #     Add two PredictionTrackers or a PredictionTracker and 
    #     an InstrumentPredictionTracker

    #     :param other: [PredictionTracker] or [InstrumentPredictionTracker]
    #     """
    #     if isinstance(other, InstrumentPredictionTracker):
    #         other = 

    def copy(self):
        """
        Create a deepcopy of the PredictionTracker object
        """
        return deepcopy(self)


    def run_preprocessing(self):
        for _k in self.stations.keys():
            self.stations[_k].


    def aggregate_windows(self):

    

    def batch_prediction(self):


    def disaggregate_preds(self):
        """
        Disassemble prediction numpy.ndarray
        into component parts and return them
        to their respective InstrumentPredicionTracker's
        contained within self.instruments
        """


    def ingest_new(self, stream):
        """
        Ingest new waveform data in obspy.core.stream.Stream format
        and update data holdings within each relevant member of 
        """
    

    def preds_to_disk()

    def runall(self):
