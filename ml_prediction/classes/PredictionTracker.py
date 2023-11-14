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
from obspy import Stream, Trace
sys.path.append('..')
import core.preprocessing as prep
import core.postprocessing as post


# class PredStreamTracker(Stream):

#     def __init__(self, prep_stream=None, model=None):
#         # Bring in everything obspy.core.stream.Stream has to offer!
#         super().__init__()
#         self.



class Tracker:
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
        self._last_windowed_stream = Stream()
        self.prep_stream = Stream()
        # Indices
        self.last_timestamp = None
        self.last_
        # Numpy ndarray attributes
        self.windows = None
        self._last_window_starttime = None
        self.predictions = None
        self._last_pred = None

        if isinstance(raw_stream, (Stream, Trace)):
            if isinstance(raw_stream, Trace):
                self.raw_stream = Stream(self.raw_stream)
            if len(self.raw_stream) > 0:
                SNCL_list = []
                for _tr in self.raw_stream:
                    net = self.raw_stream[0].stats.network
                    sta = self.raw_stream[0].stats.station
                    loc = self.raw_stream[0].stats.location
                    cha = f'{self.raw_stream[0].stats.channel[:2]}?'
                    if (sta, net, cha, loc) not in SNCL_list:
                        SNCL_list.append((sta, net, cha, loc))
                if len(SNCL_list) == 1:
                    self.stats = dict(zip(['station','network','channel','location'],
                                          [sta, net, cha, loc]))
                elif len(SNCL_list) == 0:
                    pass
                else:
                    print(f'Multiple ({len(SNCL_list)}) SNCL combinations detected!')
                    
                    
    
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
        Return a deepcopy of the InstrumentPredictionTracker
        """
        return deepcopy(self)
    

    def order_raw(self, order='Z3N1E2', merge_kwargs={'method':1}):
        """
        Merge and order traces into a vertical, horizontal1, horizontal2 order,
        with specified merge_kwargs for obspy.stream.Stream.merge(). 
        
        Acts in-place on self.raw_stream
        
        :: INPUTS ::
        :param order: [str]
                        Order of channel codes. Generally shouldn't change
                        from the default
        :param merge_kwargs: [dict]
                        key-word-arguments for obspy.stream.Stream.merge()

        :: OUTPUT ::
        No output, results in merged, ordered self.raw_stream if successful
        """
        # Create holder stream
        _stream = Stream()
        # Iterate across channel codes from order       
        for _c in order:
            # Subset data and merge from raw_stream
            _st = self.raw_stream.select(channel=f'??{_c}').copy().merge(**merge_kwargs)
            # if subset stream is now one stream, add to holder stream 
            if len(_st) == 1:
                _stream += _st
        # Overwrite raw
        self.raw_stream = _stream
        # Add information to log
        if 'raw' in list(self.log.keys()):
            self.log['raw'].append('order')
        else:
            self.log.update({'raw':['order']})
        return None


    def __to_prep__(self):
        """
        Convenience method for copying self.raw_stream to 
        self.prep_stream
        """
        self.prep_stream = self.raw_stream.copy()


    def filter(self, ftype, from_raw=True, **kwargs):
        """
        Filter data using obspy.core.stream.Stream.filter() class-method
        with added option of data source

        :: INPUTS ::
        :param ftype: [string]
                `type` argument for obspy.core.stream.Stream.filter()
        :param from_raw: [bool]
                True = copy self.raw_stream and filter the copy
                False = filter self.prep_stream in-place
        :param **kwargs: [kwargs]
                kwargs to pass to obspy.core.stream.Stream.filter()
        
        """
        if from_raw:
            _st = self.raw_stream.copy()
        else:
            _st = self.prep_stream
        self.prep_stream = _st.filter(ftype, **kwargs)
        if 'prep' in list(self.log.keys()):
            self.log['prep'].append('filter')
        else:
            self.log.update({'prep':['filter']})

        

    def homogenize(self, samp_rate,
                   interp_kwargs={'method': 'weighted_average_slopes','no_filter': False},
                   resamp_kwargs={'window': 'hann', 'no_filter': False},
                   trim_method='max', from_raw=False):
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
        :param trim_method: [string] or [None]
                        None: apply no padding
                        'max': trim to earliest starttime and latest endtime
                               contained in source stream
                        'min': trim to latest starttime and earliest endtime
                               contained in source stream
                        'med': trim to median starttime and endtime contained
                               in source stream
        :param from_raw: [BOOL]
                        Should data be sourced from self.raw_stream?
                        False --> source from self.prep_stream
        :: OUTPUT ::
        No outputs. Updated data are written to self.prep_stream
        """
        if from_raw:
            _st = self.raw_stream.copy()
        elif ~from_raw and isinstance(self.prep_stream, Stream):
            _st = self.prep_stream
        else:
            print(f'self.prep_stream is {type(self.prep_stream)} -- invalid')
            return None
        ts_list = []
        te_list = []
        for _tr in _st:
            ts_list.append(_tr.stats.starttime.timestamp)
            te_list.append(_tr.stats.endtime.timestamp)
            if _tr.stats.sampling_rate < samp_rate:
                _tr.interpolate(samp_rate, **interp_kwargs)
            elif _tr.stats.sampling_rate > samp_rate:
                _tr.resample(samp_rate, **resamp_kwargs)
        
        if trim_method is not None:
            # Minimum valid window
            if trim_method in ['min','Min','MIN','minimum','Minimum','MINIMUM']:
                # Save trimmed segment without padding to self._last_raw_stream
                self._last_raw_stream += _st.trim(starttime=UTCDateTime(np.nanmin(te_list)))
                # Trim segment
                _st = _st.trim(starttime=UTCDateTime(np.nanmax(ts_list)),
                                endtime=UTCDateTime(np.nanmin(te_list)),
                                pad=True)
            # Maximum window
            elif trim_method in ['max','Max','MAX','maximum','Maximum','MAXIMUM']:
                # Trim and pad
                _st = _st.trim(starttime=UTCDateTime(np.nanmin(ts_list)),
                                endtime=UTCDateTime(np.nanmax(te_list)),
                                pad=True)
            # Median-defined window
            elif trim_method in ['med','Med','MED','median','Median','MEDIAN']:
                # Save trimmed segment without padding to self._last_raw_stream
                self._last_raw_stream += _st.trim(starttime=UTCDateTime(np.nanmedian(te_list)))
                # Trim segment
                _st = _st.trim(starttime=UTCDateTime(np.nanmedian(ts_list)),
                                endtie=UTCDateTime(np.nanmedian(te_list)),
                                pad=True)
            else:
                print(f'Invalid value for trim_method {trim_method}')
        # Update prep_stream
        self.prep_stream = _st
        return None



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
