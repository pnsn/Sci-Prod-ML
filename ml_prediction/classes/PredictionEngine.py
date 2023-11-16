"""
:module: PredictionEngine.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT (2023)
:purpose: This module contains the PredictionEngine class used to house
          Tracker objects, handle data connections to and from EarthWorm
          buffers, parse/generate waveform/pick messages, and orchestrate
          batched (pulsed) ML predictions based on rules set by the user.

          
    TODO:
          Currently this module does not have a developed connection to
          EarthWorm messaging rings, but this may be accomplished with
          subsequent integration with packages like Boritech-Solutions'
          PyEarthworm (Hernandez, 2018) under the GNU Affero General 
          Public License terms attached to their repository:
          https://github.com/Boritech-Solutions/PyEarthworm/tree/master
    NOTE:
          I after initial testing of code compatability, I (N. Stevens) will
          need to  confirm the compatability of the current MIT license
          attached to our code repository and the GNU AGPL license prior to
          publishing 'deployment ready' code on the `main` branch.
"""
import os
import sys
import torch
import numpy as np
from time import time
from copy import deepcopy
from Tracker import Tracker


class PredictionEngine:
    """
    :: ATTRIBUTES ::
    ++ PUBLIC ++
    :attr config:           [dictionary]
    :attr ew_conn:          [!!!PLACEHOLDER!!!]
    :attr model:            [seisbench.models.<model_subclass>]
    self.device             [torch.device]
    self.tracker_list       [list of Trackers]
    -- PRIVATE --
    self._SNCL_directory    [dictionary]
    self._windex            [list of 3-tuples]
    self._pulse_rule        [dictionary]
                            dictionary holding the rules for engine pulsing
    self._pred_output_dir   [string]
    self._pred_output_fstr  [format-string]
    """

    def __init__(self, config, pe_config, ew_conn, model, device, trackers=None):
        self.config = config
        # TODO: Placeholder for abstract class that connects to
        # earthworm wave ring(s) and pick ring(s)
        self.ew_config = ew_conn
        self.model = model
        self.device = device
        self._SNCL_directory = {}
        self.tracker_list = trackers
        self._windex = None
        self._pulse_rule
        # Parse trackers at instantiation
        if trackers is not None:
            if isinstance(self.tracker_list, Tracker):
                self.tracker_list = [self.tracker_list]
            # 
            for _i, _tk in enumerate(self.tracker_list):
                # Overwrite tracker._config:
                _tk._config = self.config
                


# class PredictionEngine:

#     def __init__(self, config, trackers=None, model=None, device=None, waveform_pp_kwargs=default_pp_kwargs(), ):

#         self.trackers = dict(zip([f'{x.stats.network}.{x.stats.station}.{x.stats.location}.{x.stats.bandinst}' for x in stations],
#                                     instruments))
#         self.model = model
#         self.device = device
#         self.tensor = None
#         self.swindex = None
#         self.config = config
#         self.resample_kwargs = resample_kwargs
#         self.windowing_kwargs = windowing_kwargs

#     def __repr__(self):
#         return list(self.stations.keys())


#     # def __add__(self, other):
#     #     """
#     #     Add two PredictionTrackers or a PredictionTracker and
#     #     an InstrumentPredictionTracker

#     #     :param other: [PredictionTracker] or [InstrumentPredictionTracker]
#     #     """
#     #     if isinstance(other, InstrumentPredictionTracker):
#     #         other =

#     def copy(self):
#         """
#         Create a deepcopy of the PredictionTracker object
#         """
#         return deepcopy(self)


#     def run_preprocessing(self):
#         for _k in self.stations.keys():
#             self.stations[_k].


#     def aggregate_windows(self):


#     def batch_prediction(self):


#     def disaggregate_preds(self):
#         """
#         Disassemble prediction numpy.ndarray
#         into component parts and return them
#         to their respective InstrumentPredicionTracker's
#         contained within self.instruments
#         """


#     def ingest_new(self, stream):
#         """
#         Ingest new waveform data in obspy.core.stream.Stream format
#         and update data holdings within each relevant member of
#         """


#     def preds_to_disk()

#     def runall(self):
