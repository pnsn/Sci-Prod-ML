"""
:module:    Accelerometer_Processing_Tests.py
:purpose:   Use station UW.GNW BH? and HN? data for catalog events
            during the time window May 10-20 2017 (Bremerton EQS)
            to investigate the effects of pre-processing steps on
            EQTransformer prediction performance 

"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from obspy.clients.fdsn import Client
from libcomcat.search import search
sys.path.append(os.path.join('..','..'))
import ml_prediction.core.classes.EventMiniDB as qve

# For DEV only
from importlib import reload

# Define usgs-libcomcat event search parameters
# Bounding times
TS = datetime(2017, 5, 10)
TE = datetime(2017, 5, 20, 23, 59, 59)
# Centroid location of the Bremerton EQS (Earthquake Sequence) & Approximate Radius
lat0, lon0 = 47.5828, -122.57841
max_radius_km = 50.

OUTPUT_DATA_DIR = os.path.join('.', 'data')

# Initialize waveform client
client = Client("IRIS")

# Define kwargs for VerboseEvent.get_waveforms()
gkw = {'client': client,
       'net_include': ['UW'],
       'sta_include': ['GNW'],
       'pad_sec': [6000./40., 6000./40.],
       'attach_response': True,
       'pad_level': 'wide'}


### START OF PROCESSING SECTION ###

# Conduct usgs-libcomcat query for events
print('=== Fetching SummaryEvents from ComCat ===')
events = search(starttime=TS, endtime=TE,
                longitude=lon0, latitude=lat0,
                maxradiuskm=max_radius_km)


# Convert into VerboseCatalog & load ComCat picks
print('=== Converting ComCat Events Into VerboseCatalog With Phases')
vcat = qve.VerboseCatalog(events, load_phase= True, load_history=False)

# Conduct waveform pull for UW.GNW (or whatever you put into gkw)
for _e in tqdm(vcat.events):
    # Get waveforms
    _e.get_waveforms(**gkw)
    # Populate inventory for channels present in _e.waveforms
    _e.populate_inventory(client, level='response', source='waveforms')
    # Write collection to disk
    _e.to_layered_directory(location=OUTPUT_DATA_DIR, exclude=[])