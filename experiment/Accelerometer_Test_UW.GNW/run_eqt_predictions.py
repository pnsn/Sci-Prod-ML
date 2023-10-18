import os
import sys
from glob import glob
from tqdm import tqdm
from seisbench.models import EQTransformer, WaveformModel
sys.path.append(os.path.join('..','..'))
import query.verboseevent as qve
from util.preprocess import split_streams, order_traces
from util.translate import relabel_annotations

# Load Event Data
event_archives = glob(os.path.join('.','data','*'))
event_archives.sort()

ve_list = []
for f_ in tqdm(event_archives):
    _ve = qve.VerboseEvent()
    _ve.from_layered_directory(f_)
    ve_list.append(_ve)

vcat = qve.VerboseCatalog(verbose_catalog_list=ve_list)

# Start with high magnitude events (do 5-ish)
eve_inds = vcat.summary.sort_values('Magnitude', ascending=False).index[:5]

# Instantiate the EQTransformer model(s)
model_pnw = EQTransformer.from_pretrained('pnw')
model_ste = EQTransformer.from_pretrained('stead')
# Create dictionary keyed with training data codes
mod_dict = {'D': model_ste, 'W': model_pnw}

# Create start of directory structure
try:
    os.mkdir(os.path.join('.','processed_data'))
except FileExistsError:
    pass

breakpoint()
for _ei in eve_inds:
    # Get subset VerboseEvent
    _ve = vcat.events[_ei]
    # Create event-level subdirecotory
    try:
        os.mkdir(os.path.join('.','processed_data',_ve.event.id))
    except FileExistsError:
        pass
    # Step 1 split event data by instrument
    streams = split_streams(_ve.waveforms)

    for _key in streams.keys():
        pass


