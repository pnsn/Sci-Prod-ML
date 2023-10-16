"""
:module: Bremerton_Cluster_Processing.py
:purpose: This script applies SeisBench prediction workflows on data from the 2017 Bremerton earthquake cluster
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network (PNSN)
"""
import os
import sys
from seisbench.models import EQTransformer, WaveformModel
from glob import glob
from obspy import read
from tqdm import tqdm
sys.path.append(os.path.join('..'))
import util.translate as ut
import util.preprocess as up
import pickle


annotate_kwargs = {'parallelism':None}

# Define data directory absolute path
# presently a local copy of Bremerton-proximal station data on N. Stevens' external HDD
DATA_ROOT = '/Volumes/LaCie/PNW_Store_Local'
# Define output directory path
OUT_ROOT = os.path.join('/Volumes', 'LaCie', 'PNW_Store_Local', 'ml_pred_PNW2017')
OUT_FILE_FSTR = os.path.join(OUT_ROOT, '{NET}', '{YEAR}', '{JDAY}', 
                             '{STA}.{NET}.{YEAR}.{JDAY}.{LOC}.{BI_ID}.{FMT}')
# Get file list from disk
flist = glob(os.path.join(DATA_ROOT, 'PNW2017', '*', '2017', '131', '*'))
flist.sort()
print('Preparing to iterate across %d waveform day_volumes'%(len(flist)))
## Load models ##
# This requires saving a pair of files describing the PNW model (pnw.pt.v1 & pnw.json.v1)
# from Ni et al. (2023) in the ~/.seisbench/models/v3/eqtransformer/ directory
# Model provided by Yiyu Ni on 10. Oct 2023. (auth contact: niyiyu@uw.edu)
# Model should be provided with future versions of SeisBench
model_EQT_pnw = EQTransformer.from_pretrained('pnw')

# # These models are standard-shipped with SeisBench
# model_EQT_stead = EQTransformer.from_pretrained('stead')
# model_PN_stead = PhaseNet.from_pretrained('stead')

print('Pretrained ML models loaded')

ML_models = {'EQW': model_EQT_pnw}#, 'EQD': model_EQT_stead, 'PND': model_PN_stead}

## Preprocessing Controls
merge_kwargs = {'method': 1}
## Detection Threshold for initial pick selection
dthresh = 0.7
## Save Format
save_fmt = 'MSEED'

# Iterate through day_volume files
for f_ in tqdm(flist[6:]):
    # Read in data
    stream = read(f_)
    # Merge data
    # stream.merge(**merge_kwargs)
    # Split streams by Band and Instrument codes
    dict_BI_streams = up.split_streams(stream)
    for BI_key in dict_BI_streams.keys():
        BI_stream = dict_BI_streams[BI_key]
        # Iterate across model types
        for model_key in ML_models.keys():
            # Fetch model object from dictionary
            model = ML_models[model_key]
            # Grab model training dataset nickname 
            train_name = model_key

            # Conduct probabalistic prediction
            ann_stream = model.annotate(BI_stream,**annotate_kwargs)
            
            # Get picks from annotations given a particular threshold
            wfm = WaveformModel(model)
            picks = wfm.detections_from_annotations(ann_stream,dthresh)

            # Write picks to two formats
            # Pickle with naming conventions from Ni et al. (2023)
            # pickle.???
            # # Convert into snuffler markers and save
            # for p_ in picks:


            SEED_ann_stream = ut.relabel_annotations(BI_stream,ann_stream,model,model_key[-1])
           
            # Compose output file name
            ofile = OUT_FILE_FSTR.format(NET=SEED_ann_stream[0].stats.network,
                                        YEAR=SEED_ann_stream[0].stats.starttime.year,
                                        JDAY=SEED_ann_stream[0].stats.starttime.julday,
                                        STA=SEED_ann_stream[0].stats.station,
                                        LOC=SEED_ann_stream[0].stats.location,
                                        BI_ID=SEED_ann_stream[0].stats.channel[:2],
                                        FMT=save_fmt)
            # Create directory structure (if not already present)
            try: 
                os.makedirs(os.path.split(ofile)[0])
            except FileExistsError:
                pass
            
            SEED_ann_stream.write(ofile,fmt=save_fmt)