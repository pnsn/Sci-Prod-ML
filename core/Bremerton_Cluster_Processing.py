"""
:module: Bremerton_Cluster_Processing.py
:purpose: This script applies SeisBench prediction workflows on data from the 2017 Bremerton earthquake cluster
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network (PNSN)


"""
import os
import sys
from seisbench.models import EQTransformer, PhaseNet
from glob import glob
from obspy import read
from tqdm import tqdm
sys.path.append(os.path.join('..'))
import util.translate as ut
import util.preprocess as up

# Define data directory absolute path
DATA_ROOT = '/Volumes/LaCie/PNW_Store_Local'
# Define output directory path
OUT_ROOT = os.path.join('/Volumes','LaCie','PNW_Store_Local','Bremerton_Cluster')
# Get file list from disk
flist = glob(os.path.join(DATA_ROOT,'PNW2017','*','2017','*','*'))
flist.sort()

# Load models
model_EQT = EQTransformer.from_pretrained('stead')
model_PN = PhaseNet.from_pretrained('stead')

ML_models = [model_EQT, model_PN]
# Load an example waveform
st = read(flist[1])

breakpoint()

for f_ in tqdm(flist[:3]):
    # Read in data
    st = read(f_)

    for model in ML_models:
        ann_ST = model.annotate(st)
        ann_ST = sb_pred_st2sf_st(ann_ST)

    
    annPN = model_PN.annotate(st)
    PN_st = up.sb_pred_st2sf_st(annPN)


        


# st_ann = model.annotate(st)