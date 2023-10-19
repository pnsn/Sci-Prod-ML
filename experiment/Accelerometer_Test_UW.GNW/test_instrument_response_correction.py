"""
:module: test_instrument_response_correction.py
:auth: Nathan T. Stevens
:email: ntsteven@gmail.com
:org: Pacific Northwest Seismic Network
:purpose: Conduct a simple grid-search of water level values for 
          instrument response deconvolution of broadband (native velocity)
          and strong motion (native acceleration) sensors, identifying
          values for each 

"""


import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from obspy import Stream, UTCDateTime
sys.path.append(os.path.join('..','..'))
import query.verboseevent as qve
import matplotlib.pyplot as plt
from pyrocko import obspy_compat
obspy_compat.plant()

# Load Event Data
event_archives = glob(os.path.join('.','data','uw*'))
event_archives.sort()

# A priori knowledge of indices for 5 largest events
indices_hardset = [22]
ve_list = []
for _i in indices_hardset:
    f_ = event_archives[_i]
    _ve = qve.VerboseEvent()
    _ve.from_layered_directory(f_)
    ve_list.append(_ve)

vcat = qve.VerboseCatalog(verbose_catalog_list=ve_list)

ve = vcat.events[0]
st = ve.waveforms

stEN = st.copy().select(channel='EN?')
stBH = st.copy().select(channel='BH?')

### TEST METHODS OF INSTRUMENT RESPONSE CORRECTION
wl_vect = [1.0,3.0,10.,30.,60.,100.]
st_wl_test_vel = Stream()
for _wl in wl_vect:
    stEN_vel_wl = stEN.copy()
    for _tr in stEN_vel_wl:
        _tr.remove_response(_ve.inventory, output='VEL', water_level=_wl)
        _tr.stats.location = '%03d'%(_wl)
        st_wl_test_vel += _tr

    stBH_vel_wl = stBH.copy()
    for _tr in stBH_vel_wl:
        _tr.remove_response(_ve.inventory, output='VEL', water_level=_wl)
        _tr.stats.location = '%03d'%(_wl)
        st_wl_test_vel += _tr

# # Interactive visualization
# st_wl_test_vel.snuffle(ntracks=len(st_wl_test_vel))

# Resample to 100 Hz and interpolate to ensure identical time indices
# Trim to ensure matching sample counts
temin = UTCDateTime()
tsmax = UTCDateTime(0)
for _tr in st_wl_test_vel:
    if _tr.stats.starttime > tsmax:
        tsmax = _tr.stats.starttime
    if _tr.stats.endtime < temin:
        temin = _tr.stats.endtime

st_wl_test_vel_100t = st_wl_test_vel.copy()
st_wl_test_vel_100t = st_wl_test_vel_100t.interpolate(100, starttime=tsmax).trim(endtime=temin)

# Interactive visualization
# st_wl_test_vel_100t.snuffle(ntracks=len(st_wl_test_vel))

# Run intercomparison for accelerometer to velocity trace & broadband to 
mu = np.zeros((len(wl_vect), len(wl_vect), 3))
sig = np.zeros((len(wl_vect), len(wl_vect), 3))
iholder = []
for _i in range(len(wl_vect)):
    jholder = []
    for _j in range(len(wl_vect)):
        kholder = []
        for _k, _c in enumerate(['Z','N','E']):
            y1 = st_wl_test_vel_100t.select(channel=f'BH{_c}',location='%03d'%(wl_vect[_i]))[0].data
            y2 = st_wl_test_vel_100t.select(channel=f'EN{_c}',location='%03d'%(wl_vect[_j]))[0].data
            diff = y1 - y2
            mu[_i,_j,_k] = np.nanmean(diff)
            sig[_i,_j,_k] = np.nanstd(diff)
            kholder.append(diff)
        jholder.append(kholder)
    iholder.append(jholder)

diff_mat = np.asarray(iholder)
dims = diff_mat.shape
XX, YY = np.meshgrid(wl_vect, wl_vect)


fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,10))
for _i, _c in enumerate(['Z','N','E']):
    axs[0,_i].pcolor(XX,YY,mu[:,:,_i],vmin=np.nanmin(mu),vmax=np.nanmax(mu))
    axs[0,_i].set_title(f'{_c} Component Differences')
    axs[1,_i].pcolor(XX,YY,sig[:,:,_i],vmin=np.nanmin(sig),vmax=np.nanmax(sig))
axs[0,0].set_ylabel('Difference Means')
axs[1,0].set_ylabel('Difference Standard Deviations')

plt.figure()
plt.subplot(121)
plt.pcolor(XX,YY,np.log10(np.sum(np.abs(mu),axis=2)))
plt.colorbar()
plt.xlabel('BH water level')
plt.ylabel('EN water level')
plt.title('Cumumlative magnitude of difference means\nfor 3-C data')
plt.subplot(122)
plt.pcolor(XX,YY,np.log10(np.sum(np.abs(sig),axis=2)))
plt.colorbar()
plt.xlabel('BH water level')
plt.ylabel('EN water level')
plt.title('Cumumlative magnitude of standard deviations \nfor 3-C data')
# fig = plt.figure(figsize=(10,10))

# Plot 
fig, axs = plt.subplots(ncols=dims[0],nrows=dims[1], figsize=(10,10))
for _i in range(dims[0]):
    for _j in range(dims[1]):
        ax = axs[_i,_j]
        for _k,_c in enumerate(['Z','N','E']):
            ax.hist(diff_mat[_i,_j,_k,:],100,alpha=0.25,log=True, label=_c)
            
            if _i == 0:
                ax.set_title(f'EN {wl_vect[_j]}')
            if _j == 0:
                ax.set_ylabel(f'BH {wl_vect[_i]}')


####### REVERSE - TEST BROADBAND/ACCELEROMETER TO ACCELERATION TRACE ########

### TEST METHODS OF INSTRUMENT RESPONSE CORRECTION
wl_vect = [1.0,3.0,10.,30.,60.,100.]
st_wl_test_acc = Stream()
for _wl in wl_vect:
    stEN_vel_wl = stEN.copy()
    for _tr in stEN_vel_wl:
        _tr.remove_response(_ve.inventory, output='ACC', water_level=_wl)
        _tr.stats.location = '%03d'%(_wl)
        st_wl_test_acc += _tr

    stBH_vel_wl = stBH.copy()
    for _tr in stBH_vel_wl:
        _tr.remove_response(_ve.inventory, output='ACC', water_level=_wl)
        _tr.stats.location = '%03d'%(_wl)
        st_wl_test_acc += _tr

# # Interactive visualization
# st_wl_test_acc.snuffle(ntracks=len(st_wl_test_acc))

# Resample to 100 Hz and interpolate to ensure identical time indices
# Trim to ensure matching sample counts
temin = UTCDateTime()
tsmax = UTCDateTime(0)
for _tr in st_wl_test_acc:
    if _tr.stats.starttime > tsmax:
        tsmax = _tr.stats.starttime
    if _tr.stats.endtime < temin:
        temin = _tr.stats.endtime

st_wl_test_acc_100t = st_wl_test_acc.copy()
st_wl_test_acc_100t = st_wl_test_acc_100t.interpolate(100, starttime=tsmax).trim(endtime=temin)

# Interactive visualization
# st_wl_test_acc_100t.snuffle(ntracks=len(st_wl_test_acc))

# Run intercomparison for accelerometer to velocity trace & broadband to 
mu = np.zeros((len(wl_vect), len(wl_vect), 3))
sig = np.zeros((len(wl_vect), len(wl_vect), 3))
iholder = []
for _i in range(len(wl_vect)):
    jholder = []
    for _j in range(len(wl_vect)):
        kholder = []
        for _k, _c in enumerate(['Z','N','E']):
            y1 = st_wl_test_acc_100t.select(channel=f'BH{_c}',location='%03d'%(wl_vect[_i]))[0].data
            y2 = st_wl_test_acc_100t.select(channel=f'EN{_c}',location='%03d'%(wl_vect[_j]))[0].data
            diff = y1 - y2
            mu[_i,_j,_k] = np.nanmean(diff)
            sig[_i,_j,_k] = np.nanstd(diff)
            kholder.append(diff)
        jholder.append(kholder)
    iholder.append(jholder)

diff_mat = np.asarray(iholder)
dims = diff_mat.shape
XX, YY = np.meshgrid(wl_vect, wl_vect)


fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,10))
for _i, _c in enumerate(['Z','N','E']):
    axs[0,_i].pcolor(XX,YY,mu[:,:,_i],vmin=np.nanmin(mu),vmax=np.nanmax(mu))
    axs[0,_i].set_title(f'{_c} Component Differences')
    axs[1,_i].pcolor(XX,YY,sig[:,:,_i],vmin=np.nanmin(sig),vmax=np.nanmax(sig))
axs[0,0].set_ylabel('Difference Means')
axs[1,0].set_ylabel('Difference Standard Deviations')

plt.figure()
plt.subplot(121)
plt.pcolor(XX,YY,np.log10(np.sum(np.abs(mu),axis=2)))
plt.colorbar()
plt.xlabel('BH water level')
plt.ylabel('EN water level')
plt.title('Cumumlative magnitude of difference means\nfor 3-C data')
plt.subplot(122)
plt.pcolor(XX,YY,np.log10(np.sum(np.abs(sig),axis=2)))
plt.colorbar()
plt.xlabel('BH water level')
plt.ylabel('EN water level')
plt.title('Cumumlative magnitude of standard deviations \nfor 3-C data')
# fig = plt.figure(figsize=(10,10))

# Plot 
fig, axs = plt.subplots(ncols=dims[0],nrows=dims[1], figsize=(10,10))
for _i in range(dims[0]):
    for _j in range(dims[1]):
        ax = axs[_i,_j]
        for _k,_c in enumerate(['Z','N','E']):
            ax.hist(diff_mat[_i,_j,_k,:],100,alpha=0.25,log=True, label=_c)
            
            if _i == 0:
                ax.set_title(f'EN {wl_vect[_j]}')
            if _j == 0:
                ax.set_ylabel(f'BH {wl_vect[_i]}')