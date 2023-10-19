"""
:module: util.visualization
:purpose: Contain convenience methods for visualizing waveform and pick data
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
"""
import matplotlib.pyplot as plt
import plotly.express as px
from translate import extract_timestamp, UTCDateTime


def plot_picks(ax,trace, pick_object):
    h1 = ax.plot(trace.times('utcdatetimes'),trace.data,'k-')
    y_bounds = ax.get_ylims()
    tpick = extract_timestamp(pick_object)
    h2 = ax.plot([UTCDateTime(tpick)]*2,y_bounds,'r-')
    return h1,h2

