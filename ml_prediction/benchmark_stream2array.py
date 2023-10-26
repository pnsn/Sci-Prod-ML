"""
:module: benchmark_stream2array.py
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:purpose:
    This script runs benchmark tests on data conversion methods for
    converting an arbitrarily ordered obspy.core.stream.Stream object
    into an ordered numpy.ndarray ready for conversion into a PyTorch
    tensor.

    A comparison to the rigid "classic" procedure by Ni & Yuan for the
    ELEP project is provided, but not directly included as it is an
    order of magnitude slower than the other methods proposed here.

    https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb
"""
from time import time
from obspy import read
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def data_load_benchmark(testdata='GNW.UW.2017.131.ms', ndraw=100,
                        include_classic=False):
    """
    Compare times necessary to load data into numpy arrays
    from a loaded ObsPy Stream using a large set of test runs
    for each method provided

    Uses 1 day of 6-channel data
    """
    if include_classic:
        names = ['classic: np.array(st)[[2,1,0], :]',
                 'v0: np.float32(np.array([~ll~]))',
                 'v1: np.array([~ll~], dtype=np.float32)',
                 'v2: np.c_[[~ll~]].astype(np.float32)']
        methods = [_convert_classic,
                   _convert_obspy_listcomp_v0,
                   _convert_obspy_listcomp_v1,
                   _convert_obspy_listcomp_v2] 
    else:
        names = ['v0: np.float32(np.array([~ll~]))',
                 'v1: np.array([~ll~], dtype=np.float32)',
                 'v2: np.c_[[~ll~]].astype(np.float32)']
        methods = [_convert_obspy_listcomp_v0,
                   _convert_obspy_listcomp_v1,
                   _convert_obspy_listcomp_v2]     
    stream = read(testdata).select(channel='?N?')
    vals = np.zeros(shape=(ndraw, len(methods)))
    for _i in tqdm(range(ndraw)):
        for _m in range(len(methods)):
            vals[_i, _m] = methods[_m](stream)
    df = pd.DataFrame(vals, columns=names)
    return df


def _convert_classic(stream):
    """
    Method for data loading & coversion from obspy.core.stream.Stream
    into a numpy.float32 array provided in the ELEP example by 
    Yiyu Ni and Congcong Yuan.

    Finding: This is REALLY inefficient -- 
        takes 10's of seconds for this operation compared 
        to the `_convert_obspy_listcomp_v<X>` test methods proposed
        below.
    """
    to = time()
    np.array(stream)[[2,1,0],:]
    tf = time()
    return tf - to


def _convert_obspy_listcomp_v0(stream):
    """
    Use list comprehension and the `select` obspy.core.stream.Stream
    class method to clarify sorts and accelerate translation 
    use np.float32 wrapper 
    """
    to = time()
    np.float32(np.array([stream.select(channel=f'?N{x}')[0].data for x in 'ZNE']))
    tf = time()
    return tf - to


def _convert_obspy_listcomp_v1(stream):
    """
    Use list comprehension and the `select` obspy.core.stream.Stream
    class method to clarify sorts and accelerate translation 

    add re-assignment to np.float32 as a `dtype` argument
    """
    to = time()
    np.array([stream.select(channel=f'?N{x}')[0].data for x in 'ZNE'], dtype=np.float32)
    tf = time()
    return tf - to


def _convert_obspy_listcomp_v2(stream):
    """
    Use list comprehension and the `select` obspy.core.stream.Stream
    class method to clarify sorts and accelerate translation 

    add re-assignment to np.float32
    switch to np.c_[] for np.array assembly (st[0].data is already type <numpy.ndarray>)
    """
    to = time()
    np.c_[[stream.select(channel=f'?N{x}')[0].data for x in 'ZNE']].astype(np.float32)
    tf = time()
    return tf - to

# DATA LOADING/FORMATTING BENCHMARKS
dpi = 100
fmt = 'png'
ndraw = 1000
df = data_load_benchmark(ndraw=ndraw, include_classic=False)
df.to_csv('./benchmark_stream2array_data.csv',header=True,index=False)
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for _c in df.columns:
    ax1.hist(df[_c].values, 30, density=True, 
             alpha=0.5, label=_c.split(':')[0])
    ax2.plot(df[_c],label=_c)
ax1.set_xlabel('Conversion time (seconds)')
ax1.set_ylabel(f'Frequency')
ax2.set_xlabel('Iteration number (index)')
ax2.set_ylabel('Runtime (sec)')
ax1.legend()
ax2.legend()
fig.savefig(f'./stream2array_performance_benchmark_{dpi:d}dpi.{fmt}',
            dpi=dpi, format=fmt)
plt.show()
