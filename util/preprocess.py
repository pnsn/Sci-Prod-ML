"""
:module: util.preprocess
:purpose: This module contains support methods for preparing data inputs
          to ML workflows
"""
from obspy.core.stream import Stream

def get_unique_band_inst_codes(stream):
    """
    Get unique combinations of band and instrument codes in a stream 
    containing traces that have SEED-standard convention:

    Band Code
    |
    v
    HH?
     ^
     |
     Instrument Code

    :: INPUT ::
    :param stream: [obspy.stream.Stream] Stream containing some number of
                    traces with channel names

    :: OUTPUT ::
    :return codes: [list] list of unique band/instrument codes
    """
    codes = []
    for tr_ in stream:
        BIC = tr_.stats.channel[:2]
        if BIC not in codes:
            codes.append(BIC)
    return codes

def split_streams(stream):
    """
    Split streams by unique band/instrument codes
    :: INPUT ::
    :param stream: 

    :: OUTPUT ::
    :return streams: [dictionary] Dictionary of streams keyed to the band/instrument code
    """
    codes = get_unique_band_inst_codes(stream)
    streams = {}
    for code in codes:
        sti = stream.select(channel=code+'?')
        streams.update({code:sti})
    return streams

def order_traces(stream, order='ZNE', BIcode='??', mapping={'Z':'[Z3]','N':'[N1]','E':'[E2]'}):
    """
    Order traces in a specified manner
    :: INPUTS ::
    :param stream: [obspy.stream.Stream] Stream containing trace objects with 
                    channel names
    :param order: [str] case sensitive, order in which traces should be sorted
    :param BIcode: [str] Band/Instrument code to enforce using the select() command
                    default is "any code" == '??'
    :param mapping: [dict] mapping of order to obspy.stream.Stream.select() compliant 
                    channel code options

    :: OUTPUT ::
    :return ordered_stream: [obspy.stream.Stream] copy of the input stream, reordered
    """
    # Generate empty stream
    ordered_stream = Stream()
    for C_ in order:
        ordered_stream += stream.copy().select(channel=BIcode+mapping[C_])
    return ordered_stream

# def prepare_3C_bands(stream,reject_list=[]):
#     """
#     Prepare 3C data streams for input to ML workflows in 2 steps
#     1) Identify and segrigate unique Instrument/Band codes
#     2) Sort data into appropriate 3C sequencing: [Z3][N1][E2]
#         a) In the event of 1C data, 
#     """
#     codes = get_unique_band_inst_codes(stream)
#     streams = []
#     for code in codes:
#         if code not in reject_list:
#             st_i = stream.copy().select(channels=code+'?')
#             if len(st_i) < 3:
#                 st_
#             st_o = st_i.select(channels=code+'[Z3]')
#         if 