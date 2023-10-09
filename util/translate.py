"""
:module: util.translate
:purpose: contain utilities for translating data formats. 
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
"""
import pyrocko.gui.marker as pm


def sbm2spm(picks, nslc, method='bounds', kind=0, pick_fmt='%s'):
    """
    Translate from a list of SeisBench predicted picks into a list of Pyrocko/Snuffler phase markers

    :: INPUTS ::
    :param picks: [seisbench.util.annotations.PickList] list of picks from SeisBench
    :param nslc: [tuple] tuple of Network, Station, Location, Channel codes for associated trace
    :param method: [str] Options for how times are written to snuffler markers
                    'bounds': tmin and tmax are the start_time and end_time of the pick
                    'peak': tmin is the peak time of the pick prediction, tmax = None
    :param pick_fmt: [format string] Phase naming formatting. Default to %s

    :: OUTPUT ::
    :return mlist: [list of pyrocko.gui.marker.PhaseMarker] list of pyrocko/snuffler phase markers
                    that can be exported to disk using pm.save_markers()  
    """
    mlist = []
    
    for p_ in picks.picks:
        # Apply specified time-assignment method
        if method == 'bounds':
            t1 = p_.start_time.timestamp
            t2 = p_.end_time.timestamp
        elif method == 'peak':
            t1 = p_.peak_time.timestamp
            t2 = None
        # Create marker and annotate to list
        mark = pm.PhaseMarker((nslc,), tmin=t1, tmax=t2,
                              phasename=pick_fmt % (p_.phase),
                              kind=kind)
        mlist.append(mark)

    return mlist

def full_mapping():
    mapping = {'EQTransformer_P': 'ETP',
                'EQTransformer_S': 'ETS',
                'EQTransformer_Detection': 'ETD',
                'PhaseNet_P': 'PNP',
                'PhaseNet_S': 'PNS',
                'PhaseNet_N': 'PNN'}
    return mapping

def sb_pred_st2sf_st(stream, mapping=full_mapping()):
    """
    Shorten channel names for SeisBench model.annotate() prediction outputs into 3-character
    strings for saving predictions as MSEED files and having snuffler-friendly 
    :: INPUTS ::
    :param stream: [obspy.stream.Stream] Stream containing outputs from a SeisBench .annotate() prediction
    :param mapping: [dict] dictionary for channel re-naming mapping. 
                            if a channel name is not included in `mapping` the script
                            defaults to truncating the channel name to the first 3 characters
    :: OUTPUT ::
    :return st_out: [obspy.stream.Stream] re-labeled stream
    """

    # Create a copy of the input stream to prevent overwriting
    st_out = stream.copy()
    # Iterate across traces in stream
    for tr_ in st_out:
        # Try to re-label using key:value from `mapping`
        try:
            tr_.stats.channel = mapping(tr_.stats.channel)
        # If the mapping fails, truncate to the first 3 characters
        except KeyError:
            tr_.stats.channel = tr_.stats.channel[:3]

    return st_out