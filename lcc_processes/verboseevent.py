"""
:module: util.verboseevent
:purpose: provide multi-step data query workflows and support methods
          based on the usgs-libcomcat library:
          https://code.usgs.gov/ghsc/esi/libcomcat-python
          encapsulating data holding and method in a class:
          VerboseEvent
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:origin: 16. October 2023
:org: Pacific Northwest Seismic Network

:Depends:
 - pandas
 - obspy
 - python-libcomcat
 - pyrocko (methods in development)



:: NOTES / TODO ::
 - Merge in methods from util.translate as code refactoring to aid in stability of class-methods (NTS: 16. OCT '23)
"""
# Import supporting modules
import os
import pandas as pd
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client
from libcomcat.search import search, get_event_by_id
from libcomcat.dataframes import get_phase_dataframe
from libcomcat.classes import DetailEvent, SummaryEvent
from warnings import warn
from requests.models import JSONDecodeError
from tqdm import tqdm

# from pyrocko.model import Event
# import pyrocko.gui.marker as pm

# from libcomcat.classes import DetailEvent

class VerboseEvent:
    """
    This class acts as a container for associated DetailEvent, History_DataFrame,
    and Phase_DataFrame objects from `libcomcat` and contains a number of methods 
    for cross-referencing data, conducting waveform queries, and converting cross-
    referenced data into other data formats (e.g., Snuffler Marker/Event files).

    Current attributes are:
    self.Event = None OR libcomcat.classes.DetailEvent object
    self.History = None OR pandas.core.dataframe.DataFrame object generated by
                   the libcomcat.dataframe.get_history_data_frame() method
    """
    ### INSTANTIATION METHODS ###
    def __init__(self, DetailEvent = None, History_DataFrame = None, Phase_DataFrame = None, Waveform_Stream = None, Station_Inventory = None):
        """
        Initialize a VerboseEvent class object with the following kwarg inputs set as None by default

        Attributes are given the following short-hand names:
        self.Event = DetailEvent
        self.History = History_DataFrame
        self.Phase = Phase_DataFrame
        self.Waveforms = Waveform_Stream
        self.Station = Station_Inventory
        self.attributes = ['Event','History','Phase','Waveforms','Station']



        :: INPUTS :: 
        :param DetailEvent: [None] OR [libcomcat.classes.DetailEvent] 
                    Detailed event object produced by methods in libcomcat
                    e.g., event = libcomcat.search.get_event_by_id(EVID)
        :param History_DataFrame: [None] OR [pandas.core.dataframe.DataFrame] 
                    DataFrame formatted per the output of the method:
                    libcomcat.dataframes.get_history_dataframe()
        :param Phase_DataFrame: [None] OR [pandas.core.dataframe.DataFrame]
                    DataFrame formatted per the output of the method:
                    libcomcat.dataframes.get_phase_dataframe()
        :param Waveform_Stream: [None] OR [obspy.core.stream.Stream]
                    Stream object associated with event
        :param Station_Inventory: [None] OR [obspy.core.inventory.Inventory]
                    Inventory object associated with stations 

        :: OUTPUT ::
        :return VerboseEvent: [util.VerboseEvent]
        """
        self.Event = DetailEvent
        self.History = History_DataFrame
        self.Phase = Phase_DataFrame
        self.Waveforms = Waveform_Stream
        self.Station = Station_Inventory
        self.attributes = ['Event','History','Phase','Waveforms','Station']

    def __repr__(self):
        """
        Representation of an instantiated object when called on the command line
        """
        rep_string =  f'=== Event ===\n{self.Event}\n'
        rep_string += f'=== History ===\n{self.History}\n'
        rep_string += f'=== Phase ===\n{self.Phase}\n'
        rep_string += f'=== Waveforms ===\n{self.Waveforms}\n'
        rep_string += f'=== Station ===\n{self.Station}\n'
        return rep_string
    
    ### READ/METHODS ###
    def to_layered_directory(self, location, exclude = ['Waveform','Station']):
        """
        Create a heirarchical directory representation of a VerboseEvent class object on disk

        :: INPUTS ::
        :param location: [str] location to initiate file directory tree (see Structure below)
        :param exclude: [list] or [str] attribute(s) to exclude from the save
                        default is 'Waveform' and 'Station' to save disk space
                        NOTE: subdirectoreis are not created for any attributes = None 
                              or those on the exclude list

        :: OUTPUT ::
        :return report: [dict] A small dictionary report for each sub-directory containing content-sizes in bytes


        Structure:
            {evid}/
                History/
                    {evid}_history.csv 
                        - via pandas.dataframe.DataFrame.to_csv({filename}),
                                                                header=True,index=True)
                Phase/
                    {evid}_phase.csv
                        - via pandas.dataframe.DataFrame.to_csv({filename}),
                                                                header=True,index=True)                
                Waveform/
                    {evid}.{net}.{sta}.mseed
                        - via Waveform.select(network={net}, station={sta}).write(
                                os.path.join(location,{evid},'waveform',{filename})
                        )
                Station/
                    {evid}_inv.xml
                        - via obspy.core.inventory.Inventory.write({filename},fmt='station_xml')

        """
        report = {}
        # Conduct work only if there is a valid event entry
        if isinstance(self.Event,SummaryEvent) or isinstance(self.Event,DetailEvent):
            report.update({'Event': self.Event})
            for _attr in self.attributes:
                if _attr not in exclude and _attr == 'Event':
                    fpath = os.path.join(location,self.Event.id,_attr)
                    fsize_bytes = 0
                    try:
                        os.makedirs(fpath)
                    except FileExistsError:
                        pass
                    # Handle saving csv objects
                    if _attr in ['History', 'Phase']:
                        filename = f"{self.Event.id}_{_attr}"
                        self._attr.to_csv(os.path.join(fpath, filename),
                                          header=True,index=True)
                        fsize_bytes += os.path.getsize(os.path.join(fpath, filename))
                    # Handle saving waveform data
                    elif _attr == 'Waveform':
                        ns_list = []
                        for tr_ in self._attr:
                            nstup = (tr_.stats.network,tr_.stats.station)
                            if nstup not in ns_list:
                                ns_list.append(nstup)
                        for n_,s_ in ns_list:
                            filename = f'{self.Event.id}.{n_}.{s_}.mseed'
                            ost = self._attr.select(network=n_,station=s_)
                            ost.write(os.path.join(fpath,filename),fmt='MSEED')
                            fsize_bytes += os.path.getsize(os.path.join(fpath, filename))

                    # Handle saving Station_XML data
                    elif _attr == 'Station':
                        filename = f'{self.Event.id}_inv.xml'
                        self._attr.write(os.path.join(fpath,filename),fmt='STATIONXML')
                        fsize_bytes += os.path.getsize(os.path.join(fpath, filename))
                    report.update({_attr: fsize_bytes})


        else:
            print('No self.Event data -- nothing done')
        
        return report


    ### DATA AGGREGATION METHODS ###

    def populate_from_evid(self, evid):
        """
        Provides an option for populating an empty (or partially empty) VerboseEvent
        object, provided a valid EVID on ComCat using VerboseEvent class-method 
        wrappers for libcomcat. 

        :: INPUT ::
        :param evid: [str] valid `id` for libcomcat methods

        :: OUTPUT ::
        updated self.Event
                self.History
                self.Phase
        entries for the current instance of this VerboseEvent object
        """
        # Check that the self.Event attribute is empty
        if isinstance(self.Event, type(None)):
            try:
                self.Event = get_event_by_id(evid)
            except JSONDecodeError:
                warn('Invalid EVID for libcomcat.search.get_event_by_id() method')
            except:
                warn('...Somethign else went wrong...')
        # Provide a warning that self.Event already contains data
        elif isinstance(self.Event, SummaryEvent):
            self.SummaryEvent2DetailEvent()
        else:
            warn(f'self.Event contains data of type {type(self.Event)}')
            exit()
        
        # Conduct import attempts on self.History and self.Phase attributes
        if self.Event is not None:
            if ~isinstance(self.History, pd.DataFrame):
                self._populate_history()
            else:
                warn('History attribute already contains a DataFrame!')
            if ~isinstance(self.Phase, pd.DataFrame):
                self._populate_phase()
            else:
                warn('self.Phase already contains a DataFrame!')


    def get_waveforms(self, client, pad_sec = [0 , 60], pad_level = 'wide', channels = ['BH?','BN?','HH?','HN?','EH?','EN?']):
        """
        Wrapper for a client.get_waveforms() that uses entries from 
        self.Phase and self.Event attributes (and self.Station if not None)
        to retrieve waveform data relevant to the information contained 
        within this VerboseEvent object.

        :: INPUTS ::
        :param client: [obspy.client.Client]
                    Valid client object
        :param pad_sec: [ordered list]
                    2-element list of [front_pad_sec, back_pad_sec]
        :param pad_level: [str]
                    'event':    apply front_pad_sec and back_pad_sec relative
                                to the event origin time (self.Event.time)
                    IN DEV - 'channel':  apply front_pad_sec and back_pad_sec relative
                                to the arrival time(s) for a given channel
                                if multiple picks exist, use padding:
                                    [first_pick_time - front_pad_sec
                                     last_pick_time + back_pad_sec]
                    'wide':     apply front_pad_sec and back_pad_sec as
                                    [origin_time - front_pad_sec
                                     last_pick_time + back_pad_sec]
        :param channels: [list of unix-like strings]
                    Strings to pass to the `channel` key-word-argument in client.get_waveforms()
                    see obspy.clients.fdsn.Client.get_waveforms()

        :: OUTPUT ::
        :return self: Updated self.Waveforms containing the output                         
        """ 
        if ~isinstance(channels,list):
            channels = list(channels)

        # Make sure waveform data aren't already present
        if self.Waveforms is None:
            ## SANITY
            # If no phase data are provided, but there are event data
            if self.Phase is None and self.Event is not None:
                warn("no phase data available, defaulting to pad_level = 'event'")
                pad_level = 'event'
            # Exit if there are no phase or event data
            elif self.Phase is None and self.Event is None:
                warn("no Phase or Event data present. Exiting.")
                exit()

            ## GET BOUNDING TIMES FOR QUERY
            # Get bounding times for query type 'event'
            if pad_level.lower() == 'event':
                t0 = UTCDateTime(self.Event.time) - pad_sec[0]
                tf = UTCDateTime(self.Event.time) + pad_sec[1]                 

            # Get bounding times for query type 'wide'
            elif pad_level.lower() == 'wide':
                t0 = UTCDateTime(self.Event.time) - pad_sec[0]
                tf = UTCDateTime(self.Phase['Arrival Time'].max()) + pad_sec[1]

            # Get unique network-station list and include channel_control
            netsta_list = []
            nscl_unique = self.Phase['Channel'].unique()
            for _nscl in nscl_unique:
                _nscl_parsed = _nscl.split('.')
                if [_nscl_parsed[0],_nscl_parsed[1]] not in netsta_list:
                    netsta_list.append([_nscl_parsed[0],_nscl_parsed[1]])

            ## DO WAVEFORM QUERY ##
            # Create stream
            wf_stream = Stream()
            # Iterate across net/sta combinations
            for n_,s_ in tqdm(netsta_list):
                # Attempt all channel-code permuatations provided 
                # (cuz ObsPy Clients dont like the '??[ZNE123]' ~actual~ Unix wildcard syntax)
                for c_ in channels:
                    try:
                        ist = client.get_waveforms(network=n_, station=s_, 
                                                   location='*', channel=c_,
                                                   starttime=t0, endtime=tf)
                    # Suppresses "bad request"
                    except:
                        ist = Stream()
                    wf_stream += ist

            self.Waveforms = wf_stream 
        else:
            warn(f'self.Waveforms already contains data of type {type(self.Waveforms)} -- exiting process.')
            exit()


    ## Metadata submodule methods
    def SummaryEvent2DetailEvent(self):
        """
        Convert valid libcomcat.classes.SummaryEvent object assigned to self.Event
        into a libcomcat.classes.DetailEvent for the same event.

        Simple wrapper for the libcomcat.search.get_event_by_id() method
        """
        if isinstance(self.Event,SummaryEvent):
            self.Event = get_event_by_id(self.Event.id)
        elif(self.Event,DetailEvent):
            print('self.Event is already type libcomcat.classes.DetailEvent')
            pass
        else:
            print('...something else went wrong...')
            exit()


    def _populate_history(self):
        """
        PRIVATE METHOD
        
        Populate the self.History attribute so long as the self.Event contains
        a valid libcomcat.classes.DetailEvent object that has an `id` attribute.
        """
        if isinstance(self.Event,DetailEvent):
            _id = self.Event.id
            self.History, _ = get_history_data_frame(self.Event)
        else:
            warn(f'self.Event is type: {type(self.Event)} -- seeking libcomcat.classes.DetailEvent')


    def _populate_phase(self):
        """
        PRIVATE METHOD

        Populate the self.Phase attribute so long as the self.Event contains
        a valid libcomcat.classes.DetailEvent object that has an `id` attribute.
        """
        if isinstance(self.Event,DetailEvent):
            _id = self.Event.id
            self.Phase = get_phase_dataframe(self.Event)
        else:
            warn(f'self.Event is type: {type(self.Event)} -- seeking libcomcat.classes.DetailEvent')


    def _populate_inventory(self, client, level = 'station'):
        """
        PRIVATE METHOD - IN DEVELOPMENT

        Populate the self.Station attribute using a defined level of specificity

        :: INPUT ::
        :param client: [obspy.clients.Client]
                Valid client object
        """
        if self.Station is None or ~isinstance(client,Client):
            self.Station = Inventory()


    ### DATA CONVERSION CLASS METHODS ###

    def to_snuffler(self,label_accents={'manual':'','reviewed':'','ML':'&',}):
        """
        Convert event and phase arrival information into a list of markers compatable
        with the Pyrocko/Snuffler marker file format.
        """
        label_accents
        marker_list = []
        # Convert DetailEvent into EventMarker
        _event_marker = _DetailEvent2EventMarker()
        event_accents = label_accents(self.Event['status'])
        _hash = _event_marker.get_hash()
        for i_ in range(len(self.Phase)):
            _phz = self.Phase.iloc[i_,:]
            _nscl = _phz['Channel'].split('.')
            _nslc = ((_nscl[0],_nscl[1],_nscl[3],_nscl[2]),)
            _phase_marker = pm.PhaseMarker(nslc = _nslc, tmin=_phz['Arrival Time']
                                           phase = _phz[''])
            

    def _DetailEvent2EventMarker(self):
        """
        PRIVATE METHOD

        Convert information from a libcomcat.classes.DetailEvent into a
        pyrocko.model.Event object
        """



class VerboseCatalog:
    """
    A holder class for VerboseEvent class objects primarily for data exploration
    and visualization methods operating on structured data contained within 
    a list of VerboseEvent objects.
    """
    def __init__(self,Verbose_Catalog_List = []):
        if ~isinstance(Verbose_Catalog_List, list) and isinstance(Verbose_Catalog_List, VerboseCatalog):
            self.Events = list(Verbose_Catalog_List)
        elif isinstance(Verbose_Catalog_List, list):
            # Do sanity check on all elements of list
            self.Events = [ve for ve in Verbose_Catalog_List if isinstance(ve,VerboseEvent)]
            
        df_events = pd.DataFrame()
        for _ve in self.Events:
            # Do dataframe summary of events in PNSN standard EVENT table format
        self.event_summary = df_events

        
    def populate_from_event_ids(self):
        for _ve in self.Events:
            try: 
                _ve.populate_from_evid(_ve.Event.id)
            except:
                pass
        
    def get_waveforms(self,**kwargs):
        approval = False
        print(f'Proposed fetching waveforms for {len(self.Events)}. Are you sure?')
        # Get user confirmation

        if approval:
            for _ve in self.Events:
                try:
                    _ve.get_waveforms(**kwargs)
                except:
                    pass
        else:
            print('Download canceled - quitting')



        
