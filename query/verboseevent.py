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
 - Merge in methods from util.translate as code refactoring to aid in
   stability of class-methods (NTS: 16. OCT '23)
 - VerboseCatalog.__repr__ inputs are SUPER clunky -
"""
# Import supporting modules
import os
import pandas as pd
from glob import glob
from obspy import UTCDateTime, Stream, Inventory, read_inventory, read
from obspy.clients.fdsn import Client
from libcomcat.search import get_event_by_id
from libcomcat.dataframes import get_phase_dataframe, get_history_data_frame
from libcomcat.classes import DetailEvent, SummaryEvent
from warnings import warn
from requests.models import JSONDecodeError
from tqdm import tqdm
from pyrocko import obspy_compat
import pyrocko.gui.marker as pm
from pyrocko import model

# from pyrocko.model import Event
# import pyrocko.gui.marker as pm


class VerboseEvent:
    """
    This class acts as a container for associated DetailEvent,
    History_DataFrame, and Phase_DataFrame objects from `libcomcat`,
    waveform_Stream and Inventory objects from `obspy`, and 
    methods for cross-referencing data, conducting data queries, 
    and converting cross-referenced data into other data formats 
    (e.g., Snuffler Marker/Event files).

    Current attributes are:
    self.event      = None OR libcomcat.classes.DetailEvent object
    self.history    = None OR pandas.core.dataframe.DataFrame object generated by
                      the libcomcat.dataframe.get_history_data_frame() method
    self.waveforms  = None or obspy.core.stream.Stream object
    self.phase      = None or pandas.core.dataframe.DataFrame object generated by
                      the libcomcat.dataframe.get_phase_data_frame() method
    self.inventory  = None or obspy.core.inventory.Inventory object
    """
    ### INSTANTIATION ###
    def __init__(self, DetailEvent=None, history_DataFrame=None, phase_DataFrame=None, waveform_Stream=None, inventory=None):
        """
        Initialize a VerboseEvent class object with the following kwarg inputs set as None by default

        Attributes are given the following short-hand names:
        self.event=DetailEvent
        self.history=history_DataFrame
        self.phase=phase_DataFrame
        self.waveforms=waveform_Stream
        self.inventory=inventory
        self.attributes=['event','history','phase','waveforms','inventory']



        :: INPUTS :: 
        :param DetailEvent: [None] OR [libcomcat.classes.DetailEvent] 
                    Detailed event object produced by methods in libcomcat
                    e.g., event=libcomcat.search.get_event_by_id(EVID)
        :param history_DataFrame: [None] OR [pandas.core.dataframe.DataFrame] 
                    DataFrame formatted per the output of the method:
                    libcomcat.dataframes.get_history_dataframe()
        :param phase_DataFrame: [None] OR [pandas.core.dataframe.DataFrame]
                    DataFrame formatted per the output of the method:
                    libcomcat.dataframes.get_phase_dataframe()
        :param waveform_Stream: [None] OR [obspy.core.stream.Stream]
                    Stream object associated with event
        :param inventory: [None] OR [obspy.core.inventory.Inventory]
                    Inventory object associated with stations 

        :: OUTPUT ::
        :return VerboseEvent: [util.VerboseEvent]
        """
        self.event=DetailEvent
        self.history=history_DataFrame
        self.phase=phase_DataFrame
        self.waveforms=waveform_Stream
        self.inventory=inventory
        self.attributes=['event','history','phase','waveforms','inventory']

    ### COMMAND LINE DISPLAY ###
    def __repr__(self):
        """
        Representation of an instantiated object when called on the command line
        """
        rep_string= f'=== event ===\n{self.event}\n'
        rep_string += f'=== history ===\n{self.history}\n'
        rep_string += f'=== phase ===\n{self.phase}\n'
        rep_string += f'=== waveforms ===\n{self.waveforms}\n'
        rep_string += f'=== inventory ===\n{self.inventory}\n'
        return rep_string
    
    ### I/O METHODS ###
    def to_layered_directory(self, location, exclude=['waveforms','inventory']):
        """
        Create a heirarchical directory representation of a VerboseEvent class 
        object on disk

        :: INPUTS ::
        :param location: [str] location to initiate file directory tree 
                        (see Structure below)
        :param exclude: [list] or [str] attribute(s) to exclude from the save
                        default is 'waveform' and 'inventory' to save diskspace
                        NOTE: subdirectoreis are not created for any 
                              attributes=None 
                              or those on the exclude list

        :: OUTPUT ::
        :return report: [dict] A small dictionary report for each 
                        sub-directory containing content-sizes in bytes


        Structure:
            {evid}/
                history.csv 
                - via pandas.dataframe.DataFrame.to_csv({filename}),
                                                        header=True,index=True)
                phase.csv
                - via pandas.dataframe.DataFrame.to_csv({filename}),
                                                        header=True,index=True)                
                waveforms/
                    {evid}.{net}.{sta}.mseed
                        - via waveform.select(network={net}, station={sta}).write(
                                os.path.join(location,{evid},'waveforms',{filename})
                        )
                inventory.xml
                - via obspy.core.inventory.Inventory.write({filename},
                                                           fmt='STATIONXML')

        """
        report = {}
        # Conduct work only if there is a valid event entry
        if isinstance(self.event, SummaryEvent) or isinstance(self.event, DetailEvent):
            report.update({'event': self.event})
            for _attr in self.attributes:
                if _attr not in exclude:
                    if _attr == 'event':
                        fpath = os.path.join(location, self.event.id)
                        fsize_bytes = 0
                        try:
                            os.makedirs(fpath)
                        except FileExistsError:
                            pass
                    # Handle saving csv objects
                    if _attr in ['history', 'phase']:
                        filename = f"{_attr}.csv"
                        if _attr == 'history' and isinstance(self.history, pd.DataFrame):
                            self.history.to_csv(os.path.join(fpath, filename),
                                            header=True, index=True)
                            fsize_bytes += os.path.getsize(os.path.join(fpath, filename))


                        if _attr == 'phase' and isinstance(self.phase, pd.DataFrame):
                            self.phase.to_csv(os.path.join(fpath,filename),
                                            header=True, index=True)                        
                            fsize_bytes += os.path.getsize(os.path.join(fpath, filename))

                    # Handle saving waveform data
                    elif _attr == 'waveforms' and isinstance(self.waveforms, Stream):
                        # Try to make the `waveforms` subdirectory
                        try:
                            os.mkdir(os.path.join(fpath,'waveforms'))
                        except FileExistsError:
                            pass
                        # Get Network/Station unique subsets for mseed groupings
                        ns_list = []
                        for tr_ in self.waveforms:
                            nstup = (tr_.stats.network, tr_.stats.station)
                            if nstup not in ns_list:
                                ns_list.append(nstup)
                        # Iterate across Net/Sta entries
                        for n_,s_ in ns_list:
                            # Create filename with specified format
                            filename = os.path.join('waveforms', f'{self.event.id}.{n_}.{s_}.mseed')
                            # Subset waveforms
                            ost = self.waveforms.select(network=n_, station=s_)
                            # Write to disk
                            ost.write(os.path.join(fpath, filename), fmt='MSEED')
                            # Update filesize counter
                            fsize_bytes += os.path.getsize(
                                                os.path.join(fpath, filename)
                                            )

                    # Handle saving Station_XML data
                    elif _attr == 'inventory' and isinstance(self.inventory, Inventory):
                        filename = 'inventory.xml'
                        self.inventory.write(os.path.join(fpath, filename),
                                             'STATIONXML')
                        fsize_bytes += os.path.getsize(os.path.join(fpath,
                                                                    filename))
                    report.update({_attr: fsize_bytes})


        else:
            print('No self.event data -- nothing done')
        
        return report


    def from_layered_directory(self, location, exclude=[]):
        """
        Populate a VerboseEvent object from a layered directory structure generated
        by the `to_layered_directory()` class method.

        See VerboseEvent.to_layered_directory() help for further information on
        directory structure and file formats.

        :: INPUTS ::
        :param location: [str] - relative or absolute path to target
                                EVID labeled directory
        :param exclude: [list] - list of strings matching VerboseEvent.attributes
                                to exclude from the load
                                
        NOTE: This should probably be a stand-alone method, rather than a class method?
        """
        # Handle case where `location` ends with a directory slash
        if location[-1] in ['/','\\']:
            location = location[:-1]
        # 
        path, evid = os.path.split(location)
        for _attr in self.attributes:
            if _attr not in exclude:
                if _attr == 'event':
                    self.event = get_event_by_id(evid)

                elif _attr == 'phase':
                    try:
                        self.phase = pd.read_csv(os.path.join(location,'phase.csv'),
                                                parse_dates=['Arrival Time'])
                    except FileNotFoundError:
                        pass

                elif _attr == 'history':
                    try:
                        self.history = pd.read_csv(os.path.join(location,'history.csv'),
                                                   parse_dates=True)
                    except FileNotFoundError:
                        pass

                elif _attr == 'inventory':
                    try:
                        self.inventory = read_inventory(os.path.join(location,
                                                                    'inventory.xml'))
                    except FileNotFoundError:
                        pass

                elif _attr == 'waveforms':
                    if self.waveforms is None:
                        self.waveforms = Stream()
                    flist = glob(os.path.join(location,'waveforms','*.mseed'))
                    for f_ in flist:
                        self.waveforms += read(f_, fmt='MSEED')
                    if isinstance(self.inventory, Inventory):
                        try:
                            self.waveforms.attach_response(self.inventory)
                        except:
                            pass


        return None

    ### DATA AGGREGATION METHODS ###

    def populate_from_evid(self, evid):
        """
        Provides an option for populating an empty (or partially empty) 
        VerboseEvent object, provided a valid EVID on ComCat using 
        VerboseEvent class-method wrappers for libcomcat. 

        :: INPUT ::
        :param evid: [str] valid `id` for libcomcat methods

        :: OUTPUT ::
        updated self.event
                self.history
                self.phase
        entries for the current instance of this VerboseEvent object
        """
        # Check that the self.event attribute is empty
        if isinstance(self.event, type(None)):
            try:
                self.event=get_event_by_id(evid)
            except JSONDecodeError:
                warn('Invalid EVID for libcomcat.search.get_event_by_id() method')
            except:
                warn('...Somethign else went wrong...')
        # Provide a warning that self.event already contains data
        elif isinstance(self.event, SummaryEvent):
            self.SummaryEvent2DetailEvent()
        else:
            warn(f'self.event contains data of type {type(self.event)}')
            exit()
        
        # Conduct import attempts on self.history and self.phase attributes
        if self.event is not None:
            if ~isinstance(self.history, pd.DataFrame):
                self._populate_history()
            else:
                warn('History attribute already contains a DataFrame!')
            if ~isinstance(self.phase, pd.DataFrame):
                self._populate_phase()
            else:
                warn('self.phase already contains a DataFrame!')


    def get_waveforms(self, client, pad_sec=[0. , 60.], pad_level='wide', net_include='all', sta_include='all', loc_include='all', channels=['BH?','BN?','HH?','HN?','EH?','EN?'], attach_response=True, radius_scan=False, progressbar=False):
        """
        Wrapper for a client.get_waveforms() that uses entries from 
        self.phase and self.event attributes (and self.inventory if not None)
        to retrieve waveform data relevant to the information contained 
        within this VerboseEvent object.

        :: INPUTS ::
        :param client: [obspy.client.Client]
                    Valid client object
        :param pad_sec: [ordered list]
                    2-element list of [front_pad_sec, back_pad_sec]
        :param pad_level: [str]
                    'event':    apply front_pad_sec and back_pad_sec relative
                                to the event origin time (self.event.time)
                    IN DEV - 'channel':  apply front_pad_sec and back_pad_sec relative
                                to the arrival time(s) for a given channel
                                if multiple picks exist, use padding:
                                    [first_pick_time - front_pad_sec
                                     last_pick_time + back_pad_sec]
                    'wide':     apply front_pad_sec and back_pad_sec as
                                    [origin_time - front_pad_sec
                                     last_pick_time + back_pad_sec]
        :param net_include: [list of unix-like strings]
                    Strings to pass to the `network` key-word argument in 
                    client.get_waveforms()
        :param sta_include: [list of unix-like strings]
                    Strings to pass to the `station` key-word argument in 
                    client.get_waveforms()
        :param loc_include: [list of unix-like strings]
                    Strings to pass to the `location` key-word argument in 
                    client.get_waveforms()
        :param channels: [list of unix-like strings]
                    Strings to pass to the `channel` key-word-argument in 
                    client.get_waveforms()
                    see obspy.clients.fdsn.Client.get_waveforms()
        :param attach_response: [bool]
                    False (default) - get trace data and header information
                    True            - get only trace header information
        :param progressbar: [bool]
                    Should there be a `tqdm` progress bar initialized for each
                    station-data load?
        :: OUTPUT ::
        :return self: Updated self.waveforms containing the output                         
        """ 
        if ~isinstance(channels,list):
            channels=list(channels)

        # Make sure waveform data aren't already present
        # if self.waveforms is None:
        ## SANITY
        # If no phase data are provided, but there are event data
        if self.phase is None and self.event is not None:
            warn("no phase data available, defaulting to pad_level='event'")
            pad_level='event'
        # Exit if there are no phase or event data
        elif self.phase is None and self.event is None:
            warn("no Phase or Event data present. Exiting.")
            exit()

        ## GET BOUNDING TIMES FOR QUERY
        # Get bounding times for query type 'event'
        if pad_level.lower() == 'event':
            t0=UTCDateTime(self.event.time) - pad_sec[0]
            tf=UTCDateTime(self.event.time) + pad_sec[1]                 

        # Get bounding times for query type 'wide'
        elif pad_level.lower() == 'wide':
            t0=UTCDateTime(self.event.time) - pad_sec[0]
            tf=UTCDateTime(self.phase['Arrival Time'].max()) + pad_sec[1]
        
        
        ## TODO: Shift NS*L filters from inputs into this section
        # Get unique network-station list and include channel_control
        netsta_list = []
        nscl_unique = self.phase['Channel'].unique()
        for _nscl in nscl_unique:
            _nscl_parsed = _nscl.split('.')
            if [_nscl_parsed[0],_nscl_parsed[1]] not in netsta_list:
                
                # net_include check
                if net_include != 'all':
                    isinclude = False
                    if _nscl_parsed[0] in net_include:
                        isinclude = True
                else:
                    isinclude = True

                # station_include check
                if sta_include != 'all':
                    if _nscl_parsed[1] in sta_include:
                        isinclude = True
                    else:
                        isinclude = False
                else:
                    isinclude = True

                if isinclude:
                    netsta_list.append([_nscl_parsed[0],_nscl_parsed[1]])

        ## DO WAVEFORM QUERY ##
        # Create stream
        wf_stream = Stream()
        # Iterate across net/sta combinations
        for n_,s_ in tqdm(netsta_list, disable=progressbar):
            # Attempt all channel-code permuatations provided 
            # (cuz ObsPy Clients dont like the '??[ZNE123]' ~actual~ Unix wildcard syntax)
            for c_ in channels:
                try:
                    ist=client.get_waveforms(network=n_, station=s_,
                                            location='*', channel=c_,
                                            starttime=t0, endtime=tf,
                                            attach_response=attach_response)
                # Suppresses "bad request"
                except:
                    ist=Stream()
                wf_stream += ist



        self.waveforms = wf_stream 
        # else:
        #     warn(f'self.waveforms already contains data of type {type(self.waveforms)} -- exiting process.')
        #     exit()


    def _get_waveforms_radius_scan(self, client, pad_sec=[0. , 60.], pad_level='wide', channels=['BH?','BN?','HH?','HN?','EH?','EN?'], attach_response=True):
        """
        PRIVATE METHOD

        Subroutine to class method `VerboseEvent.get_waveforms()`

        Using a search radius inherited from self.phase['Distance'].max(), find stations and 

        """
        lon0 = self.event.longitude
        lat0 = self.event.latitude
        if radius_max_deg is None:
            r_max_deg = self.phase['Distance'].max()



    ## Metadata submodule methods

    def populate_history(self):
        """
        PRIVATE METHOD
        
        Populate the self.history attribute so long as the self.event contains
        a valid libcomcat.classes.DetailEvent object that has an `id` attribute.
        """
        if isinstance(self.event,DetailEvent):
            _id=self.event.id
            self.history, _=get_history_data_frame(self.event)
        else:
            warn(f'self.event is type: {type(self.event)} -- seeking libcomcat.classes.DetailEvent')


    def populate_phase(self):
        """
        Populate the self.phase attribute so long as the self.event contains
        a valid libcomcat.classes.DetailEvent object that has an `id` attribute.
        """
        if isinstance(self.event,DetailEvent):
            _id=self.event.id
            self.phase=get_phase_dataframe(self.event)
        else:
            warn(f'self.event is type: {type(self.event)} -- seeking libcomcat.classes.DetailEvent')


    def populate_inventory(self, client, level='station', source='phase'):
        """

        Populate the self.inventory attribute using a defined level of specificity

        :: INPUT ::
        :param client: [obspy.clients.Client]
                Valid client object
        """
        # If the inventory attribute is Nonetype 
        if self.inventory is None:
            # Initialize an inventory
            self.inventory = Inventory()
        elif isinstance(self.inventory, Inventory):
            pass
        else:
            print(f'self.inventory is type {type(self.inventory)}')
            print('overwriting with empty inventory object')
            self.inventory = Inventory()

        # If a valid client is passed
        if isinstance(client, Client):
            # Initially try to populate the inventory from unique nscl's in self.phase
            # if isinstance(self.phase,pd.DataFrame):
            if source == 'phase':
                # Conduct a check to not duplicate net/station/instrument-type entries
                nsi_list = []
                for _c in self.phase['Channel'].unique():
                    _nscl = _c.split('.')
                    # Compose a wildcard version of NSLC
                    _nslc = [_nscl[0], _nscl[1], '*', _nscl[2][:2]+'?']
                    # If unique
                    if _nslc not in nsi_list:
                        # Add to unique list
                        nsi_list.append(_nslc)
                        # Bundle data for kwargs in client.get_stations(), 
                        # including inventory completeness level
                        knslc = dict(zip(['network', 'station', 
                                          'location', 'channel',
                                          'level'],_nslc + [level]))
                        # Append station inventory to self.inventory
                        self.inventory += client.get_stations(**knslc)
            # elif isinstance(self.waveforms,Stream):
            elif source == 'waveforms':
                # Conduct a check to not duplicate net/station/instrument-type entries
                nsi_list = []
                for _tr in self.waveforms:
                    _nscl = [_tr.stats[fld] for fld in ['network','station','channel','location']]
                    # Compose a wildcard version of NSLC
                    _nslc = [_nscl[0], _nscl[1], '*', _nscl[2][:2]+'?']
                    # If unique
                    if _nslc not in nsi_list:
                        # Add to unique list
                        nsi_list.append(_nslc)
                        # Bundle data for kwargs in client.get_stations(), including 
                        # inventory completeness level
                        knslc = dict(zip(['network', 'station',
                                          'location', 'channel',
                                          'level', 'starttime',
                                          'endtime'], _nslc + 
                                          [level, _tr.stats.starttime,
                                           _tr.stats.endtime]))
                        # Append station inventory to self.inventory
                        self.inventory += client.get_stations(**knslc)                
                # pass

    ### PYROCKO EXTENSIONS ###

    def snuffle(self, **kwargs):
        """
        Initialize an interactive Snuffler with as much data as possible
        from data contained inside this VerboseEvent()

        Based on the pyrocko.obspy_compat module extension to 
        obspy.core.stream.Stream, giving streams the `.snuffle()` class method
        """
        obspy_compat.plant()
        if isinstance(self.waveforms,Stream):
            if isinstance(self.inventory,Inventory):
                _inv = self.inventory
                _sta = self.inventory.to_pyrocko_stations()
            else:
                _inv = None
                _sta = None
            
            if isinstance(self.phase,pd.DataFrame) and len(self.phase):
                _eve = self.event2snuffler_event_marker()
                _phz = self.phase_dataframe2snuffler_phase_markers(ehash=_eve.get_hash())
                

        else:
            warn('no data in self.waveforms -- exiting snuffler attempt')
            exit()
                

    def _quality_codes(self,_attr):
        """
        PRIVATE METHOD

        Contains current mapping of event review statuses and pick statuses
        into 0-4 ranking scheme plus kind=5 for ML picks 

        """
        if _attr == 'event':
            codes = {'reviewed':0}
        elif _attr == 'phase':
            codes = {'best time and polarity':0,
                     'time and polarity': 1,
                     'time': 2,
                     'dubious time': 3,
                     'unclear': 4,
                     'ML': 5}
        return codes

    def event2snuffler_event_marker(self):
        """
        Based on the obspy.core.catalog.Catalog.to_pyrocko_events() class method
        added to ObsPy by the `pyrocko.obspy_compat.plant()` method

        Generate a Snuffler event object from a single SummaryEvent or DetailEvent
        from `libcomcat`

        """
        snuf_event = model.Event(name=f"{self.event['code']} - {self.event['title']}",
                                 time=self.event.time.timestamp,
                                 lat=self.event.latitude,
                                 lon=self.event.longitude,
                                 depth=self.event.depth)
        emarker = pm.EventMarker(snuf_event,kind=self.quality_codes('event')[self.event['status']])
        
        return emarker


    def _phase2snufflerphase(self,index):
        phase_marker = pm.PhaseMarker()


    # ### DATA CONVERSION CLASS METHODS ###

    # def to_snuffler(self,label_accents={'manual':'','reviewed':'','ML':'&',}):
    #     """
    #     Convert event and phase arrival information into a list of markers compatable
    #     with the Pyrocko/Snuffler marker file format.
    #     """
    #     label_accents
    #     marker_list=[]
    #     # Convert DetailEvent into EventMarker
    #     _event_marker=_DetailEvent2EventMarker()
    #     event_accents=label_accents(self.event['status'])
    #     _hash=_event_marker.get_hash()
    #     for i_ in range(len(self.phase)):
    #         _phz=self.phase.iloc[i_,:]
    #         _nscl=_phz['Channel'].split('.')
    #         _nslc=((_nscl[0],_nscl[1],_nscl[3],_nscl[2]),)
    #         _phase_marker=pm.PhaseMarker(nslc=_nslc, tmin=_phz['Arrival Time']
    #                                        phase=_phz[''])
            

    # def _DetailEvent2EventMarker(self):
    #     """
    #     PRIVATE METHOD

    #     Convert information from a libcomcat.classes.DetailEvent into a
    #     pyrocko.model.Event object
    #     """


class VerboseCatalog:
    """
    A holder class for VerboseEvent class objects primarily for data exploration
    and visualization methods operating on structured data contained within 
    a list of VerboseEvent objects.
    """
    def __init__(self, verbose_catalog_list, load_phase = True, load_history = False):
        self.events = []
        ## Define initial class structure
        self.summary = pd.DataFrame(columns=["Evid","Magnitude","Magnitude Type",
                                              "Epoch(UTC)","Time UTC","Time Local",
                                              "Distance From","Lat","Lon","Depth Km",
                                              "Depth Mi",
                                              "event","history","phase","waveform","inventory"])

        ## Do some data characterization/sanity checks during instantiation
        if ~isinstance(verbose_catalog_list, list) and isinstance(verbose_catalog_list, VerboseCatalog):
            self.events = list(verbose_catalog_list)
        elif isinstance(verbose_catalog_list, list):
            # Do sanity check on all elements of list
            for _e in tqdm(verbose_catalog_list):
                if isinstance(_e,SummaryEvent):
                    self.events.append(VerboseEvent(_e.getDetailEvent()))
                elif isinstance(_e,DetailEvent):
                    self.events.append(VerboseEvent(_e))
                elif isinstance(_e,VerboseEvent):
                    self.events.append(_e)
                elif isinstance(_e):
                    breakpoint()
            # self.events = verbose_catalog_list
            #[ve for ve in verbose_catalog_list if isinstance(ve, VerboseEvent)]

        for _e in self.events:
            if _e.phase is None:
                _e.populate_phase()

        holder = []
        for _i, _ve in enumerate(self.events):
            # Do dataframe summary of events in PNSN standard EVENT table format
            line=[_ve.event.id, _ve.event.magnitude, _ve.event['magType'][1:],
                  _ve.event.time.timestamp(), _ve.event.time, pd.NaT,
                  _ve.event['title'], _ve.event.latitude, 
                  _ve.event.longitude, _ve.event.depth,
                  _ve.event.depth*1e3*3.281/5280.,
                  isinstance(_ve.event, DetailEvent),
                  isinstance(_ve.history, pd.DataFrame),
                  isinstance(_ve.phase, pd.DataFrame),
                  isinstance(_ve.waveforms, Stream),
                  isinstance(_ve.inventory, Inventory)]

            holder.append(line)
        # Update Dataframe
        self.summary = pd.DataFrame(holder, columns=self.summary.columns)


    def __repr__(self):
        nentries=len(self.summary)
        repr_str= f"=== Events:      {self.summary['event'].sum()}/{nentries} ===\n"
        repr_str += f"=== Histories:   {self.summary['history'].sum()}/{nentries} ===\n"
        repr_str += f"=== Phases:      {self.summary['phase'].sum()}/{nentries} ===\n"
        repr_str += f"=== Waveforms:   {self.summary['waveform'].sum()}/{nentries} ===\n"
        repr_str += f"=== Inventories: {self.summary['inventory'].sum()}/{nentries} ===\n"
        return repr_str


    def populate_from_event_ids(self):
        for _ve in self.events:
            try: 
                _ve.populate_from_evid(_ve.event.id)
            except:
                pass

    def populate_phases(self):
        for _ve in self.events:
            try:
                _ve._populate_phase()
            except:
                pass
    

    def get_waveforms(self, **kwargs):
        approval=False
        print(f'Proposed fetching waveforms for {len(self.events)}. Are you sure?')
        # Get user confirmation

        if approval:
            for _ve in self.events:
                try:
                    _ve.get_waveforms(**kwargs)
                except:
                    pass
        else:
            print('Download canceled - quitting')



        

