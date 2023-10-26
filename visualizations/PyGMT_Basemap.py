import os
import sys
import pygmt
import pandas as pd
from glob import glob
from datetime import datetime
from libcomcat.search import search
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import kilometers2degrees as km2deg
import matplotlib.pyplot as plt
import numpy as np
### Define usgs-libcomcat event search parameters ###
# Bounding times
TS = datetime(2017, 5, 10)
TE = datetime(2017, 5, 20, 23, 59, 59)
# Centroid location of the Bremerton EQS (Earthquake Sequence) & Approximate Radius
lat0, lon0 = 47.5828, -122.57841
max_radius_km = 50.
CLIST = 'HHZ,HHN,HH1,HHE,HH2,ENZ,EN1,ENN,EN2,ENE,BHZ,BHN,BHZ,EHZ,EH1,EHN,EH2,EHE'


# Initialize waveform client
client = Client("IRIS")

events = search(starttime=TS, endtime=TE,
                longitude=lon0, latitude=lat0,
                maxradiuskm=max_radius_km)

inv = client.get_stations(starttime=UTCDateTime(TS),
                          endtime=UTCDateTime(TE),
                          latitude=lat0,
                          longitude=lon0,
                          maxradius=km2deg(max_radius_km),
                          level='channel',
                          channel=CLIST)

## Extract tabular data from inventory
holder = []
for _N in inv.networks:
    for _S in _N.stations:
        if _S.end_date is None:
            sta_end = pd.Timestamp.now()
        else:
            sta_end = pd.Timestamp(_S.end_date.timestamp,unit='s')
        line = [_N.code, _S.code, len(_S.channels),
                _S.longitude, _S.latitude, _S.elevation,
                pd.Timestamp(_S.start_date.timestamp, unit='s'),
                sta_end]
        holder.append(line)

df_sta = pd.DataFrame(holder,columns=['net','sta','nchan','lon','lat','ele','ts','te'])

## Extract tabular data from events
holder = []
for _e in events:
    line = [_e.id, pd.Timestamp(_e.time), _e.time.timestamp(),
            _e['title'], _e['magType'], _e.magnitude,
            _e.longitude,_e.latitude, _e.depth, _e['status'],
            _e['url']]
    holder.append(line)

df_eve = pd.DataFrame(holder, columns=['id','time_UTC','epoch','nearby','magtype','mag','lon','lat','dep','status','url'])

minlon = -122 - (5/6)
minlat = 47.5
maxlon = -122 - (1/6)
maxlat = 47.75

# Instantiate basemap
fig = pygmt.Figure()
# Add basemap
fig.basemap(region=[minlon, maxlon, minlat, maxlat],
            projection="M15c", frame='ag')
fig.coast(land='tan', water='lightsteelblue')

# Plot stations
# pygmt.makecpt(cmap='cyclic',series=(1,6))#,categorical=True)
color_list = ['white','red','blue','yellow','orange']
# Create a colormapping for networks
net_dict = dict(zip(df_sta.net.unique(),color_list[:len(df_sta.net.unique())]))

for _N in df_sta.net.unique():
    idf_sta = df_sta[df_sta.net == _N]

    fig.plot(x=df_sta.lon, y=df_sta.lat, style='ic', pen='1p,black',
            fill=net_dict[_N], size=0.05 * df_sta.nchan)
# Plot events
pygmt.makecpt(cmap='viridis', series=(df_eve.epoch.min(), 
                                      df_eve.epoch.max()))
fig.plot(x=df_eve.lon, y=df_eve.lat, fill=df_eve.epoch,
         size=df_eve.mag*0.1, style='cc', cmap=True,
         transparency=[95]*len(df_eve), pen = '0.5p,black')



# Render result
fig.show()
