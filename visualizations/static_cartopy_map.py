"""
:module: static_cartopy_map.py
:purpose: Provide a matplotlib/cartopy example for plotting geospatial data on OpenStreetMap backgrounds
:auth: Nathan T. Stevens
:attributions:
 + This example is built on the StackExchange post by user `Bart`: https://stackoverflow.com/questions/30052990/how-to-use-openstreetmap-background-on-matplotlib-basemap
 + This example uses OpenStreetMap data, which are distributed under a Creative Commons Attribution-ShareAlike 2.0 license
    We thank OpenStreetMap.org for use of these open-source data!
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cf

# Instantiate request to OpenStreetMap
# request = cimgt.OSM()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
# Set Bounds
lat0,lon0 = 47.5828,-122.57841
bounds = [lon0 - 0.5, lon0 + 0.5, lat0 - 0.33, lat0 + 0.33]
# ax = plt.axes(projection=request.crs)
ax.set_extent(bounds)
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.add_feature(cf.COASTLINE)

# ax.add_image(request, 12)
plt.scatter(lon0,lat0,transform=ccrs.PlateCarree())

plt.savefig('test_map.png')

plt.show()