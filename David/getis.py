# load dependencies
# load dependencies

import pandas as pd
import geopandas as gpd
from datetime import datetime
import pysal 
from pysal import esda
#import libpysal #added

import numpy as np
from pyproj import CRS
import folium
from pyproj import Transformer


'''Project the given lat,lng from EPSG:4326 to EPSG: 25832'''
inputCRS = CRS.from_epsg(4326)
outputCRS = CRS.from_epsg(25832)
transformer = Transformer.from_crs(inputCRS, outputCRS)
def project(a,b):
    return transformer.transform(a, b)

'''Default getis function using pysal library'''
def getisOrd(data, value = 'value', threshold = 50, lat = 'lat', lng = 'lng'):
    coords = [project(row[lat], row[lng]) for index, row in data.iterrows()]
    #w = libpysal.weights.DistanceBand(coords, threshold)         
    w = pysal.weights.DistanceBand(coords, threshold)
    #getisOrdGlobal = esda.getisord.G(data[value], w)
    getisOrdLocal = esda.getisord.G_Local(data[value], w, transform='B')
    #getisOrdLocal.Zs

    data['z_score'] = getisOrdLocal.Zs
    data['p_value'] = getisOrdLocal.p_norm

    return data

#Defining colors for the spots
def pltcolor(p_value, z_score, confidence):
    cols=[]
    size=[]
    for p, z in zip(p_value, z_score):
        if p < confidence:
            if z > 0:
                cols.append('black')        #hotspot color
                size.append(3)
            else:
                cols.append('blue')       #coldspot color
                size.append(1)
        else:
            cols.append('grey')           #others
            size.append(0.5)
    return cols , size

def plot(data, lat = 'lat', lng = 'lng', p_field = 'p_value', z_field = 'z_score'):
    from matplotlib import pyplot as plt
    #z_score>1.96 hot spots (95% ci)
    #z_score<-1.96 cold spots
    from matplotlib import colors as cls

    f, axarr = plt.subplots(1, figsize=(10,10))
    total_range = cls.Normalize(vmin = - 1.96, vmax = 1.96)
    
    cols, sizes = pltcolor(data[p_field], data[z_field], 0.05)

    plt.scatter(x=data['lng'], y=data['lat'], s=sizes, c=cols, lw = 0) #Pass on the list created by the function here
    plt.grid(True)
    plt.show()


def makepopup(t):
        return str(t)

def map(m, data, value = 'value', style = 'hotspots', lat = 'lat', lng = 'lng', p_field = 'p_value', z_field = 'z_score', col ='#f09205'):
    data = data[data[p_field] < 0.05]
    lats = data[lat]
    lngs = data[lng]

    avg_lat = sum(lats) / len(lats)
    avg_lngs = sum(lngs) / len(lngs)

    latlngs = []
    if style == 'hotspots':
        #z-value
        z = data[z_field]
        p = data[p_field]
        color, sizes = pltcolor(p, z, 0.05)
        value = data[value]
        for lat,lng,c,v in zip(lats, lngs, color, value):
            folium.CircleMarker([lat,lng], color = c, radius =3, fill = c, popup = makepopup(v)).add_to(m)
            latlngs.append([lat,lng])
    else:
        color = [col]*len(data)
        for lat,lng,c in zip(lats, lngs, color):
            folium.CircleMarker([lat,lng], color = c, radius =3, fill = c).add_to(m)
            latlngs.append([lat,lng])

    m.fit_bounds(latlngs)




