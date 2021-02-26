import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import time
from shapely.geometry import Polygon, Point, LineString
from gradient_free import GeneticAlgorithm

from analyze_wind import init_wind_plant
import geopandas as gpd
import scipy.interpolate


if __name__=="__main__":

    global turbine_x
    global turbine_y
    global plant
    global boundary_poly

    global min_spacing
    global max_turbs
    global rotor_diameter
    global scale
    global progress_filename
    global function_calls

    global iter_number
    global nzones

    global capex_function
    global om_function
    global fcr
    global turbine_rating
    global additional_losses

    turbine = 3
    setback_mult = 3

    if turbine==1:
        powercurve_filename = 'turbine_data/low_2_43r_116d_88h.txt'
        rotor_diameter = 116.0
        hub_height = 88.0
        turbine_rating = 2.430

        capex_cost = np.array([5*1727.0,1727.0,1594.0,1517.0,1490.0,1470.0,1430.0,1420.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==2:
        powercurve_filename = 'turbine_data/med_5_5r_175d_120h.txt'
        rotor_diameter = 175.0
        hub_height = 120.0
        turbine_rating = 5.5

        capex_cost = np.array([5*1438.0,1438.0,1316.0,1244.0,1199.0,1173.0,1133.0,1124.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==3:
        powercurve_filename = 'turbine_data/high_7r_200d_135h.txt'
        rotor_diameter = 200.0
        hub_height = 135.0
        turbine_rating = 7.0

        capex_cost = np.array([5*1072.0,1072.0,970.0,908.0,877.0,862.0,840.0,829]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    minx = 892752.261995
    maxx = 907880.606408
    miny = 160974.455735
    maxy = 174585.645273

    cx = 900173.17505
    cy = 167887.783895
    dx = (maxx-cx)*0.45
    dy = (maxy-cy)*0.45
    low_x = minx+dx
    high_x = maxx-dx
    low_y = miny+dy
    high_y = maxy-dy

    safe = gpd.GeoSeries(Polygon(([low_x,low_y],[high_x,low_y],[high_x,high_y],[low_x,high_y])))

    buffer_distance = setback_mult*(hub_height+rotor_diameter/2.0)

    d1 = gpd.read_file("blue_creek_data/feature_parts/buildings.gpkg")
    buildings = d1["geometry"][0]
    for i in range(len(buildings)):
        ex = buildings[i].buffer(buffer_distance)
        safe = safe.difference(ex)


    d1 = gpd.read_file("blue_creek_data/feature_parts/roads.gpkg")
    roads = d1["geometry"][0]
    for i in range(len(roads)):
        ex = roads[i].buffer(buffer_distance)
        safe = safe.difference(ex)

    d1 = gpd.read_file("blue_creek_data/feature_parts/transmission.gpkg")
    transmission = d1["geometry"][0]
    for i in range(len(transmission)):
        ex = transmission[i].buffer(buffer_distance)
        safe = safe.difference(ex)

    d1 = gpd.read_file("blue_creek_data/feature_parts/rails.gpkg")
    rails = d1["geometry"][0]
    for i in range(len(rails)):
        ex = rails[i].buffer(buffer_distance)
        safe = safe.difference(ex)

    polys = list(safe[0])

    additional_losses = 0.088
    fcr = 0.063
    
    def om_function(capacity):
        return 37.0*capacity*1000.0

    plant = init_wind_plant(hub_height,rotor_diameter,powercurve_filename)

    nzones = len(polys)
    order = np.arange(nzones)
    np.random.shuffle(order)
    aep = np.zeros(nzones)
    coe = np.zeros(nzones)
    turbine_x = np.array([])
    turbine_y = np.array([])

    for i in range(nzones):
        print(i)
        turbine_x = np.append(turbine_x,polys[order[i]].centroid.x)
        turbine_y = np.append(turbine_y,polys[order[i]].centroid.y)
    
        plant.modify_coordinates(turbine_x,turbine_y)
        plant.simulate(1)
        aep[i] = plant.annual_energy_kw()

        nturbs = len(turbine_x)
        capacity = nturbs*turbine_rating
        annual_cost = fcr*capex_function(capacity) + om_function(capacity)
        coe[i] = annual_cost/((1-additional_losses)*aep[i]/1000.0) # $/MWh

        
    zones = np.arange(nzones)
    plt.figure(1)
    plt.plot(zones+1,aep)
    plt.title("aep")
    plt.savefig("sweep_aep.png")

    plt.figure(2)
    plt.plot(zones+1,coe)
    plt.title("coe")
    plt.savefig("sweep_coe.png")
    
    