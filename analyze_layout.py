import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from shapely.geometry import Polygon
from gradient_free import GeneticAlgorithm

from analyze_wind import init_wind_plant
import geopandas as gpd
import scipy.interpolate

import csv


def read_aep_file(filename):

    with open(filename) as f:
        reader = csv.reader(f,delimiter=',')
        linenum = 0

        x = False
        y = False

        for row in reader:
            if len(row) > 0:
                
                # read in middle lines
                if x==True and y==False:
                        for i in range(len(row)):
                            if row[i] != "":
                                chars = list(row[i])
                                read = True
                                num = ""
                                for i in range(len(chars)):
                                    if chars[i] == "]":
                                        read = False
                                        y = True
                                    if read == True:
                                        num = num+chars[i]
                                turbine_x = np.append(turbine_x,float(num))
                    

                # read in first line
                if row[0].split()[0] == "turbine_x" and x==False and y==False:
                    chars = list(row[0].split()[-1])
                    n1 = ""
                    read = False
                    for i in range(len(chars)):
                        if read == True:
                            n1 = n1+chars[i]
                        if chars[i] == "[":
                            read = True
                    turbine_x = np.array([float(n1)])
                    x = True
                
                    for i in range(len(row)-1):
                        if row[i+1] != "":
                            turbine_x = np.append(turbine_x,float(row[i+1]))

                

                if y==True and x==False:
                        
                        for i in range(len(row)):
                            if row[i] != "":
                                chars = list(row[i])
                                read = True
                                num = ""
                                for i in range(len(chars)):
                                    if chars[i] == "]":
                                        read = False
                                        y = False
                                    if read == True:
                                        num = num+chars[i]
                                turbine_y = np.append(turbine_y,float(num))
                    

                # read in first line
                if row[0].split()[0] == "turbine_y" and x==True and y==True:
                    chars = list(row[0].split()[-1])
                    n1 = ""
                    read = False
                    for i in range(len(chars)):
                        if read == True:
                            n1 = n1+chars[i]
                        if chars[i] == "[":
                            read = True
                    turbine_y = np.array([float(n1)])
                    x = False
                
                    for i in range(len(row)-1):
                        if row[i+1] != "":
                            turbine_y = np.append(turbine_y,float(row[i+1]))


    return turbine_x, turbine_y


if __name__=="__main__":

    # THESE SHOULD BE THE ONLY THINGS YOU NEED TO CHANGE BETWEEN RUNS

    objs = np.array(["aep","coe","profit"])
    turbs = np.array([1,2,3],dtype=int)
    mults = np.array([0.0,1.1,2.0,3.0])
    ppa_mult_arr = np.array([1.01,1.05,1.1,1.2])

    # objs = np.array(["coe","profit"])
    # turbs = np.array([3],dtype=int)
    # mults = np.array([2.0,3.0])
    # ppa_mult_arr = np.array([1.2])

    objective = np.array([])
    turbine = np.array([],dtype=int)
    setback_mult = np.array([])
    ppa_mult = np.array([])
    for i in range(len(objs)):
        for j in range(len(turbs)):
            for k in range(len(mults)):
                for m in range(len(ppa_mult_arr)):
                    objective = np.append(objective,objs[i])
                    turbine = np.append(turbine,turbs[j])
                    setback_mult = np.append(setback_mult,mults[k])
                    ppa_mult = np.append(ppa_mult,ppa_mult_arr[m])

    nruns = len(objective)
    nturbs = np.zeros(nruns)
    capacity = np.zeros(nruns)
    aep = np.zeros(nruns)
    aep_w_losses = np.zeros(nruns)
    annual_cost = np.zeros(nruns)
    annual_income = np.zeros(nruns)
    coe = np.zeros(nruns)
    profit = np.zeros(nruns)
    total_area = np.zeros(nruns)
    safe_area = np.zeros(nruns)
    total_cap_density = np.zeros(nruns)
    safe_cap_density = np.zeros(nruns)
    ppa = np.zeros(nruns)

    

    for i in range(nruns):
        print("%s/%s"%(i+1,nruns))


        if objective[i] == "aep" or objective[i] == "coe":
            filename = "%s/turbine%s_setback%s.txt"%(objective[i],turbine[i],setback_mult[i])
        elif objective[i] == "profit":
            filename = "%s/turbine%s_setback%s_ppa%s.txt"%(objective[i],turbine[i],setback_mult[i],ppa_mult[i])
        turbine_x, turbine_y = read_aep_file(filename)


        if turbine[i]==1:
            powercurve_filename = 'turbine_data/low_2_43r_116d_88h.txt'
            rotor_diameter = 116.0
            hub_height = 88.0
            turbine_rating = 2.430

            capex_cost = np.array([2*1727.0,1727.0,1594.0,1517.0,1490.0,1470.0,1430.0,1420.0]) # $/kW
            capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
            cost = capex_size*capex_cost*1000.0
            capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
            ppa[i] = 46.56550870915244*ppa_mult[i]

        elif turbine[i]==2:
            powercurve_filename = 'turbine_data/med_5_5r_175d_120h.txt'
            rotor_diameter = 175.0
            hub_height = 120.0
            turbine_rating = 5.5

            capex_cost = np.array([2*1438.0,1438.0,1316.0,1244.0,1199.0,1173.0,1133.0,1124.0]) # $/kW
            capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
            cost = capex_size*capex_cost*1000.0
            capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
            ppa[i] = 35.371107013888526*ppa_mult[i]

        elif turbine[i]==3:
            powercurve_filename = 'turbine_data/high_7r_200d_135h.txt'
            rotor_diameter = 200.0
            hub_height = 135.0
            turbine_rating = 7.0

            capex_cost = np.array([2*1072.0,1072.0,970.0,908.0,877.0,862.0,840.0,829]) # $/kW
            capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
            cost = capex_size*capex_cost*1000.0
            capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
            ppa[i] = 27.515987098794344*ppa_mult[i]

        additional_losses = 0.088
        fcr = 0.063

        plant = init_wind_plant(hub_height,rotor_diameter,powercurve_filename)
        tip_height = hub_height+rotor_diameter/2.0
        buffer_distance = setback_mult[i]*tip_height

        scale = 1E3
            
        # 1 SETUP THE SAFE ZONES
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

        boundary_poly = Polygon(([low_x,low_y],[high_x,low_y],[high_x,high_y],[low_x,high_y]))
        safe = gpd.GeoSeries(boundary_poly)

        if setback_mult[i] != 0.0:
            d1 = gpd.read_file("blue_creek_data/feature_parts/buildings.gpkg")
            buildings = d1["geometry"][0]
            for j in range(len(buildings)):
                ex = buildings[j].buffer(buffer_distance)
                safe = safe.difference(ex)


            d1 = gpd.read_file("blue_creek_data/feature_parts/roads.gpkg")
            roads = d1["geometry"][0]
            for j in range(len(roads)):
                ex = roads[j].buffer(buffer_distance)
                safe = safe.difference(ex)

            d1 = gpd.read_file("blue_creek_data/feature_parts/transmission.gpkg")
            transmission = d1["geometry"][0]
            for j in range(len(transmission)):
                ex = transmission[j].buffer(buffer_distance)
                safe = safe.difference(ex)

            d1 = gpd.read_file("blue_creek_data/feature_parts/rails.gpkg")
            rails = d1["geometry"][0]
            for j in range(len(rails)):
                ex = rails[j].buffer(buffer_distance)
                safe = safe.difference(ex)

            polys = list(safe[0])

        else:
            polys = list(safe)

        total_area[i] = boundary_poly.area/1E6
        for j in range(len(polys)):
            safe_area[i] += polys[j].area/1E6

        def om_function(cap):
            return 37.0*cap*1000.0

        nturbs[i] = len(turbine_x)
        plant.modify_coordinates(turbine_x,turbine_y)
        plant.simulate(1)
        aep[i] = plant.annual_energy_kw()/scale
        aep_w_losses[i] = (1-additional_losses)*aep[i]
        capacity[i] = nturbs[i]*turbine_rating
        annual_cost[i] = fcr*capex_function(capacity[i]) + om_function(capacity[i])
        annual_income[i] = aep_w_losses[i]*ppa[i]
        coe[i] = annual_cost[i]/((1-additional_losses)*aep[i]) # $/MWh
        profit[i] = (annual_income[i] - annual_cost[i])


        
        total_cap_density[i] = capacity[i]/(total_area[i])
        safe_cap_density[i] = capacity[i]/(safe_area[i])

        # print("capacity (MW): " + "\t".expandtabs(20) + "%s"%capacity)
        # print("aep (GWh): " + "\t".expandtabs(24) + "%s"%(aep/scale))
        # print("aep with losses (GWh): " + "\t".expandtabs(12) + "%s"%((1-additional_losses)*aep/scale))
        # print("coe ($/MWh): " + "\t".expandtabs(22) + "%s"%coe)
        # print("total area (km^2): " + "\t".expandtabs(16) + "%s"%(total_area/1E6))
        # print("safe area (km^2): " + "\t".expandtabs(17) + "%s"%(safe_area/1E6))
        # print("capacity density total (MW/km^2): " + "\t".expandtabs(1) + "%s"%(capacity/(total_area/1E6)))
        # print("capacity density safe (MW/km^2): " + "\t".expandtabs(2) + "%s"%(capacity/(safe_area/1E6)))



    output_data = pd.DataFrame([objective,turbine,setback_mult,ppa_mult,nturbs,capacity,aep/1E3,aep_w_losses/1E3,annual_cost/1E6,annual_income/1E6,
            coe,total_area,safe_area,total_cap_density,safe_cap_density,ppa,profit/1E6],['Objective','Turbine Type',
            'Setback Tip Height Multiplier','PPA Multiplier','Number of Turbines',
            'Total Capacity (MW)','AEP (GWh)','AEP with Losses (GWh)','Annual Cost ($MM)','Annual Income ($MM)','COE ($/MWh)','Boundary Area (km^2)','Available Area (km^2)',
            'Boundary Capacity Density (MW/km^2)','Available Capactiy Density (MW/km^2)','PPA ($/MWh)','Profit ($MM)'])
    

    transposed = output_data.transpose()
    transposed.to_csv("full_data.csv")

    # print("objective: ", objective)
    # print("turbine: ", turbine)
    # print("setback_mult: ", setback_mult)
    # print("ppa_mult: ", ppa_mult)
    # print("nturbs: ", nturbs)
    # print("capacity: ", capacity)
    # print("aep: ", aep)
    # print("aep_w_losses: ", aep_w_losses)
    # print("annual_cost: ", annual_cost)
    # print("coe: ", coe)
    # print("total_area: ", total_area)
    # print("safe_area: ", safe_area)
    # print("total_cap_density: ", total_cap_density)
    # print("safe_cap_density: ", safe_cap_density)
    # print("ppa: ", ppa)
    # print("profit: ", profit/1E6)
    