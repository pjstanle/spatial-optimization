import os
import sys

import numpy as np
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

    turbine = int(sys.argv[1]) # 1: low 2: meduim 3: high
    setback_mult = float(sys.argv[2]) # float
    objective = sys.argv[3] # coe or profit
    try:
        ppa_mult = float(sys.argv[4])
    except:
        ppa_mult = 0.0

    start_filename = "aep/turbine%s_setback%s.txt"%(turbine,setback_mult)
    turbine_x, turbine_y = read_aep_file(start_filename)

    if objective == "coe":
        save_filename = "coe/turbine%s_setback%s.txt"%(turbine,setback_mult)
    elif objective == "profit":
        save_filename = "profit/turbine%s_setback%s_ppa%s.txt"%(turbine,setback_mult,ppa_mult)
    
    if turbine==1:
        powercurve_filename = 'turbine_data/low_2_43r_116d_88h.txt'
        rotor_diameter = 116.0
        hub_height = 88.0
        turbine_rating = 2.430

        capex_cost = np.array([2*1727.0,1727.0,1594.0,1517.0,1490.0,1470.0,1430.0,1420.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
        ppa = 46.56550870915244*ppa_mult

    elif turbine==2:
        powercurve_filename = 'turbine_data/med_5_5r_175d_120h.txt'
        rotor_diameter = 175.0
        hub_height = 120.0
        turbine_rating = 5.5

        capex_cost = np.array([2*1438.0,1438.0,1316.0,1244.0,1199.0,1173.0,1133.0,1124.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
        ppa = 35.371107013888526*ppa_mult

    elif turbine==3:
        powercurve_filename = 'turbine_data/high_7r_200d_135h.txt'
        rotor_diameter = 200.0
        hub_height = 135.0
        turbine_rating = 7.0

        capex_cost = np.array([2*1072.0,1072.0,970.0,908.0,877.0,862.0,840.0,829]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')
        ppa = 27.515987098794344*ppa_mult

    additional_losses = 0.088
    fcr = 0.063
    
    def om_function(capacity):
        return 37.0*capacity*1000.0

    plant = init_wind_plant(hub_height,rotor_diameter,powercurve_filename)

    
    converged = False
    start_time = time.time()

    # get starting values
    plant.modify_coordinates(turbine_x,turbine_y)
    plant.simulate(1)

    aep = plant.annual_energy_kw()
    capacity = len(turbine_x)*turbine_rating
    annual_cost = fcr*capex_function(capacity) + om_function(capacity)
    if objective == "coe":
        coe = annual_cost/((1-additional_losses)*aep/1000.0) # $/MWh
        start_obj = coe
    elif objective == "profit":
        profit = (((1-additional_losses)*aep/1000.0)*ppa - annual_cost)/1E6 # millions of dollars
        start_obj = -profit

    best_sol = start_obj
    best_x = np.zeros_like(turbine_x)
    best_y = np.zeros_like(turbine_y)
    best_x[:] = turbine_x[:]
    best_y[:] = turbine_y[:]

    while converged == False:
        last_sol = best_sol
        print("nturbs: ", len(turbine_x))
        for i in range(len(turbine_x)):
            temp_x = np.delete(turbine_x,i)
            temp_y = np.delete(turbine_y,i)
            plant.modify_coordinates(temp_x,temp_y)
            plant.simulate(1)

            aep = plant.annual_energy_kw()
            capacity = len(temp_x)*turbine_rating
            annual_cost = fcr*capex_function(capacity) + om_function(capacity)
            if objective == "coe":
                coe = annual_cost/((1-additional_losses)*aep/1000.0) # $/MWh
                obj = coe
            elif objective == "profit":
                profit = (((1-additional_losses)*aep/1000.0)*ppa - annual_cost)/1E6 # millions of dollars
                obj = -profit

            if obj < best_sol:
                best_sol = obj
                best_x = np.zeros_like(temp_x)
                best_y = np.zeros_like(temp_y)
                best_x[:] = temp_x[:]
                best_y[:] = temp_y[:]

                file = open('%s'%save_filename, 'w')
                file.write('%s'%obj + '\n' + '\n')
                if objective == "coe":
                    file.write("COE: " + '%s'%coe + '\n' + '\n')
                elif objective == "profit":
                    file.write("profit: " + '%s'%profit + '\n' + '\n')
                file.write("turbine_x = np." + '%s'%repr(best_x) + '\n' + '\n')
                file.write("turbine_y = np." + '%s'%repr(best_y) + '\n' + '\n')
                file.write("nturbs = " + '%s'%len(best_x) + '\n')

        if best_sol == last_sol:
            converged = True
        else:
            turbine_x = best_x
            turbine_y = best_y

    
    run_time = time.time() - start_time
    print("start_obj: ", start_obj)
    print("opt_val: ", best_sol)
    print("time to run: ", run_time)

    print("xf: ", repr(best_x))
    print("yf: ", repr(best_y))
    print("nturbs: ", len(turbine_x))



    file = open('%s'%save_filename, 'w')
    file.write('%s'%best_sol + '\n' + '\n')
    if objective == "coe":
        file.write("COE: " + '%s'%best_sol + '\n' + '\n')
    elif objective == "profit":
        file.write("profit: " + '%s'%(-best_sol) + '\n' + '\n')
    file.write("turbine_x = np." + '%s'%repr(best_x) + '\n' + '\n')
    file.write("turbine_y = np." + '%s'%repr(best_y) + '\n' + '\n')

    file.write("nturbs = " + '%s'%len(best_x) + '\n')
    file.write("CONVERGED")
    file.close()
