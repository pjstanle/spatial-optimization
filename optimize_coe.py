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


def place_turbines(turbine_array):

    global x_array
    global y_array

    nturbs = int(np.sum(turbine_array))
    turbine_x = np.zeros(nturbs)
    turbine_y = np.zeros(nturbs)
    index = 0
    for i in range(len(turbine_array)):
        if int(turbine_array[i]) == 1:
            turbine_x[index] = x_array[i]
            turbine_y[index] = y_array[i]
            index += 1

    return turbine_x, turbine_y


def COE_obj(x):

    # calculate the wind farm AEP as a function of the grid design variables

    global plant
    global progress_filename
    global function_calls
    global capex_function
    global om_function

    # calculate x, y turbine layout from the grid design variables
    layout_x, layout_y = place_turbines(x)
    nturbs = len(layout_x)

    # check if there are zero turbines
    if nturbs == 0:
        return 1E16

    else:
        function_calls += 1
        plant.modify_coordinates(layout_x,layout_y)
        plant.simulate(1)
        aep = plant.annual_energy_kw()
        capacity = len(layout_x)*turbine_rating
        additional_losses = 0.088
        fcr = 0.063
        annual_cost = fcr*capex_function(capacity) + om_function(capacity)
        coe = annual_cost/((1-additional_losses)*aep/1000.0) # $/MWh

        if progress_filename:
            if os.path.exists('%s'%progress_filename):
                with open('%s'%progress_filename) as progress_file:
                    progress = progress_file.readlines()
                best_sol = float(progress[0])
            else:
                best_sol = 1E20

            if (coe) < best_sol:
                    file = open('%s'%progress_filename, 'w')
                    file.write('%s'%(coe) + '\n' + '\n')
                    file.write("COE: " + '%s'%(coe) + '\n' + '\n')
                    file.write("turbine_x = np." + '%s'%repr(layout_x) + '\n' + '\n')
                    file.write("turbine_y = np." + '%s'%repr(layout_y) + '\n' + '\n')
                    file.write("nturbs = " + '%s'%nturbs + '\n')
                    file.close()

        return coe


if __name__=="__main__":

    global x_array
    global y_array

    global plant
    global progress_filename
    global function_calls
    global capex_function
    global om_function

    # THESE SHOULD BE THE ONLY THINGS YOU NEED TO CHANGE BETWEEN RUNS

    turbine = int(sys.argv[1]) # 1: low 2: meduim 3: high
    setback_mult = float(sys.argv[2]) # float
    pricing = sys.argv[3] # realistic or ATB

    start_filename = "aep/turbine%s_setback%s.txt"%(turbine,setback_mult)
    x_array, y_array = read_aep_file(start_filename)

    # progress_filename = "coe_%s_ga/turbine%s_setback%s.txt"%(pricingturbine,setback_mult)
    progress_filename = "test_ga5.txt"

    if turbine==1:
        powercurve_filename = 'turbine_data/low_2_43r_116d_88h.txt'
        rotor_diameter = 116.0
        hub_height = 88.0
        turbine_rating = 2.430

        if pricing == "ATB":
            capex_cost = np.array([2*1727.0,1727.0,1594.0,1517.0,1490.0,1470.0,1430.0,1420.0]) # $/kW ATB
        elif pricing == "realistic":
            capex_cost = np.array([2*1786.0,1786.0,1622.0,1528.0,1494.0,1470.0,1421.0,1408.0]) # $/kW realistic
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==2:
        powercurve_filename = 'turbine_data/med_5_5r_175d_120h.txt'
        rotor_diameter = 175.0
        hub_height = 120.0
        turbine_rating = 5.5

        if pricing == "ATB":
            capex_cost = np.array([2*1438.0,1438.0,1316.0,1244.0,1199.0,1173.0,1133.0,1124.0]) # $/kW ATB
        elif pricing == "realistic":
            capex_cost = np.array([2*1599.0,1599.0,1421.0,1316.0,1250.0,1212.0,1153.0,1141.0]) # $/kW realistic
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==3:
        powercurve_filename = 'turbine_data/high_7r_200d_135h.txt'
        rotor_diameter = 200.0
        hub_height = 135.0
        turbine_rating = 7.0

        if pricing == "ATB":
            capex_cost = np.array([2*1072.0,1072.0,970.0,908.0,877.0,862.0,840.0,829]) # $/kW ATB
        elif pricing == "realistic":
            capex_cost = np.array([2*1382.0,1382.0,1124.0,966.0,887.0,849.0,792.0,765.0]) # $/kW realistic
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')


    def om_function(capacity):
        return 37.0*capacity*1000.0


    plant = init_wind_plant(hub_height,rotor_diameter,powercurve_filename)

    start_time = time.time()
    function_calls = 0

    npts = int(len(x_array))
    bits = np.zeros(npts,dtype=int)
    bounds = np.zeros((npts,2))
    variable_type = np.array([])
    for j in range(npts):
        bits[j] = 1
        bounds[j,:] = (0,1)
        variable_type = np.append(variable_type,"int")

    ga = GeneticAlgorithm()
    ga.bits = bits
    ga.bounds = bounds
    ga.variable_type = variable_type
    
    # ga.population_size = 100
    # ga.convergence_iters = 100 # number of iterations within the tolerance required for convergence
    ga.population_size = 100
    ga.convergence_iters = 25 # number of iterations within the tolerance required for convergence
    ga.max_generation = 1000 # maximum generations. Eveything I have run has converged in a couple of hundered

    ga.objective_function = COE_obj 

    ga.crossover_rate = 0.1 # this is unused for non-random crossover
    ga.mutation_rate = 0.01 # this is usually a pretty low value. I've seen 0.01 as pretty common. I used 0.02 because I am not allowing the best solution to mutate
    ga.tol = 1E-6 # convergence tolerance

    ga.optimize_ga(crossover="random",initialize="limit",print_progress=False,save_progress=False) # run the optimization. I have flags set to print progress and save the results every 1 iterations.
    
    opt_val = ga.optimized_function_value
    DVopt = ga.optimized_design_variables
    opt_x, opt_y = place_turbines(DVopt)
    
    run_time = time.time() - start_time


    file = open('%s'%progress_filename, 'w')
    file.write('%s'%(opt_val) + '\n' + '\n')
    file.write("COE: " + '%s'%(opt_val) + '\n' + '\n')
    file.write("turbine_x = np." + '%s'%repr(opt_x) + '\n' + '\n')
    file.write("turbine_y = np." + '%s'%repr(opt_y) + '\n' + '\n')
    file.write("nturbs = " + '%s'%len(opt_x) + '\n')
    file.write("run time: " + "%s"%run_time + '\n')
    file.write("function calls: " + "%s"%function_calls + '\n')
    file.write("CONVERGED")
    file.close()
