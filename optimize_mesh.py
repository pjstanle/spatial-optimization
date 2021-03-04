import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import time
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
from gradient_free import GeneticAlgorithm

from analyze_wind import init_wind_plant
import geopandas as gpd
import scipy.interpolate


def initialize_custom(self):
    global nrows
    global ncols

    n_ones = 1

    self.parent_population = np.zeros((self.population_size,self.nbits),dtype=int)
    for i in range(self.population_size):
        self.parent_population[i][0:n_ones] = 1
        np.random.shuffle(self.parent_population[i])
    self.offspring_population = np.zeros_like(self.parent_population)
    # one corner
    self.parent_population[0] = np.zeros(self.nbits)
    self.parent_population[0][0] = 1

    self.parent_population[1] = np.zeros(self.nbits)
    self.parent_population[1][ncols-1] = 1

    self.parent_population[2] = np.zeros(self.nbits)
    self.parent_population[2][-1] = 1

    self.parent_population[3] = np.zeros(self.nbits)
    self.parent_population[3][-ncols] = 1

    # two corners
    self.parent_population[4] = np.zeros(self.nbits)
    self.parent_population[4][0] = 1
    self.parent_population[4][ncols-1] = 1

    self.parent_population[5] = np.zeros(self.nbits)
    self.parent_population[5][0] = 1
    self.parent_population[5][-ncols] = 1

    self.parent_population[6] = np.zeros(self.nbits)
    self.parent_population[6][0] = 1
    self.parent_population[6][-1] = 1

    self.parent_population[7] = np.zeros(self.nbits)
    self.parent_population[7][ncols-1] = 1
    self.parent_population[7][-ncols] = 1

    self.parent_population[8] = np.zeros(self.nbits)
    self.parent_population[8][ncols-1] = 1
    self.parent_population[8][-1] = 1

    self.parent_population[9] = np.zeros(self.nbits)
    self.parent_population[9][-ncols] = 1
    self.parent_population[9][-1] = 1

    # three corners

    self.parent_population[10] = np.zeros(self.nbits)
    self.parent_population[10][0] = 1
    self.parent_population[10][ncols-1] = 1
    self.parent_population[10][-ncols] = 1

    self.parent_population[11] = np.zeros(self.nbits)
    self.parent_population[11][0] = 1
    self.parent_population[11][ncols-1] = 1
    self.parent_population[11][-1] = 1

    self.parent_population[12] = np.zeros(self.nbits)
    self.parent_population[12][-1] = 1
    self.parent_population[12][ncols-1] = 1
    self.parent_population[12][-ncols] = 1

    self.parent_population[13] = np.zeros(self.nbits)
    self.parent_population[13][0] = 1
    self.parent_population[13][-1] = 1
    self.parent_population[13][-ncols] = 1

    # four corners
    self.parent_population[14] = np.zeros(self.nbits)
    self.parent_population[14][0] = 1
    self.parent_population[14][ncols-1] = 1
    self.parent_population[14][-ncols] = 1
    self.parent_population[14][-1] = 1

   
def second_sweep(obj_function,nrows,ncols,start=np.array([1E20]),ntech=1): 

    nlocs = nrows*ncols
    if start[0] != 1E20:
        plant_array = start
    else:
        plant_array = np.zeros(nlocs)

    plant_solution = obj_function(plant_array)

    converged = False
    phase = np.array(["search","switch_x","switch_y"])
    phase_iter = 2 # start with the search phase
    
    converged_counter = 0

    while converged == False:
        phase_iter = (phase_iter+1)%len(phase)
        current_phase = phase[phase_iter]

        # this is the search phase
        # sweep through every point, and try every technology at this point
        # if the solution is better, keep the new change
        
        if current_phase == "search":
            print("search")
            order = np.arange(nlocs)
            np.random.shuffle(order)
            for i in range(nlocs):
                temp_array = np.zeros_like(plant_array)
                temp_array[:] = plant_array[:]
                for j in range(ntech):
                    temp_array[order[i]] = (temp_array[order[i]]+1)%(ntech+1)
                    temp_solution = obj_function(temp_array)
                    if temp_solution < plant_solution:
                        plant_solution = temp_solution
                        plant_array[order[i]] = temp_array[order[i]]
                        converged_counter = 0
                        print(plant_solution)
                    else:
                        converged_counter += 1
                
                if converged_counter > (len(phase)+1)*nlocs:
                    converged = True
                    break
                    
        elif current_phase == "switch_x":
            print("switch x")
            order = np.arange(nlocs)
            np.random.shuffle(order)
            for i in range(nlocs):
                temp_array = np.zeros_like(plant_array)
                temp_array[:] = plant_array[:]
                try:
                    if temp_array[order[i]] != temp_array[order[i]+1]:
                        temp_array[order[i]],temp_array[order[i]+1] = temp_array[order[i]+1],temp_array[order[i]]
                        temp_solution = obj_function(temp_array)
                    
                        if temp_solution < plant_solution:
                            plant_solution = temp_solution
                            plant_array[i] = temp_array[i]
                            converged_counter = 0
                            print(plant_solution)
                        else:
                            converged_counter += 1
                    
                    else:
                        converged_counter += 1

                except:
                    converged_counter += 1

                if converged_counter > (len(phase)+1)*nlocs:
                        converged = True
                        break


        elif current_phase == "switch_y":
            print("switch y")
            order = np.arange(nlocs)
            np.random.shuffle(order)
            for i in range(nlocs):
                temp_array = np.zeros_like(plant_array)
                temp_array[:] = plant_array[:]
                try:
                    if temp_array[order[i]] != temp_array[order[i]+ncols]:
                        temp_array[order[i]],temp_array[order[i]+ncols] = temp_array[order[i]+ncols],temp_array[order[i]]
                        temp_solution = obj_function(temp_array)
                    
                        if temp_solution < plant_solution:
                            plant_solution = temp_solution
                            plant_array[i] = temp_array[i]
                            converged_counter = 0
                            print(plant_solution)
                        else:
                            converged_counter += 1
                    
                    else:
                        converged_counter += 1

                except:
                    converged_counter += 1

                if converged_counter > (len(phase)+1)*nlocs:
                        converged = True
                        break

    return plant_solution, plant_array


def mesh_poly(poly,rotor_diameter):
    x = poly.exterior.coords.xy[0]
    y = poly.exterior.coords.xy[1]
    cx = poly.centroid.x
    cy = poly.centroid.y

    right = max(x)-cx
    left = cx-min(x)
    top = max(y)-cy
    bot = cy-min(y)

    N = 1
    nleft = int(left/(N*rotor_diameter))+2
    nright = int(right/(N*rotor_diameter))+2
    ntop = int(top/(N*rotor_diameter))+2
    nbot = int(bot/(N*rotor_diameter))+2

    xleft = np.linspace(min(x),cx,nleft)
    xright = np.linspace(cx,max(x),nright)
    ybot = np.linspace(min(y),cy,nbot)
    ytop = np.linspace(cy,max(y),ntop)
    # xleft = np.linspace(cx-(nleft-1)*rotor_diameter,cx,nleft)
    # xright = np.linspace(cx,cx+(nright-1)*rotor_diameter,nright)
    # ybot = np.linspace(cy-(nbot-1)*rotor_diameter,cy,nbot)
    # ytop = np.linspace(cy,cy+(ntop-1)*rotor_diameter,ntop)

    xgrid = np.append(xleft[0:-1],xright)
    ygrid = np.append(ybot[0:-1],ytop)

    nrows = len(ygrid)
    ncols = len(xgrid)

    xlocs,ylocs = np.meshgrid(xgrid,ygrid)
    xlocs = np.ndarray.flatten(xlocs)
    ylocs = np.ndarray.flatten(ylocs)

    for i in range(len(xlocs)):
        point = Point(xlocs[i],ylocs[i])
        if poly.contains(point) or poly.touches(point):
            pass
        else:
            p1, p2 = nearest_points(poly, point)
            # pt = p1.wkt
            xlocs[i] = p1.x
            ylocs[i] = p1.y
        

    return xlocs,ylocs,ncols,nrows


def get_xy(A):
    x = np.zeros(len(A))
    y = np.zeros(len(A))
    for i in range(len(A)):
        x[i] = A[i][0]
        y[i] = A[i][1]
    return x,y


def plot_poly(geom,ax):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        x,y = get_xy(exterior_coords)
        ax.plot(x,y,"k")

        for interior in geom.interiors:
            interior_coords = interior.coords[:]
            x,y = get_xy(interior_coords)
            ax.plot(x,y,"b")

    elif geom.type == 'MultiPolygon':

        for part in geom:
            exterior_coords = part.exterior.coords[:]
            x,y = get_xy(exterior_coords)
            ax.plot(x,y,"k")
            for interior in part.interiors:
                interior_coords = interior.coords[:]
                x,y = get_xy(interior_coords)
                ax.plot(x,y,"b")


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


def calc_spacing(layout_x,layout_y):

    #calculate the spacing between each turbine and every other turbine (without repeating)
    nTurbs = len(layout_x)
    npairs = int((nTurbs*(nTurbs-1))/2)
    spacing = np.zeros(npairs)

    ind = 0
    for i in range(nTurbs):
        for j in range(i,nTurbs):
            if i != j:
                spacing[ind] = np.sqrt((layout_x[i]-layout_x[j])**2+(layout_y[i]-layout_y[j])**2)
                ind += 1

    return spacing


def calc_boundary(layout_x,layout_y):
    global boundary_poly
    meets_constraints = True
    for i in range(len(layout_x)):
        point = Point(layout_x[i], layout_y[i])
        if (boundary_poly.contains(point)==True or boundary_poly.touches(point)==True):
            pass
        else: 
            meets_constraints = False
            break
    
    return meets_constraints


def AEP_obj(x):

    # calculate the wind farm AEP as a function of the grid design variables

    global turbine_x
    global turbine_y
    global plant

    global min_spacing
    global max_turbs
    global rotor_diameter
    global scale
    global progress_filename
    global function_calls

    global iter_number
    global nzones
    global turbs_per_zone

    # calculate x, y turbine layout from the grid design variables
    new_x, new_y = place_turbines(x)
    
    layout_x = np.append(turbine_x,new_x)
    layout_y = np.append(turbine_y,new_y)

    nturbs = len(layout_x)

    # meets_boundary_constraint = calc_boundary(new_x,new_y)
    # check if there are too many turbines
    if nturbs > max_turbs:
        return 1E16

    # check if there are zero turbines
    elif nturbs == 0:
        return 1E16

    # elif meets_boundary_constraint == False:
    #     return 1E16

    # check the boundary constraint
    
    else:
        # check spacing constraint
        if len(layout_x) > 1:
            spacing = calc_spacing(layout_x,layout_y)
            spacing_con = np.min(spacing) - min_spacing*rotor_diameter
        else:
            spacing_con = 0.0

        # if spacing constraint is violated
        if spacing_con < 0.0:
            return 1E16
        
        # if spacing constraint is not violated
        else:
            function_calls += 1
            plant.modify_coordinates(layout_x,layout_y)
            plant.simulate(1)
            aep = plant.annual_energy_kw()

            if progress_filename:
                if os.path.exists('%s'%progress_filename):
                    with open('%s'%progress_filename) as progress_file:
                        progress = progress_file.readlines()
                    best_sol = float(progress[0])
                else:
                    best_sol = 1E20

                if (-aep/scale) < best_sol:
                        turbs_per_zone[iter_number-1] = len(new_x)
                        file = open('%s'%progress_filename, 'w')
                        file.write('%s'%(-aep/scale) + '\n' + '\n')
                        file.write("AEP: " + '%s'%(aep/scale) + '\n' + '\n')
                        file.write("turbine_x = np." + '%s'%repr(layout_x) + '\n' + '\n')
                        file.write("turbine_y = np." + '%s'%repr(layout_y) + '\n' + '\n')

                        file.write("nturbs = " + '%s'%nturbs + '\n')
                        file.write("turbs_per_zone = " + '%s'%turbs_per_zone  + '\n')
                        file.write("zone " + '%s/%s'%(iter_number,nzones) + '\n')
                        file.close()

            return -aep/scale


def COE_obj(x):
    # calculate the wind farm COE as a function of the grid design variables

    global turbine_x
    global turbine_y
    global plant

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

    new_x, new_y = place_turbines(x)
    
    layout_x = np.append(turbine_x,new_x)
    layout_y = np.append(turbine_y,new_y)

    nturbs = len(layout_x)

    # check if there are too many turbines
    if nturbs > max_turbs:
        return 1E16

    # check if there are zero turbines
    elif nturbs == 0:
        return 1E16
    
    else:
        # check spacing constraint
        if len(layout_x) > 1:
            spacing = calc_spacing(layout_x,layout_y)
            spacing_con = np.min(spacing) - min_spacing*rotor_diameter
        else:
            spacing_con = 0.0

        # if spacing constraint is violated
        if spacing_con < 0.0:
            return 1E16
        
        # if spacing constraint is not violated
        else:
            function_calls += 1
            plant.modify_coordinates(layout_x,layout_y)
            plant.simulate(1)
            aep = plant.annual_energy_kw()

            capacity = nturbs*turbine_rating
            annual_cost = fcr*capex_function(capacity) + om_function(capacity)
            coe = annual_cost/((1-additional_losses)*aep/1000.0) # $/MWh

            if progress_filename:
                if os.path.exists('%s'%progress_filename):
                    with open('%s'%progress_filename) as progress_file:
                        progress = progress_file.readlines()
                    best_sol = float(progress[0])
                else:
                    best_sol = 1E20

                if coe < best_sol:
                        file = open('%s'%progress_filename, 'w')
                        file.write('%s'%coe + '\n' + '\n')
                        file.write("COE: " + '%s'%coe + '\n' + '\n')
                        file.write("turbine_x = np." + '%s'%repr(layout_x) + '\n' + '\n')
                        file.write("turbine_y = np." + '%s'%repr(layout_y) + '\n' + '\n')

                        file.write("nturbs = " + '%s'%nturbs + '\n')
                        file.write("zone " + '%s/%s'%(iter_number,nzones) + '\n')
                        file.close()

            return coe    


if __name__=="__main__":

    global turbine_x
    global turbine_y
    global plant
    global boundary_poly
    global x_array
    global y_array

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
    global turbs_per_zone

    global nrows
    global ncols

    # THESE SHOULD BE THE ONLY THINGS YOU NEED TO CHANGE BETWEEN RUNS

    turbine = 2
    setback_mult = 2.0
    # run_number = 1
    objective = "aep"

    debug = False
    # try:
    #     print("try")
    #     debug = bool(float(sys.argv[5]))
    # except IndexError:
    #     print("except")
    #     debug = False

    # 0 INITIALIZE GLOBAL VARIABLES
    turbine_x = np.array([])
    turbine_y = np.array([])

    if turbine==1:
        powercurve_filename = 'turbine_data/low_2_43r_116d_88h.txt'
        rotor_diameter = 116.0
        hub_height = 88.0
        turbine_rating = 2.430

        capex_cost = np.array([2*1727.0,1727.0,1594.0,1517.0,1490.0,1470.0,1430.0,1420.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==2:
        powercurve_filename = 'turbine_data/med_5_5r_175d_120h.txt'
        rotor_diameter = 175.0
        hub_height = 120.0
        turbine_rating = 5.5

        capex_cost = np.array([2*1438.0,1438.0,1316.0,1244.0,1199.0,1173.0,1133.0,1124.0]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    elif turbine==3:
        powercurve_filename = 'turbine_data/high_7r_200d_135h.txt'
        rotor_diameter = 200.0
        hub_height = 135.0
        turbine_rating = 7.0

        capex_cost = np.array([2*1072.0,1072.0,970.0,908.0,877.0,862.0,840.0,829]) # $/kW
        capex_size = np.array([1.0,20.0,50.0,100.0,150.0,200.0,400.0,1000.0]) # MW
        cost = capex_size*capex_cost*1000.0
        capex_function = scipy.interpolate.interp1d(capex_size, cost, kind='cubic')

    additional_losses = 0.088
    fcr = 0.063
    
    def om_function(capacity):
        return 37.0*capacity*1000.0

    plant = init_wind_plant(hub_height,rotor_diameter,powercurve_filename)
    tip_height = hub_height+rotor_diameter/2.0
    buffer_distance = setback_mult*tip_height

    min_spacing = 5.0
    max_turbs = 300
    scale = 1E6

    if os.path.exists('%s'%objective):
        pass
    else:
        os.mkdir(objective)
        
    # progress_filename = "debug.txt"
    # figname = "debug.png"
    if debug == True:
        progress_filename = "debug1.txt"
        if os.path.exists('%s'%progress_filename):
            os.remove(('%s'%progress_filename))
        
        # figname = "debug.png"
    else:
        progress_filename = "%s/turbine%s_setback%s_mesh.txt"%(objective,turbine,setback_mult)
        # figname = "%s/turbine%s_setback%s_mesh.png"%(objective,turbine,setback_mult)

    function_calls = 0

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

    safe = gpd.GeoSeries(Polygon(([low_x,low_y],[high_x,low_y],[high_x,high_y],[low_x,high_y])))

    if setback_mult != 0.0:
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

    else:
        polys = list(safe)
    
    # 2 LOOP THROUGH EACH SAFE AREA AND OPTIMIZE
    nzones = len(polys)
    areas = np.zeros(nzones)
    for i in range(nzones):
        boundary_poly = polys[i]
        areas[i] = boundary_poly.area
    order = np.argsort(areas)
    # order = np.flip(order)
    # order = np.arange(nzones)
    # np.random.shuffle(order)

    start_time = time.time()

    # if debug == True:
    #     iters = 3
    # else:
    #     iters = nzones
    iters = nzones

    turbs_per_zone = np.zeros(nzones,dtype=int)
    for i in range(iters):
        iter_number = i+1

        boundary_poly = polys[order[i]]
        x_array, y_array, ncols, nrows = mesh_poly(boundary_poly,rotor_diameter)
        npts = len(x_array)

        ga = GeneticAlgorithm()

        bits = np.zeros(npts,dtype=int)
        bounds = np.zeros((npts,2))
        variable_type = np.array([])
        for i in range(npts):
            bits[i] = 1
            bounds[i,:] = (0,1)
            variable_type = np.append(variable_type,"int")
        ga.bits = bits
        ga.bounds = bounds
        ga.variable_type = variable_type
        
        ga.population_size = 100
        ga.convergence_iters = 20 # number of iterations within the tolerance required for convergence
        ga.max_generation = 1000 # maximum generations. Eveything I have run has converged in a couple of hundered

        if objective == "aep":
            ga.objective_function = AEP_obj 
        elif objective == "coe":
            ga.objective_function = COE_obj 

        ga.crossover_rate = 0.1 # this is unused for non-random crossover
        ga.mutation_rate = 0.02 # this is usually a pretty low value. I've seen 0.01 as pretty common. I used 0.02 because I am not allowing the best solution to mutate
        ga.tol = 1E-6 # convergence tolerance

        # ga.optimize_ga(initialize="limit",print_progress=False,save_progress=False) # run the optimization. I have flags set to print progress and save the results every 1 iterations.
        ga.optimize_ga(crossover="chunk",initialize="custom",initialize_custom=initialize_custom,print_progress=False,save_progress=False) # run the optimization. I have flags set to print progress and save the results every 1 iterations.
        
        if objective == "aep":
            opt_val = -ga.optimized_function_value
            DVopt = ga.optimized_design_variables
            if opt_val > 0.0:
                # sweep through greedily
                # sol, arr = second_sweep(AEP_obj,nrows,ncols,start=DVopt)
                opt_x, opt_y = place_turbines(DVopt) 
                turbine_x = np.append(turbine_x,opt_x)
                turbine_y = np.append(turbine_y,opt_y)
        
        elif objective == "coe":
            opt_val = ga.optimized_function_value
            DVopt = ga.optimized_design_variables
            if opt_val < 1E3:
                opt_x, opt_y = place_turbines(DVopt) 
                turbine_x = np.append(turbine_x,opt_x)
                turbine_y = np.append(turbine_y,opt_y)

    
    run_time = time.time() - start_time
    print("opt_val: ", opt_val)
    print("time to run: ", run_time)
    print("function calls: ", function_calls)

    print("xf: ", repr(turbine_x))
    print("yf: ", repr(turbine_y))
    print("nturbs: ", len(turbine_x))

    # if figname:
    #     for i in range(len(polys)):
    #         plot_poly(polys[i],plt.gca())
    #     for i in range(len(turbine_x)):
    #         turb = plt.Circle((turbine_x[i],turbine_y[i]),radius=rotor_diameter/2.0,edgecolor=None,facecolor="C0")
    #         plt.gca().add_patch(turb)

    #     plt.axis("equal")
    #     plt.savefig(figname,dpi=300)
    #     plt.show()