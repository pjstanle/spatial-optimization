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


def get_grid_locs(nrows,ncols,farm_width,farm_height,shear,rotation,center_x,center_y):

    global boundary_poly

    # create grid
    nrows = int(nrows)
    ncols = int(ncols)
    xlocs = np.linspace(0.0,farm_width,ncols)
    ylocs = np.linspace(0.0,farm_height,nrows)
    y_spacing = ylocs[1]-ylocs[0]
    nturbs = nrows*ncols
    layout_x = np.zeros(nturbs)
    layout_y = np.zeros(nturbs)
    turb = 0
    for i in range(nrows):
        for j in range(ncols):
            layout_x[turb] = xlocs[j] + float(i)*y_spacing*np.tan(shear)
            layout_y[turb] = ylocs[i]
            turb += 1
    
    # rotate
    rotate_x = np.cos(rotation)*layout_x - np.sin(rotation)*layout_y
    rotate_y = np.sin(rotation)*layout_x + np.cos(rotation)*layout_y

    # move center of grid
    rotate_x = (rotate_x - np.mean(rotate_x)) + center_x
    rotate_y = (rotate_y - np.mean(rotate_y)) + center_y

    # get rid of points outside of boundary and violate setback constraints

    meets_constraints = np.zeros(len(rotate_x),dtype=int)
    for i in range(len(rotate_x)):
        point = Point(rotate_x[i], rotate_y[i])
        if (boundary_poly.contains(point)==True or boundary_poly.touches(point)==True):
                        meets_constraints[i] = 1

    # arrange final x,y points
    return_x = np.zeros(sum(meets_constraints))
    return_y = np.zeros(sum(meets_constraints))
    index = 0
    for i in range(len(meets_constraints)):
        if meets_constraints[i] == 1:
            return_x[index] = rotate_x[i]
            return_y[index] = rotate_y[i]
            index += 1

    return return_x, return_y


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

    nrows = x[0]
    ncols = x[1]
    farm_width = x[2]
    farm_height = x[3]
    shear = x[4]
    rotation = x[5]
    center_x = x[6]
    center_y = x[7]

    # calculate x, y turbine layout from the grid design variables
    new_x, new_y = get_grid_locs(nrows,ncols,farm_width,farm_height,shear,rotation,center_x,center_y)
    
    layout_x = np.append(turbine_x,new_x)
    layout_y = np.append(turbine_y,new_y)

    nturbs = len(layout_x)

    # check if there are too many turbines
    if nturbs > max_turbs:
        return 1E16

    # check if there are zero turbines
    elif nturbs == 0:
        return 0.0
    
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
                        file = open('%s'%progress_filename, 'w')
                        file.write('%s'%(-aep/scale) + '\n' + '\n')
                        file.write("AEP: " + '%s'%(aep/scale) + '\n' + '\n')
                        file.write("turbine_x = np." + '%s'%repr(layout_x) + '\n' + '\n')
                        file.write("turbine_y = np." + '%s'%repr(layout_y) + '\n' + '\n')

                        file.write("nturbs = " + '%s'%nturbs + '\n')
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

    nrows = x[0]
    ncols = x[1]
    farm_width = x[2]
    farm_height = x[3]
    shear = x[4]
    rotation = x[5]
    center_x = x[6]
    center_y = x[7]

    # calculate x, y turbine layout from the grid design variables
    new_x, new_y = get_grid_locs(nrows,ncols,farm_width,farm_height,shear,rotation,center_x,center_y)
    
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

    # THESE SHOULD BE THE ONLY THINGS YOU NEED TO CHANGE BETWEEN RUNS

    turbine = int(sys.argv[1]) # 1: low 2: meduim 3: high
    setback_mult = float(sys.argv[2]) # float
    run_number = int(sys.argv[3])
    objective = str(sys.argv[4])
    try:
        print("try")
        debug = bool(float(sys.argv[5]))
    except IndexError:
        print("except")
        debug = False

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
        
    if debug == True:
        progress_filename = "%s/debug_%s.txt"%(objective,run_number)
        if os.path.exists('%s'%progress_filename):
            os.remove(('%s'%progress_filename))
        
        figname = "%s/debug_%s.png"%(objective,run_number)
    else:
        progress_filename = "%s/turbine%s_setback%s_run%s.txt"%(objective,turbine,setback_mult,run_number)
        figname = "%s/turbine%s_setback%s_run%s.png"%(objective,turbine,setback_mult,run_number)

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
    # order = np.arange(nzones)
    # np.random.shuffle(order)

    areas = np.zeros(nzones)
    for i in range(nzones):
        boundary_poly = polys[i]
        areas[i] = boundary_poly.area
    order = np.argsort(areas)
    # order = np.flip(order)

    start_time = time.time()

    if debug == True:
        iters = 3
    else:
        iters = nzones
    for i in range(iters):
        iter_number = i+1
        boundary_poly = polys[order[i]]
        bx,by = polys[order[i]].exterior.coords.xy
        width = np.max(bx) - np.min(bx)
        height = np.max(by) - np.min(by)

        cx = (np.min(bx)+np.max(bx))/2.0
        cy = (np.min(by)+np.max(by))/2.0

        ga = GeneticAlgorithm()
        if setback_mult != 0.0:
            ga.bits = np.array([3,3,8,8,8,8,8,8]) # the number of bits assigned to each variable. 
            ga.bounds = np.array([(2.0,8.0),(2.0,8.0),(1.0,2*width),(1.0,2*height),(-np.pi,np.pi),(0.0,2.0*np.pi),(cx-width/4.0,cx+width/4.0),(cy-height/4.0,cy+height/4.0)]) # bounds for each design variable
        else:
            ga.bits = np.array([5,5,8,8,8,8,8,8]) # the number of bits assigned to each variable. 
            ga.bounds = np.array([(2.0,32.0),(2.0,32.0),(1.0,2*width),(1.0,2*height),(-np.pi,np.pi),(0.0,2.0*np.pi),(cx-width/4.0,cx+width/4.0),(cy-height/4.0,cy+height/4.0)]) # bounds for each design variable

        ga.variable_type = np.array(["int","int","float","float","float","float","float","float"]) # variable types (float or int)
        if debug == True:
            ga.population_size = 25
            ga.convergence_iters = 10 # number of iterations within the tolerance required for convergence
        else:
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

        ga.optimize_ga(print_progress=False,save_progress=False) # run the optimization. I have flags set to print progress and save the results every 1 iterations.
        
        if objective == "aep":
            opt_val = -ga.optimized_function_value
            DVopt = ga.optimized_design_variables
            if opt_val > 0.0:
                opt_x, opt_y = get_grid_locs(DVopt[0],DVopt[1],DVopt[2],DVopt[3],DVopt[4],DVopt[5],DVopt[6],DVopt[7]) 
                turbine_x = np.append(turbine_x,opt_x)
                turbine_y = np.append(turbine_y,opt_y)
        
        elif objective == "coe":
            opt_val = ga.optimized_function_value
            DVopt = ga.optimized_design_variables
            if opt_val < 1E3:
                opt_x, opt_y = get_grid_locs(DVopt[0],DVopt[1],DVopt[2],DVopt[3],DVopt[4],DVopt[5],DVopt[6],DVopt[7]) 
                turbine_x = np.append(turbine_x,opt_x)
                turbine_y = np.append(turbine_y,opt_y)

    
    run_time = time.time() - start_time
    print("opt_val: ", opt_val)
    print("time to run: ", run_time)
    print("function calls: ", function_calls)

    print("xf: ", repr(turbine_x))
    print("yf: ", repr(turbine_y))
    print("nturbs: ", len(turbine_x))

    if figname:
        plot_poly(safe[0],plt.gca())
        for i in range(len(turbine_x)):
            turb = plt.Circle((turbine_x[i],turbine_y[i]),radius=rotor_diameter/2.0,edgecolor=None,facecolor="C0")
            plt.gca().add_patch(turb)

        plt.axis("equal")
        plt.savefig(figname,dpi=300)