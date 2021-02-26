import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.wkt
import shapely

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
        ax.plot(x,y,"--k",linewidth=0.5)

        for interior in geom.interiors:
            interior_coords = interior.coords[:]
            x,y = get_xy(interior_coords)
            ax.plot(x,y,"b")

    elif geom.type == 'MultiPolygon':

        for part in geom:
            exterior_coords = part.exterior.coords[:]
            x,y = get_xy(exterior_coords)
            ax.plot(x,y,"--k",linewidth=0.5)

            for interior in part.interiors:
                interior_coords = interior.coords[:]
                x,y = get_xy(interior_coords)
                ax.plot(x,y,"b")


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

rotor_diameter = 116.0
hub_height = 88.0
# rotor_diameter = 175.0
# hub_height = 120.0
# rotor_diameter = 200.0
# hub_height = 135.0
setback_mult = 0.0
buffer_distance = setback_mult*(hub_height+rotor_diameter/2.0)

# d1 = gpd.read_file("blue_creek_data/feature_parts/buildings.gpkg")
# buildings = d1["geometry"][0]
# for i in range(len(buildings)):
#     ex = buildings[i].buffer(buffer_distance)
#     safe = safe.difference(ex)


# d1 = gpd.read_file("blue_creek_data/feature_parts/roads.gpkg")
# roads = d1["geometry"][0]
# for i in range(len(roads)):
#     ex = roads[i].buffer(buffer_distance)
#     safe = safe.difference(ex)

# d1 = gpd.read_file("blue_creek_data/feature_parts/transmission.gpkg")
# transmission = d1["geometry"][0]
# for i in range(len(transmission)):
#     ex = transmission[i].buffer(buffer_distance)
#     safe = safe.difference(ex)

# d1 = gpd.read_file("blue_creek_data/feature_parts/rails.gpkg")
# rails = d1["geometry"][0]
# for i in range(len(rails)):
#     ex = rails[i].buffer(buffer_distance)
#     safe = safe.difference(ex)


turbine_x = np.array([896848.77774991, 897886.51905495, 898924.26036   , 899962.00166505,
       896566.3805702 , 897604.12187525, 898641.8631803 , 899679.60448534,
       900717.34579039, 901755.08709544, 902792.82840048, 896283.9833905 ,
       897321.72469555, 898359.46600059, 899397.20730564, 900434.94861069,
       901472.68991573, 902510.43122078, 903548.17252583, 897039.32751584,
       898077.06882089, 899114.81012594, 900152.55143098, 901190.29273603,
       902228.03404108, 903265.77534612, 904303.51665117, 899870.15425128,
       900907.89555632, 901945.63686137, 902983.37816642, 904021.11947147,
       902700.98098671, 903738.72229176])

turbine_y = np.array([165769.76208826, 165212.5067764 , 164655.25146454, 164097.99615268,
       167654.12439532, 167096.86908346, 166539.6137716 , 165982.35845974,
       165425.10314788, 164867.84783601, 164310.59252415, 169538.48670238,
       168981.23139052, 168423.97607866, 167866.7207668 , 167309.46545494,
       166752.21014307, 166194.95483121, 165637.69951935, 170865.59369758,
       170308.33838572, 169751.08307386, 169193.827762  , 168636.57245014,
       168079.31713827, 167522.06182641, 166964.80651455, 171078.19006906,
       170520.9347572 , 169963.67944533, 169406.42413347, 168849.16882161,
       171290.78644053, 170733.53112867])

plot_poly(safe[0],plt.gca())

for i in range(len(turbine_x)):
    turb = plt.Circle((turbine_x[i],turbine_y[i]),radius=rotor_diameter/2.0,color="C0")
    plt.gca().add_patch(turb)


plt.axis("equal")
plt.axis("off")
plt.savefig("fig.png",dpi=300)


