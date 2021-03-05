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


turbine_x = np.array([904364.14073513, 903347.39871299, 902839.02770192, 903855.76972406,
       903347.39871299, 901822.28567978, 904364.14073513, 903855.76972406,
       900805.54365764, 900297.17264657, 904364.14073513, 902330.65669085,
       901822.28567978, 901313.91466871, 899280.43062443, 903347.39871299,
       902839.02770192, 900297.17264657, 899788.8016355 , 898263.68860229,
       904364.14073513, 903855.76972406, 898772.05961336, 898263.68860229,
       901822.28567978, 900297.17264657, 898772.05961336, 896738.57556907,
       896230.204558  , 904364.14073513, 903855.76972406, 900805.54365764,
       896230.204558  , 902839.02770192, 902330.65669085, 901822.28567978,
       898263.68860229, 896230.204558  , 904364.14073513, 903855.76972406,
       900805.54365764, 899788.8016355 , 898263.68860229, 896230.204558  ,
       903347.39871299, 902839.02770192, 902330.65669085, 898263.68860229,
       896230.204558  , 904364.14073513, 903855.76972406, 901313.91466871,
       900805.54365764, 899788.8016355 , 896738.57556907, 903347.39871299,
       902839.02770192, 899788.8016355 , 898772.05961336, 896738.57556907,
       902330.65669085, 901822.28567978, 901313.91466871, 897246.94658015,
       896738.57556907, 900297.17264657, 897246.94658015, 900297.17264657,
       899280.43062443, 898772.05961336, 897246.94658015, 899280.43062443,
       897755.31759122, 897246.94658015, 896230.204558  , 897755.31759122,
       896230.204558  , 896738.57556907, 896230.204558  ])

turbine_y = np.array([171062.83585498, 171044.53637561, 171325.6491237 , 170182.89865199,
       170464.01140007, 171307.34964433, 169321.26092836, 169602.37367645,
       171289.05016497, 171570.16291305, 168740.73595282, 169865.18694517,
       170146.29969325, 170427.41244134, 171551.86343368, 168722.43647345,
       169003.54922154, 170409.11296197, 170690.22571006, 171533.56395432,
       167579.68600174, 167860.79874982, 170671.92623069, 170953.03897878,
       168404.72476663, 169248.06301089, 170091.40125515, 171215.85224749,
       171496.96499558, 166418.63605066, 166699.74879874, 168386.42528726,
       170916.44002004, 166681.44931937, 166962.56206746, 167243.67481555,
       169211.46405215, 170335.9150445 , 165257.58609957, 165538.69884766,
       167225.37533618, 167787.60083235, 168630.93907661, 169755.39006896,
       165239.28662021, 165520.39936829, 165801.51211638, 168050.41410107,
       169174.86509342, 164096.53614849, 164377.64889658, 165783.21263701,
       166064.3253851 , 166626.55088127, 168313.22736979, 164078.23666912,
       164359.34941721, 166046.02590573, 166608.2514019 , 167732.70239425,
       164059.93718976, 164341.04993784, 164622.16268593, 166871.06467062,
       167152.17741871, 164603.86320656, 166290.53969508, 164023.33823102,
       164585.56372719, 164866.67647528, 165710.01471954, 164005.03875165,
       164848.37699591, 165129.489744  , 165691.71524017, 164267.85202037,
       165111.19026463, 164249.552541  , 164530.66528909])

plot_poly(safe[0],plt.gca())

for i in range(len(turbine_x)):
    turb = plt.Circle((turbine_x[i],turbine_y[i]),radius=rotor_diameter/2.0,color="C0")
    plt.gca().add_patch(turb)


plt.axis("equal")
plt.axis("off")
plt.savefig("fig_pro.png",dpi=300)


