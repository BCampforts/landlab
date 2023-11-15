
""" 01/06/2022
Script identique au code de Benjamin campforts, partie 'Run on real DEM' """



# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid, imshow_grid_at_node
from landlab.components import (FlowAccumulator,
                                Space,
                                DepthDependentTaylorDiffuser,
                                ExponentialWeathererIntegrated, SpaceLargeScaleEroder, PriorityFloodFlowRouter,
                                Tr_L_diff)
from bmi_topography import Topography
from landlab.io.netcdf import write_netcdf
from landlab.io import read_esri_ascii
from landlab.plot import imshowhs_grid
from landlab.components import SpaceLargeScaleEroder

# Download mini DEM
topo = Topography(
    dem_type="SRTMGL1",
    south=44.175,
    north=44.205,
    west=6.48,
    east=6.52,
    output_format="AAIGrid",
    cache_dir="DEMData//"
    )

 # 44° 8' 25.026" N, 6° 21' 34.1568" E

fname = topo.fetch()
print(fname)
dem = topo.load()
print(dem)

# Space
K_sed = 1e-3,
K_br = 1e-7,
phi_poro = 0,
H_star = 0.1,
v_s = 0.1,
m_sp = 0.5,
n_sp = 1.0,
sp_crit_br = 0,
sp_crit_sed = 0

# TR_L
transportLengthCoefficient = 30
erodibility = 0.1
slope_crit = 1.2
soil_Z = 0.1

# Rate of soil formation
P0 = 0.001

# Runoff
runoff_rate = 1

# Read DEM as Lanlab grid
grid_geog, elev = read_esri_ascii(fname, name='topographic__elevation')

# Show dem
# plt.figure()
# imshowhs_grid(grid_geog, 'topographic__elevation', cmap='terrain',
#               grid_units=("deg", "deg"), var_name="Elevation (m)", cbar_label_color='white')
# plt.show()

# Reproject
grid = RasterModelGrid(
    (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns), xy_spacing=30.0
)
z = grid.add_field("topographic__elevation", elev, at="node")

bed = grid.add_zeros('bedrock__elevation', at='node')
soil = grid.add_zeros('soil__depth', at='node')
# Add soil thickness reletive to elevation
soil_rel = (z - min(z)) / (max(z) - min(z))
soil_init = 0.1
print('soil reel', np.mean(soil_rel))


#soil[:] = soil_rel
soil[:] = soil_init
soil[grid.boundary_nodes] = 0
bed[:] = z
z[:] = bed + soil

# Instantiate Components
fdir = PriorityFloodFlowRouter(grid,
                               surface="topographic__elevation",
                               flow_metric='D8',
                               runoff_rate=1)

diff = Tr_L_diff(
    grid,
    erodibility=erodibility,
    slope_crit=slope_crit,
    depthDependent=True,
    depositOnBoundaries=False,
    H_star=soil_Z,
    transportLengthCoefficient=30)

fluv = SpaceLargeScaleEroder(grid, K_sed=1e-3, K_br=1e-7,
                             phi=0, H_star=0.1, v_s=0.1, m_sp=0.5, n_sp=1.0,
                             sp_crit_br=0, sp_crit_sed=0)

# Show dem
# plt.figure()
# imshowhs_grid(grid, 'topographic__elevation', cmap='terrain',
#               grid_units=("m", "m"), var_name="Elevation (m)", cbar_label_color='white')
# plt.show()

# Show soil thickness
plt.figure()
imshowhs_grid(grid, 'topographic__elevation', plot_type='Drape1', drape1=soil, cmap='jet', alpha=0.5,
              grid_units=("m", "m"), var_name="Soil depth (m)", cbar_label_color='white', limits=(0, 0.5))
plt.show()
#
# # Slope
# fdir.run_one_step()
# plt.figure()
# imshowhs_grid(grid, z, plot_type='Drape1', drape1=grid.at_node["topographic__steepest_slope"],
#               var_name='Slope, m/m', alpha=0.5, cmap='jet', cbar_label_color='white')
# plt.show()

# Model run
dt = 1
totT = 50
plotDiff = totT / 4
thresholdPlot = plotDiff
fdir.run_one_step()
DA = grid.at_node['drainage_area']
outlet = np.argmax(DA)

# Store output fluxes
outlet_flux_diff_var = []
outlet_flux_fluv_var = []
for t in range(int(totT / dt)):
    # expweath.calc_soil_prod_rate()
    # soil[grid.core_nodes]+= grid.at_node['soil_production__rate'][grid.core_nodes]*dt
    # bed[grid.core_nodes]-= grid.at_node['soil_production__rate'][grid.core_nodes]*dt
    # added_soil = np.sum(grid.at_node['soil_production__rate']*dt)

    # soil[:] += soil_rel * P0 * dt
    # added_soil = np.sum(soil_rel * P0 * dt)
    # soil[:] += soil_init * P0 * dt
    # added_soil = np.sum(soil_init * P0 * dt)
    #print('added soil', added_soil)

    z[:] = bed + soil

    # bed[grid.core_nodes]+=U*dt

    fdir.run_one_step()

    if np.mod(t, 1) == 0:
        fluv.run_one_step(dt)


    plt.figure()
    plt.title(str(np.round(t * dt, 2)) + ' year, avt diff')
    imshowhs_grid(grid, 'topographic__elevation', plot_type='Drape1', drape1=np.sqrt(soil), cmap='jet', alpha=0.5,
                  grid_units=("m", "m"), var_name="Soil depth (m)", cbar_label_color='white', limits=(0, 0.5))
    plt.show()

    diff.run_one_step(dt)

    plt.figure()
    plt.title(str(np.round(t * dt, 2)) + ' year, ap diff')
    imshowhs_grid(grid, 'topographic__elevation', plot_type='Drape1', drape1=np.sqrt(soil), cmap='jet', alpha=0.5,
                  grid_units=("m", "m"), var_name="Soil depth (m)", cbar_label_color='white', limits=(0, 0.5))
    plt.show()


    nlin_flux_at_out = grid.at_node['sediment_flux_out'][outlet]  # 'sediment_flux_out' est un champ de TLD
    fluv_flux_at_out = grid.at_node['sediment__influx'][outlet]  # 'sediment_influx' est un champ de Space
    outlet_flux_diff_var.append(nlin_flux_at_out)
    outlet_flux_fluv_var.append(fluv_flux_at_out)

    if t * dt > thresholdPlot or t == 2:
        print('mean soil depht', np.mean(soil))
        plt.figure()
        #         imshowhs_grid(grid,z, plot_type='DEM',var_name='DEM, m/m',alpha=0.5,cmap='jet')
        #         plt.show()
        plt.title(str(np.round(t * dt, 2)) + ' year')
        imshowhs_grid(grid, z, plot_type='Drape1', drape1=np.sqrt(soil), var_name='Sediment', alpha=0.5, cmap='jet', limits = (0, 0.5))
        thresholdPlot += plotDiff
        plt.show()



print('somme export diffusion', np.sum(outlet_flux_diff_var))
print('somme export fluviaux', np.sum(outlet_flux_fluv_var))

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.plot(outlet_flux_diff_var, label='nonlinear creep')
ax0.set_title('nonlinear_creep')
ax1.plot( outlet_flux_fluv_var,  label='fluvial export', c='orange')
ax1.set_title('fluvial export')
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('sediment export (m)')
plt.show()