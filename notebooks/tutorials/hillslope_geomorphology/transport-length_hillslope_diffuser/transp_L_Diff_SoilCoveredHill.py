#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: beca4397
"""

import numpy as np
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from landlab.components import TransportLengthHillslopeDiffuser, ExponentialWeatherer, TransportLengthHillslopeDiffuser_v2,\
    Tr_L_diff, DepthDependentTaylorDiffuser, PerronNLDiffuse, FlowDirectorSteepest
from landlab.plot.imshow import imshow_grid
import matplotlib.pyplot as plt
from landlab.io.netcdf import write_raster_netcdf, read_netcdf
import time 

#%%
#plt.figure()
filename = 'Steady_State_nlinDiff'
grid = read_netcdf(filename)


grid.set_closed_boundaries_at_grid_edges(False, True
                                    , False, True)

nx = grid.number_of_node_columns
dx = grid.dx


topo = grid.at_node['topographic__elevation']
topo[:] /=2 
plt.figure()
imshow_grid(grid,'topographic__elevation', plot_name='topographic__elevation')
z_center_ini = np.array(topo[nx:nx*2])


## add Soil.
soil = grid.add_zeros('soil__depth', at='node')
soil[grid.core_nodes] = 1
plt.figure()
imshow_grid(grid,'soil__depth', plot_name = 'soil__depth')
plt.show()

bed = grid.add_ones('bedrock__elevation', at='node')
bed[:]= topo
plt.figure()
imshow_grid(grid,'bedrock__elevation', plot_name = 'bedrock__elevation')
# %%
# Instantiate Flow director (steepest slope type) 
fdir = PriorityFloodFlowRouter(grid, surface="topographic__elevation", flow_metric='D8')

erodibility=.1
slope_crit = 1.42
soil_Z = 0.1
P0 = 0.1
U = 0 #1*1e-3;


expweath = ExponentialWeatherer(grid, soil_production__maximum_rate=P0, 
                                soil_production__decay_depth=soil_Z)

TRLim = True
if TRLim: 
    # diff = TransportLengthHillslopeDiffuser_v2(
    #     grid,
    #     erodibility=erodibility,
    #     slope_crit=slope_crit,
    #     depthDependent = True,    
    #     depositOnBoundaries = True)
    diff = Tr_L_diff(
        grid,
        erodibility=erodibility,
        slope_crit=slope_crit,
        depthDependent = True,
        depositOnBoundaries = True,
        H_star = soil_Z)
else:
    diff = DepthDependentTaylorDiffuser(grid, 
                                    slope_crit=slope_crit,
                                    soil_transport_velocity=erodibility*dx,
                                    courant_factor=0.4,
                                    nterms = 2,
                                    soil_transport_decay_depth = soil_Z)
    
    
    
                                   


# Run the components for ten short timepsteps

dt = 10 # imshow_grid(grid,'topographic__elevation')

plt.figure(figsize=(5,5), dpi=400)
x_row = grid.x_of_node[range(nx)]

totT=10000
plotDiff =totT/8
thresholdPlot =plotDiff

store_z=[]
store_t=[]
t1 = time.time()


plt.figure()
for t in range(int(totT/dt)):
    
    bed[grid.core_nodes]+=U*dt
    fdir.run_one_step()
    topo[:] = bed + soil
    
    expweath.calc_soil_prod_rate()
    if TRLim:
        soil[:]+= grid.at_node['soil_production__rate']*dt
        bed[:]-= grid.at_node['soil_production__rate']*dt
        
    
    diff.run_one_step(dt)    
    
    if t*dt>thresholdPlot:
        # z_center = topo[nx:nx*2]           
        # plt.plot(x_row,z_center)        
        s_center = soil[nx:nx*2]           
        plt.plot(x_row, s_center, label=thresholdPlot)
        thresholdPlot += plotDiff

    #plt.legend()
    plt.title('x row versus soil')

# %% Plotting

plt.figure() 
z_center = topo[nx:nx*2]           
s_center = soil[nx:nx*2]     
plt.plot(x_row, z_center_ini, label = 'Initial topo', linewidth = 3)
plt.plot(x_row, z_center, label = 'Final topo')
plt.legend()
plt.xlabel('Distance, m') 
plt.ylabel('Elevation, m')
plt.title('topo init versus final')
plt.show()

plt.figure()           
plt.plot(x_row,s_center)
plt.xlabel('Distance, m') 
plt.ylabel('Soil thickness, m')
plt.title('Soil thickness')
plt.show()

# plt.figure()        
# imshow_grid(grid,'topographic__elevation', plot_name = 'topographic__elevation')
# plt.figure()   
# imshow_grid(grid,'soil__depth', plot_name = 'soil__depth')
# plt.figure()   
# imshow_grid(grid,'bedrock__elevation', plot_name = 'bedrock__elevation')



