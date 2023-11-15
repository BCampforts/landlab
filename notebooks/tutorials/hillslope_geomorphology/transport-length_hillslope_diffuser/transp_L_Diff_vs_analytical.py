#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: beca4397
"""

import numpy as np
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from landlab.components import TransportLengthHillslopeDiffuser, TransportLengthHillslopeDiffuser_v2, Tr_L_diff, TaylorNonLinearDiffuser, PerronNLDiffuse, FlowDirectorSteepest
from landlab.plot.imshow import imshow_grid
import matplotlib.pyplot as plt
from landlab.io.netcdf import write_raster_netcdf, read_netcdf
import time 

#%%
nx=61
ny=3
xy_spacing=50
mg = RasterModelGrid((ny, nx),xy_spacing=xy_spacing)
mg.set_closed_boundaries_at_grid_edges(False, True
                                    , False, True)
z = np.zeros(nx*ny)
z[mg.boundary_nodes]=0
z = mg.add_field("topographic__elevation", z, at="node")

s = np.zeros(nx*ny)
s = mg.add_field("soil__depth", s, at="node")
b = np.zeros(nx*ny)
b = mg.add_field("bedrock__elevation", b, at="node")
# z = mg.at_node['topographic__elevation']
slope_crit=0.8
imshow_grid(mg,'topographic__elevation')
plt.show()


# Instantiate Flow director (steepest slope type) and TL hillslope diffuser

fdir = PriorityFloodFlowRouter(mg, surface="topographic__elevation", flow_metric='D8')


erodibility=.01
taylorFlux = False
depthDependent = True


# diff = TransportLengthHillslopeDiffuser(
#     mg,
#     erodibility=erodibility,
#     slope_crit=slope_crit)



diff = TransportLengthHillslopeDiffuser_v2(
    mg,
    erodibility=erodibility,
    slope_crit=slope_crit,
    depthDependent = depthDependent)


# diff = Tr_L_diff(
#     mg,
#     erodibility=erodibility,
#     slope_crit=slope_crit,
#     depthDependent = depthDependent)


# diff = TaylorNonLinearDiffuser(mg, 
#                                 slope_crit=slope_crit,
#                                 linear_diffusivity=erodibility*xy_spacing,
#                                 courant_factor=0.45,
#                                 dynamic_dt= True,
#                                 nterms = 2); taylorFlux = True

# diff = PerronNLDiffuse(mg, nonlinear_diffusivity=0.01,S_crit = slope_crit)

# Run the components for ten short timepsteps
U = .3*1e-3;
plt.figure()
dt = 500 # imshow_grid(mg,'topographic__elevation')

plt.figure(figsize=(5,5), dpi=400)
x_row = mg.x_of_node[range(nx)]
print('x_row', x_row.shape, x_row)

totT=int(5e6)
plotDiff =totT/5
thresholdPlot =plotDiff

store_z=[]
store_t=[]
t1 = time.time()
for t in range(int(totT/dt)):
    if depthDependent:
        s[mg.core_nodes]+=U*dt
        z[:] = b + s 
    else:
        z[mg.core_nodes]+=U*dt
    if not taylorFlux: 
        fdir.run_one_step()    
    diff.run_one_step(dt)    
    
    
    if t*dt>thresholdPlot:
        z_center = z[nx:nx*2]
        print('zcenter', z_center)
        plt.plot(x_row, z_center)
        plt.scatter(x_row, z_center, facecolors='none', edgecolors='k')
        
        store_z.append(z_center)
        store_t.append(t*dt)
        
        thresholdPlot += plotDiff

raise NotImplementedError

# plt.figure()        
# imshow_grid(mg,'topographic__elevation')
#%% timing
t2=time.time();
#%% Dimentionless E
runT =t2-t1
E = U
K = erodibility*xy_spacing
x = x_row[0:int(np.ceil(nx/2))]

rho_r = 1.35
rho_s = 1.35

L_H = x[-1]
Er = K*slope_crit/(2*L_H*(rho_r/rho_s))
Es = E/Er
xs = x/L_H

zs = (1/Es)*(
    np.log(0.5*(1+np.sqrt(1+(Es*xs)**2)))
    +1
    - np.sqrt(1+(Es*xs)**2)
    )
zs += -np.min(zs)
plt.figure()
plt.plot(xs,zs)
plt.show()

z_dim = zs*slope_crit*L_H
z_comb = np.concatenate((np.flip(z_dim), z_dim[1:]))


plt.figure()
plt.plot(x_row, z_comb, label='analytical')
# plt.plot(x_row,z_center) 
z_center = z[nx:nx*2]

plt.scatter(x_row, z_center, facecolors='none', edgecolors='k', label='numerical')
plt.legend()

# %%
filename = 'Steady_State_nlinDiff'
write_raster_netcdf(filename, mg,  names='topographic__elevation', format="NETCDF4")
plt.show()


