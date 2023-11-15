# Import Linear diffuser:
import numpy as np
from matplotlib.pyplot import figure, plot, show, title, xlabel, ylabel
from landlab.components import LinearDiffuser, FlowDirectorSteepest, Tr_L_diff, PriorityFloodFlowRouter
from landlab import RasterModelGrid, imshow_grid

# Create grid and topographic elevation field:
mg2 = RasterModelGrid((100, 100), xy_spacing=10.0)
mg2.set_closed_boundaries_at_grid_edges(
    False, True, False, True)

z = np.zeros(mg2.number_of_nodes)
z[mg2.node_x > 500] = mg2.node_x[mg2.node_x < 490] / 10
mg2.add_field("topographic__elevation", z, at="node")

# Set boundary conditions:
mg2.set_closed_boundaries_at_grid_edges(False, True, False, True)

# Show initial topography:
imshow_grid(mg2, "topographic__elevation", grid_units=["m", "m"], var_name='Elevation (m)')

# Plot an east-west cross-section of the initial topography:
z_plot = z[100:199]
x_plot = range(0, 1000, 10)
figure(2)
plot( z_plot)
title("East-West cross section")
xlabel("x (m)")
ylabel("z (m)")

total_t = 1000000.0  # total run time (yr)
dt = 1000.0  # time step (yr)
nt = int(total_t // dt)  # number of time steps

fdir = PriorityFloodFlowRouter(mg2)

tl_diff = Tr_L_diff(mg2, erodibility=0.001, slope_crit=0.6)


z = mg2.at_node['topographic__elevation']
for t in range(nt):
    fdir.run_one_step()
    tl_diff.run_one_step(dt)

    # add some output to let us see we aren't hanging:
    if t % 100 == 0:
        print(t * dt)
        z_plot = z[100:199]
        figure(2)
        plot(z_plot)