#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Feb 25, 2022.

@author: BenjaminCampforts
"""


import numpy as np

from landlab import Component

from .cfuncs import non_local_Depo


class Tr_L_diff(Component):

    r"""Transport length hillslope diffusion.

    #TODO
    Correct for rho soil vs bedrock

    Component written by Benjamin Campforts, 2022

    Parameters
    ----------


    Examples
    --------



    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    None Listed

    **Additional References**

    Carretier, S., Martinod, P., Reich, M., Godderis, Y. (2016). Modelling
    sediment clasts transport during landscape evolution. Earth Surface Dynamics
    4(1), 237-251. https://dx.doi.org/10.5194/esurf-4-237-2016

    Davy, P., Lague, D. (2009). Fluvial erosion/transport equation of landscape
    evolution models revisited. Journal of Geophysical Research  114(F3),
    F03007. https://dx.doi.org/10.1029/2008jf001146

    """

    _name = "TransportLengthHillslopeDiffuser"

    _unit_agnostic = True

    _info = {
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
    }

    def __init__(
        self,
        grid,
        erodibility=0.001,
        slope_crit=1.0,
        depositOnBoundaries=False,
        depthDependent=False,
        H_star=1.0,
    ):

        """Initialize Diffuser.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        erodibility: float
            Erodibility coefficient [L/T]
        slope_crit: float (default=1.)
            Critical slope [L/L]
        depositOnBoundaries: boolean (default=False)

        depthDependent: boolean (default=False)

        H_star=1.0 : float (default=1.)

        """
        self._depthDependent = depthDependent
        if self._depthDependent:
            self._info["soil__depth"]["optional"] = False
            self._info["bedrock__elevation"]["optional"] = False
            # Depth scale
            self._H_star = H_star

        super().__init__(grid)

        if grid.at_node["flow__receiver_node"].size != grid.size("node"):
            msg = (
                "A route-to-multiple flow director has been "
                "run on this grid. The landlab development team has not "
                "verified that TransportLengthHillslopeDiffuser is compatible "
                "with route-to-multiple methods. Please open a GitHub Issue "
                "to start this process."
            )
            raise NotImplementedError(msg)

        # Store grid and parameters

        self._k = erodibility
        self._slope_crit = slope_crit

        # Create fields:
        # Elevation
        self._el = self._grid.at_node["topographic__elevation"]
        # Soil
        if self._depthDependent:
            self._soil = self._grid.at_node["soil__depth"]
            self._bed = self._grid.at_node["bedrock__elevation"]

        self._steepest = self._grid.at_node["topographic__steepest_slope"]
        self._r = self._grid.at_node["flow__receiver_node"]
        self._lk_rcvr = self.grid.at_node["flow__link_to_receiver_node"]
        self._stack = self.grid.at_node["flow__upstream_node_order"]
        self._link_lengths = self.grid.length_of_d8

        self.initialize_output_fields()
        self._depositOnBoundaries = depositOnBoundaries

    def tldiffusion(self, dt):
        """Calculate hillslope diffusion for a time period 'dt'.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        dt: float (time)
            The imposed timestep.
        """

        dx = self._grid.dx
        lakes = self._steepest < 0

        if self._depthDependent:
            ero = self._k * self._steepest * (1 - np.exp(-self._soil / self._H_star))
            ero = np.minimum(ero * dt, self._soil) / dt
        else:
            # Calcualte erosion -- in comparison to v1, not curring off at Sc
            ero = self._k * self._steepest

        ero[lakes] = 0

        L = np.where(
            self._steepest < self._slope_crit,
            dx / (1 - (self._steepest / self._slope_crit) ** 2),
            1e9,
        )

        qs_out = np.zeros_like(self._el)
        depo = np.zeros_like(self._el)

        # for node in np.flipud(self._stack):
        #     depo[node] = qs_out[node]/L[node]
        #     qs_out[self._r[node]] = qs_out[node]+ (ero[node] -depo[node])*dx

        non_local_Depo(dx, np.flipud(self._stack), self._r, qs_out, L, ero, depo)

        # Calculate deposition rate on node
        if not self._depositOnBoundaries:
            depo[self._grid.boundary_nodes] = 0

        # Update elevation
        if self._depthDependent:
            self._soil += (-ero + depo) * dt
            self._el = self._soil + self._bed
        else:
            self._el += (-ero + depo) * dt

    def run_one_step(self, dt):
        """Advance one timestep.

        Advance transport length-model hillslope diffusion component
        by one time step of size dt and tests for timestep stability.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        elev_dif_before = self._el - self._el[self._r]
        flow__sink_flag = elev_dif_before < 0
        self.tldiffusion(dt)

        # Test code stability for timestep dt
        # Raise unstability error if local slope is reversed by erosion
        # and deposition during a timestep dt
        elev_dif = self._el - self._el[self._r]
        s = elev_dif[np.where(flow__sink_flag == 0)]
        if np.any(s < -1) is True:
            raise ValueError(
                "The component is unstable" " for such a large timestep " "on this grid"
            )
        else:
            pass
