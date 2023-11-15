"""créer par Coline Ariagno le 08/09/2021

Modéle du Laval avec le TLD et avec les fonction pour suivre et extraire les variables au cours de l'année
( ex: Soil depth, sed export, etc...)
E1 = Première étape du modéle, model simplifié avec TLD et erosion fluviale l'été et épaisseur illimité et pas de hiver"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress
import time
from landlab import RasterModelGrid, imshow_grid, imshow_grid_at_node
from landlab.components import (FlowAccumulator,
                                DepthDependentTaylorDiffuser,
                                ExponentialWeathererIntegrated, SpaceLargeScaleEroder, PriorityFloodFlowRouter,
                                Tr_L_diff)

import os.path as op
from landlab.io import read_esri_ascii
from relation_P0_FCI_vt import determine_P0


def main(year, air_hillslope, export_file):

    # script options
    activate_successive_year_with_reset = True
    activate_run_previous_year = True
    count = 0

    #condition to initiate with running n time previous yr or just open DEM
    if activate_run_previous_year:
        grid_laval, out_id = previous_init_year(count, air_hillslope)
    else:
        grid_laval, out_id = open_DEM_laval()


    # Loop needed for several annual analysis in a row.( without reset the field)
    for yr in year:
        # open and read rainfall event characteristics (time / cumul):
        df_rain = rainfall_event(yr)
        print(f'len df_rain is for the year', len(df_rain), yr)

        # read weathering model for the different year:
        if yr == year[0]:
            grid_laval, sed_flux, df_compare_brut = weathering_model(yr, df_rain, grid_laval,
                                                    out_id, count, air_hillslope)

        elif yr != year[0]:
            if activate_successive_year_with_reset:
                if activate_run_previous_year:
                    grid_laval, out_id = open_DEM_laval()
                else:
                    grid_laval, out_id = open_DEM_laval()

            grid_laval, sed_flux, df_compare_brut = weathering_model(yr, df_rain, grid_laval, out_id, count, air_hillslope)



def open_DEM_laval():
    # Open and read DEM initial:
    (grid_laval, z_laval) = read_esri_ascii("extract_mntla2_filled_modifCA.asc", name="topographic__elevation")

    # set boundary conditions:
    out_id_arr = grid_laval.set_watershed_boundary_condition(z_laval, nodata_value=9999.0, return_outlet_id=True)
    out_id = out_id_arr[0]  # out_id[0]      # sinon out_id = ndarray.    out_id = 175050
    print('out id and status', out_id, grid_laval.status_at_node[out_id])

    return grid_laval, out_id



def previous_init_year(count, air_hillslope):
    grid_laval, out_id = open_DEM_laval()
    yr = 2012
    nb_previous_yr = 10     # nunmber of run to initialize the grid
    # open and read rainfall event characteristics (time / cumul):
    df_rain = rainfall_event(yr)

    # read weathering model for the different year:
    for nb in list(range(nb_previous_yr)):
        count += 1
        grid_laval, sed_flux, df_compare_brut = weathering_model(yr, df_rain, grid_laval,
                                                                 out_id, count,
                                                                 air_hillslope)
    return grid_laval, out_id


def rainfall_event(year):
    filepath = 'C:\\Users\\coline.ariagno\\Documents\\Landlab_22\\draix_model\\model_weathering\\all_rainfall_event.xlsx'
    assert op.exists(filepath)
    dic_rain = pd.read_excel(filepath, sheet_name=f'event_{year}')
    df_rainfall_event = pd.DataFrame.from_dict(dic_rain)
    df_rainfall_event.set_index('Date', inplace=True)
    print(df_rainfall_event.head)

    return df_rainfall_event



def weathering_model(year, df_rain,
                     # Laval parameters
                     grid_laval,
                     out_id,
                     count,
                     air_hillslope,
                     annual_sediment_export=12000,
                     surface_Laval=86e4,
                     annual_runoff_coef=0.38,

                     # time parameters
                     time_step='year',  # 2 options : 'day' or 'year'
                     nb_day_winter_efficient=105,              # roughly nb days between Nov 15 and February 28 => 105
                     nb_day_winter=164,  # nb days between Oct 18 and Marsh 31
                     nb_day_summer=200,

                     # Set parameters for landlab component:
                     # SPACE component
                     K_sed=0.01,  # [m]
                     K_br=1.0e-7,
                     frac_fines=0.0,
                     porosity=0,
                     alluv_depth_scale=0.10,
                     depo_parameter=1e-4,
                     m_sp=0.5,
                     n_sp=1.0,
                     threshold_for_sediment_erosion=0.0,
                     threshold_for_rock_erosion=0.0,

                     # Exponential weatherer component
                     soil_prod_max_rate=0.1,
                     soil_prod_depth_scale=0.01, # h_star = 0.05   # [m]
                     
                     # Transport length diffusion:
                     dx=2,
                     erodibility=0.1,

                     # Depth dependent diffuser component
                     soil_transport_depth_scale=0.01,  # cf Carrière, 2020   [m]
                     velocity_transport_coef=5.3,  # [m/yr]   cf Carriere, 2020 D = K * H* -> K= D / H*
                     S_crit=1.42, # [m/m]   correspond �  55°=0.96rad -> tan(0.96)= h/l [m/m]       Alexandra a pris 1.48

                     # initial statement for winter and summer loop
                     h_nov=0.1,  # [m]               # soil thickness at the beginning of the winter (November)
                     coef_dir_P0=-1e-3,
                     P0moy=0.1,
                     ):
    Trspt_length_diff = True
    activate_frac_winter_loop = True
    plot_SD_winter_vs_location = False          # plot pour regarder effet des cycles W-TLD
    activate_weatherer = False

    # INITIALIZE stable parameters:
    # #time parameters
    dt_winter_efficient = nb_day_winter_efficient / 365

    # INITIALIZE records and fields
    records_soil_depth = []
    records_topo = []
    records_bedrock = [] # list pour enregistré �  chaque event la masse de sed exporté ( débit massiqe * temps de l'event)
    record_sed_influx_outlet_mass = []
    record_sed_outflux_TLD_outlet_mass = []

    records_soil_depth_outlet = []
    records_soil_depth_slope = []
    records_soil_depth_steep_slope = []
    records_soil_depth_channel = []
    records_soil_depth_hillslopes = []
    num_event = 0

    # INITIALIZE records for hysteresis field:
    list_num_month = [0]  # stat 0 is for record of the winter
    df_compare = pd.DataFrame()

    # Initialize input field:
    z_laval = grid_laval.at_node['topographic__elevation']
    topo_init = np.mean(z_laval[grid_laval.core_nodes])  # needed to compute final topographic difference

    if 'soil__depth' not in grid_laval.at_node:  # means that it's the first year and soil depth and bedrock field doesn't exist
        print(f'\n ROUND {year}')
        runoff_rate_event = grid_laval.add_zeros('runoff_rate__event', at='node',
                                                 units='m/time')  # initialize field to update the discharge field in SPACE. Ce champ n'existe pas dans les composants -> il faut le redéfinir chaque année.
        soil_depth = grid_laval.add_zeros('soil__depth', at='node', units='m')
        bed_rock = grid_laval.add_zeros('bedrock__elevation', at='node', units='m')
        soil_depth[:] = h_nov
        bed_rock[:] = z_laval[:] - h_nov
    else:  # the grid already store all the field from the year before
        print(f'\n ROUND {count}')
        runoff_rate_event = grid_laval.at_node['runoff_rate__event']
        bed_rock = grid_laval.at_node['bedrock__elevation']
        soil_depth = grid_laval.at_node['soil__depth']
        runoff_rate_event[:] = 0
        print('runoff syr+1', np.mean(runoff_rate_event), runoff_rate_event)

    # Instantiate components

    flow_accumulator = FlowAccumulator(grid_laval,                                # When FA runs at every summer step
                                       surface='topographic__elevation',
                                       flow_director='FlowDirectorD8', runoff_rate=runoff_rate_event, depression_finder='DepressionFinderAndRouter')

    space_model_eroder = SpaceLargeScaleEroder(grid_laval,
                                               K_sed=K_sed,
                                               K_br=K_br,
                                               F_f=frac_fines,
                                               phi=porosity,
                                               H_star=alluv_depth_scale,
                                               v_s=depo_parameter,
                                               m_sp=m_sp,
                                               n_sp=n_sp,
                                               sp_crit_sed=threshold_for_sediment_erosion,
                                               sp_crit_br=threshold_for_rock_erosion,
                                               discharge_field='surface_water__discharge',
                                               )

    weatherer = ExponentialWeathererIntegrated(grid_laval,
                                               soil_production__maximum_rate=soil_prod_max_rate,
                                               soil_production__decay_depth=soil_prod_depth_scale)

    tsp_lg_diff = Tr_L_diff(grid_laval, erodibility=erodibility, slope_crit=S_crit,
                            H_star=soil_transport_depth_scale,
                            depthDependent=True, depositOnBoundaries=False, transportLengthCoefficient=dx)

    # RENAME existing field:
    discharge = grid_laval.at_node['surface_water__discharge']
    da = grid_laval.at_node['drainage_area']
    sed_flux = grid_laval.at_node['sediment__flux']
    sed_influx = grid_laval.at_node['sediment__influx']
    sed_flux_out = grid_laval.at_node['sediment_flux_out']          # export field forTLD
    soil_prod_dt = grid_laval.at_node['soil_production__dt_produced_depth']
    bed_prod_dt = grid_laval.at_node['soil_production__dt_weathered_depth']
    if np.mean(discharge) != 0:  # means that we are in a second year
        discharge[:] = 0
        sed_influx[:] = 0
        bed_prod_dt[:] = 0
        soil_prod_dt[:] = 0

    # store initial elevation
    records_topo.append(np.mean(z_laval[grid_laval.core_nodes]))
    record_sed_influx_outlet_mass.append(0.0)

    records_soil_depth.append(h_nov)
    records_soil_depth_slope.append(h_nov)
    records_soil_depth_steep_slope.append(h_nov)
    records_soil_depth_channel.append(0.0)
    records_soil_depth_hillslopes.append(h_nov)

    print('INITIALE FIELD VALUE')
    print('mean topo', np.mean(z_laval[grid_laval.core_nodes]))
    print('mean bedrock update init', np.mean(bed_rock[grid_laval.core_nodes]))


    # RUN

    flow_accumulator.run_one_step()
    steepest_slope = grid_laval.at_node['topographic__steepest_slope']

    # da is updated by FA -> remove soil in the main channel ( to be more sensitive of sed from hillslopes)
    soil_depth[da > 10000] = 0

    # check if mass conservation
    MC_before = np.sum(soil_depth[grid_laval.core_nodes])
    print('mass conservation before', np.sum(soil_depth[grid_laval.core_nodes]), np.sum(soil_depth)/da[out_id])


    # WINTER LOOP:
    if activate_frac_winter_loop:
        dt_frac = dt_winter_efficient / (int(nb_day_winter_efficient / 2))      # dt_winter = 105/365 = 0.28 ,  nb day = 105
        dt_frac_init = dt_frac
        nb_frac = 0
        while dt_frac <= dt_winter_efficient:
            nb_frac += 1
            soil_prod_max_rate = 0.1
    
            # instantiation avec new val de soil_prod_max_rate
            if activate_weatherer:
                weatherer = ExponentialWeathererIntegrated(grid_laval,
                                                           soil_production__maximum_rate=soil_prod_max_rate,
                                                           soil_production__decay_depth=soil_prod_depth_scale)
                # Hillslope processes: weathering and diffusion
                weatherer.run_one_step(dt_frac_init)  # run_one_step ne retourne rien mais les grid field sont mis �  jour.

                # update of h = h_marsh:
                soil_depth[:] = np.add(soil_depth, soil_prod_dt)  # Formule Ok -> soil depth bien modifié

                # update of bedrock elevation because weatherer doesn't did it
                bed_rock[:] = np.add(bed_rock, - bed_prod_dt)  # equivalent to np.add(bed_rock, - soil_prod_dt)
    
                print('\n After WEATHERER: after update of SD and bedrock')
                print('mean topo and bed', np.mean(z_laval[grid_laval.core_nodes]), np.mean(bed_rock[grid_laval.core_nodes]))
                print('mean soil depth end weatherer all', np.mean(soil_depth[grid_laval.core_nodes]))
                print('mean soil depth end weatherer on slope', np.mean(soil_depth[da < 10000]))
                print('Hillslope soil depth end winter', np.mean(soil_depth[da <= air_hillslope]))

                # Activate juste qd je regarde winter
                # records_soil_depth.append(np.mean(soil_depth[grid_laval.core_nodes]))
                # records_soil_depth_outlet.append(soil_depth[out_id])
                # records_soil_depth_slope.append(np.mean(soil_depth[(da < 10000) & (steepest_slope < 1.4)]))
                # records_soil_depth_steep_slope.append(np.mean(soil_depth[(da < 10000) & (steepest_slope > 1.4)]))
                # records_soil_depth_channel.append(np.mean(soil_depth[da > 10000]))
                # records_soil_depth_hillslopes.append(np.mean(soil_depth[(da <= air_hillslope) & (steepest_slope < 1.4)]))
    
            # WINTER DIFFUSION :
            if Trspt_length_diff:  # elevation and SD field are updated by the component itself
                tsp_lg_diff.run_one_step(dt_frac_init)
    
                print('\n After WINTER DIFFUSION: after update of SD and bedrock')
                print('Mean soil depth after winter_diffusion', np.mean(soil_depth[grid_laval.core_nodes]))
                print('mean soil depth after diff on slope', np.mean(soil_depth[da < 10000]))
                print('min et max SD', np.min(soil_depth[grid_laval.core_nodes]), np.max(soil_depth[grid_laval.core_nodes]))
                print('sed_flux_out', sed_flux_out[out_id])


            records_soil_depth.append(np.mean(soil_depth[grid_laval.core_nodes]))
            records_soil_depth_outlet.append(soil_depth[out_id])
            records_soil_depth_slope.append(np.mean(soil_depth[(da < 10000) & (steepest_slope < 1.4)]))
            records_soil_depth_steep_slope.append(np.mean(soil_depth[(da < 10000) & (steepest_slope > 1.4)]))
            records_soil_depth_channel.append(np.mean(soil_depth[da > 10000]))
            records_soil_depth_hillslopes.append(np.mean(soil_depth[(da <= air_hillslope) & (steepest_slope < 1.4)]))
            record_sed_outflux_TLD_outlet_mass.append(sed_flux_out[out_id] * 2.6 * dt_frac)

            dt_frac += dt_frac_init
            print('dt_frac', nb_frac, dt_frac)

        # mass conversation balance
        MC_after = np.sum(soil_depth[grid_laval.core_nodes])
        print('sed flux out TLD', record_sed_outflux_TLD_outlet_mass)
        print('The sum of soil depth in the catchment is {} before and {} after'.format(MC_before, MC_after))
        print('mass conservation bilan', MC_before - MC_after - np.sum(record_sed_outflux_TLD_outlet_mass))

        if plot_SD_winter_vs_location:
            fig, ax = plt.subplots()
            position_txt_x = 9
            print('dt', len(list(range(int(dt_winter_efficient / dt_frac_init)))) + 1, len(records_soil_depth))
            ax.plot(list(range((int(dt_winter_efficient / dt_frac_init)) +2 )), records_soil_depth, c='red')

            ax.plot(list(range((int(dt_winter_efficient / dt_frac_init)) +2 )), records_soil_depth_slope, c='cyan')
            ax.annotate(str(np.round(records_soil_depth_slope[-1], 4)), xy=(position_txt_x, records_soil_depth_slope[-1]), fontsize=20)

            ax.plot(list(range((int(dt_winter_efficient / dt_frac_init)) +2 )), records_soil_depth_steep_slope, c='dodgerblue')
            ax.annotate(str(np.round(records_soil_depth_steep_slope[-1], 4)), xy=(position_txt_x, records_soil_depth_steep_slope[-1]), fontsize=20)

            ax.plot(list(range((int(dt_winter_efficient / dt_frac_init)) +2 )), records_soil_depth_hillslopes, c='blue')

            ax.plot(list(range((int(dt_winter_efficient / dt_frac_init)) +2 )), records_soil_depth_channel, c='k')
            ax.annotate(str(np.round(records_soil_depth_channel[-1], 4)), xy=(position_txt_x, records_soil_depth_channel[-1]), fontsize=20)

            ax.set_xlabel('nb cycle W-TLD', fontsize=20)
            ax.set_ylabel('Soil depth (mm)', fontsize=20)
            ax.tick_params(direction='out', length=5, width=1, labelsize=20)
            ax.legend(['total catchment', 'da < 10000 & slope < 1.4', 'da < 10000 & slope > 1.4', 'da < 4', 'da > 10000'], fontsize=18, loc='best')
            plt.show()

            raise NotImplementedError



# SUMMER LOOP:

    # For each rainfall event, erosion and transport fluvial happenned
    for i, event in enumerate(list(df_rain.index)):
        num_event += 1
        print('\n i et event', i, num_event, event)

        # characterictic of the event:
        duration_event = (df_rain['time_event'][event] / (24 * 60 * 365))  # [yr], le 1440 pour passer des min en jour, le /365 c'est pour passer des [j] en [année]
        cumul_event = df_rain['cumul_event'][event] * 1e-3  # [m]
        intensity_event = (df_rain['max_intensity_event'][event])
        print('event duration and cumul', duration_event, cumul_event)

        rainfall_rate_event = cumul_event / duration_event  # [m/j] ou [m/yr] si /365 au dessus
        runoff_rate_event[:] = (annual_runoff_coef * rainfall_rate_event)  # / (24*3600*365)         # [m/yr] car le CR est annuel?  update in Flowaccumulator
        print('the mean runoff_rate is {} and the runoff grid is  {}'.format(np.mean(runoff_rate_event),
                                                                             runoff_rate_event))


        # FLOWACCUMULATOR: update directions and drainage area: modifie le champ discharge car il est utilisé par SPACE -> unit [m/yr] *
        print('\n FlowACCUMULATOR')
        flow_accumulator.run_one_step()
        discharge[:] = da[:] * runoff_rate_event[:]
        print('mean discharge after', np.mean(discharge[grid_laval.core_nodes]))

        # DIFFUSION:
        # TRANSPORT LENGTH DIFFUSION:
        if Trspt_length_diff:
            print('mean topo and bed avt TLD', np.mean(z_laval[grid_laval.core_nodes]), np.mean(bed_rock[grid_laval.core_nodes]))
            print('Mean soil depth avt TLD', np.mean(soil_depth[grid_laval.core_nodes]))
            print('min et max soil depth avt TLD', np.min(soil_depth[grid_laval.core_nodes]), np.max(soil_depth[grid_laval.core_nodes]))
            tsp_lg_diff.run_one_step(duration_event)

            print('\n After TRANSPORT LENGTH DIFFUSION SUMMER')
            print('mean topo and bed ap TLD', np.mean(z_laval[grid_laval.core_nodes]), np.mean(bed_rock[grid_laval.core_nodes]))
            print('Mean soil depth ap TLD', np.mean(soil_depth[grid_laval.core_nodes]))
            print('min et max soil depth ap TLD', np.min(soil_depth[grid_laval.core_nodes]), np.max(soil_depth[grid_laval.core_nodes]))

        # SPACE: Surface water erosion/deposition
        space_model_eroder.run_one_step(duration_event)

        print('\n After SPACE')
        print('mean topo', np.mean(z_laval[grid_laval.core_nodes]))
        print('mean bed', np.mean(bed_rock[grid_laval.core_nodes]))
        print('mean SD', np.mean(soil_depth[grid_laval.core_nodes]))
        print('min et max SD', np.min(soil_depth[grid_laval.core_nodes]),
              np.max(soil_depth[grid_laval.core_nodes]))
        print('mean SD on hillslope', np.mean(soil_depth[da <= air_hillslope]))

        # records value of the event from the field
        records_topo.append(np.mean(z_laval[grid_laval.core_nodes]))
        records_bedrock.append(np.mean(bed_rock[grid_laval.core_nodes]))
        records_soil_depth.append(np.mean(soil_depth[grid_laval.core_nodes]))

        # Records variable at the outlet node
        record_sed_influx_outlet_mass.append(sed_influx[out_id] * 2.6 * duration_event)
        records_soil_depth_outlet.append(soil_depth[out_id])  # out_id = array =[....]
        records_soil_depth_hillslopes.append(np.mean(soil_depth[(da <= air_hillslope) & (steepest_slope < 1.4)]))

        print('\n VARIABLE AT OUTLET NODE')
        print('outlet sed influx MASS: val [tons]', sed_influx[out_id] * 2.6 * duration_event)  # débit volumique [m^3/an] * densité [t/m^3] * temps event[an]
        print('record SD outlet', records_soil_depth_outlet[-1], records_soil_depth_outlet)
        print('TOTAL INTERMEDIATE Sed export expected',
              (records_soil_depth[0] - records_soil_depth[-1]) * surface_Laval * 2.6)

        # Records for hysteresis intra annual analyse
        month = int(event.split('/', 5)[3])  # get the month of the event (for hysteresis records)
        list_num_month.append(month)  # to compare de month of the actual date to the previous month's date.

        print('len', len(list_num_month), len(records_soil_depth), len(record_sed_influx_outlet_mass))

    # df_compare = pd.DataFrame({'list_num_month': list_num_month, 'SD_simul': records_soil_depth,
    #                            'Sed_influx_simul': record_sed_influx_outlet_mass,
    #                            'SD_hillslope_simul': records_soil_depth_hillslopes})

    time = list(range(0, num_event + 1))

    cumul_sed_influx_outlet_mass = list(np.cumsum(record_sed_influx_outlet_mass))

    print('\n END OF YEAR')
    print('diff_topo', topo_init - records_topo[-1])
    print('mean soil depth difference and SD at the end', records_soil_depth[-1] - records_soil_depth[0], records_soil_depth[-1])
    print('min et max SD', np.min(soil_depth[grid_laval.core_nodes]), np.max(soil_depth[grid_laval.core_nodes]))
    print('sum mass sed Influx(tons)', cumul_sed_influx_outlet_mass[-1], cumul_sed_influx_outlet_mass)
    print('Total Sed export expected from soil depth difference [tons]', (records_soil_depth[0] - records_soil_depth[-1]) * surface_Laval * 2.6)  # resultat en tonnes, Caro met [1] car sa première valeur est la valeur initiale avt le weatherer)

    return grid_laval, cumul_sed_influx_outlet_mass[-1], df_compare


if __name__ == '__main__':
    main(year=[2020], air_hillslope=4, export_file='Mass_conservative.xlsx')