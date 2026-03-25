import numpy as np
import datetime
from sgp4.api import Satrec, WGS72
from skyfield.api import load, wgs84
import skyfield.api as sf
from GEO import LEO_GEO
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import copy



if __name__ == '__main__':




 def Satellites_in_orbits_constellations(LEOs, groundstation, elevation_angle=10, ts = None, safetyMargin = 0):

    ts = sf.load.timescale()
    time1 = ts.utc(2021,9,22,00,00,00)      # Start point, UTC time
    time2 = ts.utc(2021,9,26,00,00,00)      # End point, UTC time

    seconds_difference_time = LG.difference_time_in_seconds(time1, time2) # How many seconds between start and end
    DF = LG.simulateConstellation(LEOs, groundstation, 10, time1, time2, ts = None, safetyMargin = 0)    
    DF = LG.simulateConstellation(LEOs, groundstation, 10, time1, time2, ts = None, safetyMargin = 0)

    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []

    Sats_name = []
    for i in range(1,numSat+1):
        Sats_name.append('Sat '+str(i))

    for i in range(len(LEOs)):
       rise_df = []
       rise_df = DF.loc[Sats_name[i]]  # Classify each satellite information

       rise_df1 = rise_df['Rise']  # Only put the 'Rise' information of satellites
       end_time_df1 = rise_df['end_time']  # Only put the 'end_time' information of satellites

       arr1_start = []
       arr1_end = []
       distancee = []

       #print(f"rise_df1 = {rise_df1}")
       #print(f"end_df1 = {end_time_df1}")
       for ii in range(len(rise_df1)):

            t_r = rise_df1[ii]  # The moment of rising
            t_e = end_time_df1[ii]


            time_rise_now = ts.utc(t_r.year, t_r.month, t_r.day, t_r.hour, t_r.minute, t_r.second)

            position_LEO = LEOs[i].at(time_rise_now)
            position_GS = groundstation.at(time_rise_now)
            position_LEO = LEOs[i].at(time_rise_now)
            distancee.append(LG.distance(position_GS.position.km, position_LEO.position.km))
            time_end_now = ts.utc(t_e.year, t_e.month, t_e.day, t_e.hour, t_e.minute, t_e.second)
            #time_offline_now = ts.utc(t_off.year, t_off.month, t_off.day, t_off.hour, t_off.minute, t_off.second)
            arr1_start.append(LG.difference_time_in_seconds(time1, time_rise_now)) # One array with the size of (Num_sat*) with the rise time of satellites
            arr1_end.append(LG.difference_time_in_seconds(time1, time_end_now)) # One array with the size of (Num_sat*) with the rise time of satellites


       visiting_start_GS_LEO.append(arr1_start)
       visiting_end_GS_LEO.append(arr1_end)

    return visiting_start_GS_LEO, visiting_end_GS_LEO, seconds_difference_time

 


    
    ####################################*********** GS as the server **************##########################################
 # Website  for checking latitude and longitude: https://latitude.to/articles-by-country/fr/france/1064/charles-de-gaulle-airport
 groundstation = wgs84.latlon(+53.00, 8.80) #GS-BR
 GS_name = 'Bremen'
 #groundstation = wgs84.latlon(+49.00, 2) #Paris
 #GS_name = 'Paris' 
 #groundstation = wgs84.latlon(-23.55, -46.63) #Brazil 
 #GS_name = 'Sao Paulo' 
 #groundstation = wgs84.latlon(+90.0, 0.0) #GS-NP





h_LEO = 2000e3
inclination = 60
inc_GEO = 0
numSat = 80
numPlanes = 5
phasing = 1
r_E = 6371e3
gm = 3.986004418e14
h_GEO = 500e3
num_sat_each_orbit = int(numSat/numPlanes)
## Determne whether the correct number of satellites in each orbit for intra-plane ISLs
Threshold_dist = 2*np.sqrt((r_E + h_LEO)**2 - (r_E)**2 )
print(Threshold_dist)
print(np.sin(np.deg2rad(360/(num_sat_each_orbit*2))))
Real_dist = np.sin(np.deg2rad(360/(num_sat_each_orbit*2))) * (r_E + h_LEO) * 2
print(Real_dist)
print(np.sin(np.deg2rad(0)))

if Real_dist <= Threshold_dist:


    LG = LEO_GEO(r_E, gm, h_GEO)
    GEO = LG.GEO(inc_GEO)
    LEOs = LG.walkerConstellation(h_LEO, inclination, numSat, numPlanes, phasing, name = "Sat")

    arr_diff_Second_rise, arr_diff_Second_set, seconds_difference_time =  Satellites_in_orbits_constellations(LEOs, groundstation, elevation_angle=10,ts = None, safetyMargin = 0)


    satellites_index = []
    for k in range(len(arr_diff_Second_rise)):
        
        satellites_index.append([k for idx in range(len(arr_diff_Second_rise[k]))])

    All_satellite_visits = []
    All_satellite_index = []
    for k in range(len(arr_diff_Second_rise)):
        All_satellite_visits.extend(arr_diff_Second_rise[k])
        All_satellite_index.extend(satellites_index[k])

    #print(f"All_satellite_visits = {All_satellite_visits}")
    print(f"All_satellite_index = {All_satellite_index}")

    indexes_satellite_visits = np.argsort(All_satellite_visits)

    Satellite_indexes = []
    for g1 in range(len(All_satellite_index)):
        Satellite_indexes.append(All_satellite_index[indexes_satellite_visits[g1]])
    print(f"^^ Satellite_indexes = {Satellite_indexes}")


    print(f"***SAT_IND  = {satellites_index}")

    print(f"$$ indexes_satellite_visits = {indexes_satellite_visits}")


    def find_union_of_areas(lists):
        # Combine all the input lists
        
        combined_list = lists
        #print(f"$$ combined_list = {combined_list}")
        # Sort the combined list by the start value of each interval
        combined_list.sort(key=lambda x: x[0])

        # Initialize the result list with the first interval
        result = [combined_list[0]]

        # Iterate through the combined list
        for interval in combined_list[1:]:
            # Get the last interval in the result list
            last_interval = result[-1]

            # Check for overlapping intervals
            if interval[0] <= last_interval[1]:
                # Update the end value of the last interval if necessary
                last_interval[1] = max(last_interval[1], interval[1])
            else:
                # Add the interval to the result list
                result.append(interval)

        return result


    #arr_diff_Second_rise = [[1,6], [2, 8], [5.5] , [7]]
    #arr_diff_Second_set = [[3,7], [5, 9], [6.5] , [8.5]]
    #seconds_difference_time  = 36
    print(f"arr_diff_Second_rise = {arr_diff_Second_rise}")
    print(f"arr_diff_Second_set = {arr_diff_Second_set}")


    satellite_visiting = [[] for _ in range(numSat)]
    for idx_sat in range(len(arr_diff_Second_rise)):
        for idx_visit in range(len(arr_diff_Second_rise[idx_sat])):
            satellite_visiting[idx_sat].append([arr_diff_Second_rise[idx_sat][idx_visit], arr_diff_Second_set[idx_sat][idx_visit]])

    #print(f"satellite_visiting = {satellite_visiting}")

    min_time_array = []
    max_time_array = []
    for i in range(len(arr_diff_Second_rise)):
        min_time_array.append(min(arr_diff_Second_rise[i]))
        max_time_array.append(max(arr_diff_Second_set[i]))

    start_time = 0
    end_time = seconds_difference_time
    print(f"%% end_time = {end_time}")


    sat_orbit = [[] for _ in range(numPlanes)]
    for orbit_idx in range(numPlanes):
        for sat_idx in range(num_sat_each_orbit):
            sat_orbit[orbit_idx].extend(satellite_visiting[sat_idx + orbit_idx * num_sat_each_orbit])

    #print(f"sat_orbit = {sat_orbit}")

    time_orbit = []
    for orbit_idx in range(numPlanes):
        #print(f"sat_orbit[orbit_idx] = {sat_orbit[orbit_idx]}")
        print(type(sat_orbit[orbit_idx]))
        time_orbit.append(find_union_of_areas(sat_orbit[orbit_idx]))

    print(f" time_orbit = {time_orbit}")


    xmin = []
    xmax = []
    non_visit_min = []
    non_visit_max = []
    y = []
    y_non_visit = []
    set_idx_sat =[0]
    a = 0



    arr_diff_Second_rise = [[] for idx in range(numPlanes)]
    arr_diff_Second_set = [[] for idx in range(numPlanes)]
    orbits_list = []
    for kdx_orbit in range(numPlanes):
        for kdx_visit in range(len(time_orbit[kdx_orbit])):
            arr_diff_Second_rise[kdx_orbit].append(time_orbit[kdx_orbit][kdx_visit][0])
            arr_diff_Second_set[kdx_orbit].append(time_orbit[kdx_orbit][kdx_visit][1])
        
    for i in range(len(arr_diff_Second_rise)):
        xmin.append(arr_diff_Second_rise[i])    # Rise time
        xmax.append(arr_diff_Second_set[i]) # Set time

        non_visit_start_list = list(arr_diff_Second_set[i])
        non_visit_start_list.append(start_time)
        non_visit_start_values = sorted(non_visit_start_list)


        non_visit_end_list = list(arr_diff_Second_rise[i])
        non_visit_end_list.append(end_time)
        non_visit_end_values = sorted(non_visit_end_list)

        non_visit_min.append(non_visit_start_values)
        non_visit_max.append(non_visit_end_values)

        y.append([a for i in range(len(arr_diff_Second_rise[i]))])
        y_non_visit.append([a for i in range(len(non_visit_min[i]))])
    #    if i< 5:
        a += 1
    #    else:
    #        a=0.5


    xmin1 = copy.deepcopy(xmin)
    xmax1 = copy.deepcopy(xmax)
    non_visit_min1 = copy.deepcopy(non_visit_min)
    non_visit_max1 = copy.deepcopy(non_visit_max)


    colors  = []
    colors1  = []
    for j in range(len(arr_diff_Second_rise)):
        colors.append('lightseagreen')
        colors1.append('pink')
    print(f" % colors = {colors}")
    print(f" % colors1 = {colors1}")


    for j in range(len(arr_diff_Second_rise)):
     for i in range(len(y[j])):
        if i ==0 and j==0:
            plt.hlines(y[j][i], xmin1[j][i], xmax1[j][i], colors[j], label="Visible",linewidth=10.0)
        else:

            plt.hlines(y[j][i], xmin1[j][i], xmax1[j][i], colors[j],linewidth=10.0)
     for ii in range(len(y_non_visit[j])):
        if ii==0 and j==0:
            plt.hlines(y_non_visit[j][ii], non_visit_min1[j][ii], non_visit_max1[j][ii], colors1[j], label="Non-visible",linewidth=10.0)
        else:

            plt.hlines(y_non_visit[j][ii], non_visit_min1[j][ii], non_visit_max1[j][ii], colors1[j],linewidth=10.0)


    ylim = np.linspace(0, numPlanes-1, num=numPlanes)
    print(f"ylim = {ylim}")
    Orbit_names = []
    for idx in range(1, numPlanes+1):
        Orbit_names.append(f'$\mathcal{{C}}_{str(idx)}$')
    plt.yticks(ylim, Orbit_names)
    plt.ylim([-1, numPlanes-0.3])

    
    start_process = 0
    end_process = seconds_difference_time
    print(f"end_process = {end_process}")
    xlim = np.linspace(start_process, end_process, 7)
    xlim_num = []
    for k in range(len(xlim)):
        xlim_num.append(str(int(xlim[k]/3600)))
    plt.xticks(xlim, xlim_num)

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['left'].set_color('none')

    plt.xlabel('Time [hour]')
    #plt.title('Visit pattern of a constellation with ' + str(numSat) + ' Satellites ' + 'in ' + str(numPlanes) + ' Orbits ' + ', inclination = ' + str(inclination)+ ', altitude =' + str(h_LEO/1000) + 'km' + ', GS = ' + GS_name , fontsize = 7)

    #plt.legend(loc='upper center',ncol=4, fancybox=False, shadow=False)
    plt.legend(loc= (0.53,0.93),ncol=4, fancybox=False, shadow=False)
    #plt.legend(loc=(x, y))
    plt.tick_params(left = False)

    plt.show()

else:

    print('This distribution is not good')

