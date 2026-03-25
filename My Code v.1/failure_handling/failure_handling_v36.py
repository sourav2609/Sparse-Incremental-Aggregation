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
from operator import itemgetter
import math



def Extract_satellite_rise_set(numSat1):

    def Satellites_in_orbits_constellations(LEOs, groundstation, elevation_angle=10, ts = None, safetyMargin = 0):

        ts = sf.load.timescale()
        time1 = ts.utc(2021,9,23,00,00,00)      # Start point, UTC time
        time2 = ts.utc(2021,9,25,00,00,00)      # End point, UTC time

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
    #groundstation = wgs84.latlon(+53.00, 8.80) #GS-BR
    #GS_name = 'Bremen'
    #groundstation = wgs84.latlon(+49.00, 2) #Paris
    #GS_name = 'Paris' 
    #groundstation = wgs84.latlon(-23.55, -46.63) #Brazil 
    #GS_name = 'Sao Paulo' 
    #groundstation = wgs84.latlon(+90.0, 0.0) #GS-NP


    #groundstation = [wgs84.latlon(+53.00, 8.80), wgs84.latlon(-23.55, -46.63), wgs84.latlon(+49.00, 2)]
    groundstation = [wgs84.latlon(+53.00, 8.80)]

    h_LEO = 2000e3
    inclination = 60
    inc_GEO = 0
    numSat = numSat1
    numPlanes = 1
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

    x = 1
    if x == 1:

        satellites_index = []
        arr_diff_Second_rise = []
        arr_diff_Second_set = []
        seconds_difference_time = []

        for idx_gs in range(len(groundstation)):

            LG = LEO_GEO(r_E, gm, h_GEO)
            #GEO = LG.GEO(inc_GEO)
            LEOs = LG.walkerConstellation(h_LEO, inclination, numSat, numPlanes, phasing, name = "Sat")

            arr_diff_Second_rise1, arr_diff_Second_set1, seconds_difference_time1 =  Satellites_in_orbits_constellations(LEOs, groundstation[idx_gs], elevation_angle=10, ts = None, safetyMargin = 0)
        
        
            #print(f" seconds_difference_time1 = {seconds_difference_time1}")
            arr_diff_Second_rise.extend(arr_diff_Second_rise1)
            arr_diff_Second_set.extend(arr_diff_Second_set1)
            seconds_difference_time = seconds_difference_time1
            
        #print(f"$$$CHECK = {arr_diff_Second_set}")         
        for k in range(len(arr_diff_Second_rise)):
                
                satellites_index.append([k for idx in range(len(arr_diff_Second_rise[k]))])

        All_satellite_visits = []
        All_satellite_index = []
        for k in range(len(arr_diff_Second_rise)):
                All_satellite_visits.extend(arr_diff_Second_rise[k])
                All_satellite_index.extend(satellites_index[k])

        #print(f"All_satellite_visits = {All_satellite_visits}")
        #print(f"All_satellite_index = {All_satellite_index}")

        indexes_satellite_visits = np.argsort(All_satellite_visits)

        Satellite_indexes = []
        for g1 in range(len(All_satellite_index)):
                Satellite_indexes.append(All_satellite_index[indexes_satellite_visits[g1]])
        #print(f"^^ Satellite_indexes = {Satellite_indexes}")


        #print(f"***SAT_IND  = {satellites_index}")

        #print(f"$$ indexes_satellite_visits = {indexes_satellite_visits}")





        satellite_visiting = [[] for _ in range(len(arr_diff_Second_rise))]
        for idx_sat in range(len(arr_diff_Second_rise)):
            for idx_visit in range(len(arr_diff_Second_rise[idx_sat])):
                satellite_visiting[idx_sat].append([arr_diff_Second_rise[idx_sat][idx_visit], arr_diff_Second_set[idx_sat][idx_visit]])

        #print(f"satellite_visiting = {satellite_visiting}")

    print(f"arr_diff_Second_rise = {arr_diff_Second_rise}")
    print(f"arr_diff_Second_set = {arr_diff_Second_set}")
    return arr_diff_Second_rise, arr_diff_Second_set




    
def algorithms_delay(arr_diff_Second_rise, arr_diff_Second_set, start_process, time_comm_ISL_distribution, time_comm_ISL_aggregation, time_training, delay_learning, delay_transmission):


    def source_distribution_time(num_satellites, source_id):
        
        items =np.arange(num_satellites)
        source = source_id
        items1 = np.roll(items, -source)
        #print(f"items = {items1}")

        len_distribution_list = int(len(items1)/2) + 1
        distribution_list = [[] for i in range(len_distribution_list)]
        distribution_list[0] = [items1[0]]
        items1 = np.delete(items1, 0)
        for idx in range(1, len_distribution_list):
            if len(items1) != 0 and len(items1) >= 2:
             distribution_list[idx] = [items1[0], items1[-1]]
             items1 = np.delete(items1, 0)
             items1 = np.delete(items1, -1)
            elif len(items1) != 0 and len(items1) == 1:
             distribution_list[idx] = [items1[0]]
             items1 = np.delete(items1, 0)    
        return distribution_list

    def Time_Reception(num_satellites, time_start, time_isl, distribution_list):
          #print(f"1111 distribution_list = {distribution_list}")
          time_reception = [0 for idx_sat in range(num_satellites)]
          
          time_reception[distribution_list[0][0]] = time_start
          #print(f"EEEE time_reception = {time_reception}")
          
          for idx_sat in range(1, len(distribution_list)):
                    
             if  len(distribution_list[idx_sat]) == 2: 
                    sat_1, sat_2 = distribution_list[idx_sat]
                    if idx_sat == 1:
                      source = distribution_list[0]
                      time_reception[sat_1] = time_reception[source[0]] + time_isl[source[0]]
                      time_reception[sat_2] = time_reception[source[0]] + time_isl[source[0]]
                    elif idx_sat > 1:  
                        
                           sat_past_1 = distribution_list[idx_sat-1][0]
                           sat_past_2 = distribution_list[idx_sat-1][1]
                           time_reception[sat_1] = time_reception[sat_past_1] + time_isl[sat_past_1]
                           time_reception[sat_2] = time_reception[sat_past_2] + time_isl[sat_past_2]   

             elif len(distribution_list[idx_sat]) == 1: 
                         sat_past_1 = distribution_list[idx_sat-1][0]
                         sat_past_2 = distribution_list[idx_sat-1][1]                        
                         sat_1 = distribution_list[idx_sat]
                         time_reception[sat_1[0]] = max(time_reception[sat_past_1] + time_isl[sat_past_1], time_reception[sat_past_2] + time_isl[sat_past_2])
          #print(f"222EEEE time_reception = {time_reception}")
          return time_reception

    def determine_started_points_for_aggregation1(number_satellites, sink_id):
     satellites_number = np.arange(number_satellites)
     start_satellite1 = sink_id - int(number_satellites/2)
     if start_satellite1 < 0:
           start_satellite1 = np.remainder(start_satellite1, number_satellites)
     if start_satellite1 == 0:
           start_satellite2 = number_satellites - 1 
     else:                
        start_satellite2 = start_satellite1+1
        if start_satellite2 >= number_satellites:
            start_satellite2 = start_satellite1 - 1     
     return start_satellite1, start_satellite2
    
    def group_pattern_satellites_aggregation(number_satellites, sink_id):
 
    
        start_1, start_2 = determine_started_points_for_aggregation1(number_satellites, sink_id)
        start = start_1
        end = sink_id


        len_list  = int((number_satellites/2) + 1)
        list = []
        for i in range(len_list):
            value = (start - i) % number_satellites
            list.append(value)

        list1 = []
        for i in range(len_list):
            value = (start + i) % number_satellites
            list1.append(value)    


        group_satellite_1 = []
        group_satellite_2 = []
        if list[1] == start_2:
            group_satellite_1 = list1
            group_satellite_2 = list
        else:
            group_satellite_1 = list
            group_satellite_2 = list1    

        group_satellite_2 = group_satellite_2[1:]

        #print(f" sink = {sink_id} , list_main = {group_satellite_1}, list_main2 = {group_satellite_2}")
        return group_satellite_1, group_satellite_2

    def detect_source(arr_diff_Second_rise, arr_diff_Second_set, start_process):
        First_visit = [-1 for idx_sat in range(len(arr_diff_Second_rise))]
        for idx_sat in range(len(arr_diff_Second_rise)):
            for idx_visit in range(len(arr_diff_Second_rise[idx_sat])):
                if First_visit[idx_sat] == -1:
                 if (arr_diff_Second_rise[idx_sat][idx_visit] < start_process and arr_diff_Second_set[idx_sat][idx_visit] > start_process) or (arr_diff_Second_rise[idx_sat][idx_visit] > start_process and arr_diff_Second_set[idx_sat][idx_visit] > start_process):
                    First_visit[idx_sat] = max(arr_diff_Second_rise[idx_sat][idx_visit], start_process)

        return First_visit

    def detect_sink_proposed_approach(arr_diff_Second_rise, arr_diff_Second_set, time_finish_process):
        
        #print(f"^^^ time_finish_process = {time_finish_process}")
        visibility_of_satellite = [-1 for idx_sat in range(len(arr_diff_Second_rise))]
        for idx_sat in range(len(arr_diff_Second_rise)):
            for idx_visit in range(len(arr_diff_Second_rise[idx_sat])):
                if visibility_of_satellite[idx_sat] == -1:
                 if (arr_diff_Second_rise[idx_sat][idx_visit] <= time_finish_process  and arr_diff_Second_set[idx_sat][idx_visit] >= time_finish_process):
                                visibility_of_satellite[idx_sat] =  time_finish_process
                                #print(f"&&& visibility_of_satellite[idx_sat] = {visibility_of_satellite[idx_sat]}")
                 if   visibility_of_satellite[idx_sat] == -1: 
                    #print(f"HIIIII")     
                    #print(f"*** arr_diff_Second_rise[idx_sat][idx_visit]  = {arr_diff_Second_rise[idx_sat][idx_visit]}")
                    #print(f"### arr_diff_Second_set[idx_sat][idx_visit]  = {arr_diff_Second_set[idx_sat][idx_visit]}")
                    if (arr_diff_Second_rise[idx_sat][idx_visit] >= time_finish_process  and arr_diff_Second_set[idx_sat][idx_visit] >= time_finish_process):
                                                    #print(f"HEEEELLLLLLLOOOO")  
                                                    visibility_of_satellite[idx_sat] =  arr_diff_Second_rise[idx_sat][idx_visit]
        return visibility_of_satellite   

    def detect_sink_proposed_approach_failure_handling(arr_diff_Second_rise, arr_diff_Second_set, time_finish_process):
        
        #print(f"^^^ time_finish_process = {time_finish_process}")
        visibility_of_satellite = 1e30
        for idx_visit in range(len(arr_diff_Second_rise)):
                if visibility_of_satellite == 1e30:
                 if (arr_diff_Second_rise[idx_visit] <= time_finish_process  and arr_diff_Second_set[idx_visit] >= time_finish_process):
                                visibility_of_satellite =  time_finish_process
                                #print(f"&&& visibility_of_satellite[idx_sat] = {visibility_of_satellite[idx_sat]}")
                 if   visibility_of_satellite == 1e30: 
                    #print(f"HIIIII")     
                    #print(f"*** arr_diff_Second_rise[idx_sat][idx_visit]  = {arr_diff_Second_rise[idx_sat][idx_visit]}")
                    #print(f"### arr_diff_Second_set[idx_sat][idx_visit]  = {arr_diff_Second_set[idx_sat][idx_visit]}")
                    if (arr_diff_Second_rise[idx_visit] >= time_finish_process  and arr_diff_Second_set[idx_visit] >= time_finish_process):
                                                    #print(f"HEEEELLLLLLLOOOO")  
                                                    visibility_of_satellite =  arr_diff_Second_rise[idx_visit]
        #print(f"visibility_of_satellite = {visibility_of_satellite}")
        return visibility_of_satellite

    def time_after_aggregation_func(time_finish_learning_satellite, pattern_distribution_satellites,time_comm_ISL):
        #print(f"$$$ time_finish_learning_satellite = {time_finish_learning_satellite}")  
        #print(f"$$$ pattern_distribution_satellites = {pattern_distribution_satellites}")
        check_sink_time = [0 for i in range(2)]
        time_after_aggregation = [0 for idx_sat in range(len(time_finish_learning_satellite))]
        for i_group in range(2):
          for idx_sat in range(len(pattern_distribution_satellites[i_group])):
              idx1 = pattern_distribution_satellites[i_group][idx_sat]
              if idx_sat == 0:
                    time_after_aggregation[idx1] = time_finish_learning_satellite[idx1]
                    idxx = idx1
              elif idx_sat != 0:
                  time_after_aggregation[idx1] = max(time_finish_learning_satellite[idx1], time_after_aggregation[idxx] + time_comm_ISL[idxx])
                  idxx = idx1
              if   idx1 == pattern_distribution_satellites[i_group][-1]:  
                  check_sink_time[i_group] = time_after_aggregation[idx1]
        #print(f"check_sink_time = {check_sink_time}")
        time_after_aggregation[idx1] = max(check_sink_time[0], check_sink_time[1])
        return time_after_aggregation    

    def update_time_without_failure_handling(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_id, time_finish_process_updated):
        id_sink_updated = sink_satellite_id
        time_sink_updated = 1e30
        for idx_visit in range(len(arr_diff_Second_rise[id_sink_updated])):
                if time_sink_updated == 1e30:
                 if (arr_diff_Second_rise[id_sink_updated][idx_visit] <= time_finish_process_updated  and arr_diff_Second_set[id_sink_updated][idx_visit] >= time_finish_process_updated):
                                time_sink_updated =  time_finish_process_updated
                 if   time_sink_updated == 1e30: 
                    #print(f"HIIIII")     
                    #print(f"*** arr_diff_Second_rise[idx_sat][idx_visit]  = {arr_diff_Second_rise[idx_sat][idx_visit]}")
                    #print(f"### arr_diff_Second_set[idx_sat][idx_visit]  = {arr_diff_Second_set[idx_sat][idx_visit]}")
                    if (arr_diff_Second_rise[id_sink_updated][idx_visit] >= time_finish_process_updated  and arr_diff_Second_set[id_sink_updated][idx_visit] >= time_finish_process_updated):
                                                    #print(f"HEEEELLLLLLLOOOO")  
                                                    time_sink_updated =  arr_diff_Second_rise[id_sink_updated][idx_visit]

        return id_sink_updated, time_sink_updated
    
    def update_time_with_failure_handling(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_id, sink_satellite_time, time_training_delayed, time_comm_ISL_aggregation_delayed):
        
        time_reception = Time_Reception(len(arr_diff_Second_rise), time_start, time_comm_ISL_distribution, pattern_distribution_satellites)
        print(f"% time_reception = {time_reception}")
        time_finish_learning_satellite = [time_reception[idx_sat] + time_training_delayed[idx_sat] for idx_sat in range(len(arr_diff_Second_rise))]
        #print(f"time_finish_learning_satellite = {time_finish_learning_satellite}")
        reception_sink = detect_sink_proposed_approach(arr_diff_Second_rise, arr_diff_Second_set, max(time_finish_learning_satellite))
        sink_satellite_id1, sink_satellite_time1 =  min(enumerate(reception_sink), key=itemgetter(1))    
        pattern_aggregation_satellites = group_pattern_satellites_aggregation(len(arr_diff_Second_rise), sink_satellite_id1)
        #print(f"pattern_aggregation_satellites = {pattern_aggregation_satellites}")
        #print(f"time_comm_ISL_aggregation_delayed  = {time_comm_ISL_aggregation_delayed}")    
        time_after_aggregation1 = time_after_aggregation_func(time_finish_learning_satellite, pattern_aggregation_satellites, time_comm_ISL_aggregation_delayed)
        
        time_end_transmission_satellite_GS = max(time_after_aggregation1)
        #print(f"^^^ time_after_aggregation  ={time_end_transmission_satellite_GS}")
        
        sink_satellite_id = sink_satellite_id1  # The first time that one satellite from orbit visit the GS at time time_finish_process
        sink_satellite_time = time_end_transmission_satellite_GS

        return sink_satellite_id, sink_satellite_time

    def transmission_time_sink_GS(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_time, sink_satellite_id):

            visibility_of_satellite = -1 
            
            for idx_visit in range(len(arr_diff_Second_rise[sink_satellite_id])):
                #print(arr_diff_Second_rise[sink_satellite_id][idx_visit], sink_satellite_time, arr_diff_Second_set[sink_satellite_id][idx_visit])
                if visibility_of_satellite == -1:
                 if (arr_diff_Second_rise[sink_satellite_id][idx_visit] <= sink_satellite_time  and arr_diff_Second_set[sink_satellite_id][idx_visit] >= sink_satellite_time):
                                visibility_of_satellite =  sink_satellite_time
                                #print(f"HEEEELLLLLLLOOOO")  
                                #print(f"777 visibility_of_satellite[idx_sat] = {visibility_of_satellite}")
                 if   visibility_of_satellite == -1: 
                    if (arr_diff_Second_rise[sink_satellite_id][idx_visit] >= sink_satellite_time  and arr_diff_Second_set[sink_satellite_id][idx_visit] >= sink_satellite_time):
                                                   visibility_of_satellite =  arr_diff_Second_rise[sink_satellite_id][idx_visit]
            return visibility_of_satellite

    def update_time_multi_hop_FH(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_id, time_start_process_multi_hop_satellite, time_comm_ISL_aggregation):
            
            check_sat_idx = -1
            visibility_of_satellite = time_start_process_multi_hop_satellite
            check_whether_previos_sink = 0
            idxx = copy.copy(sink_satellite_id)
            while check_sat_idx == -1:
              
              if check_whether_previos_sink == 0:
                idxx1 = (idxx)  % len(arr_diff_Second_rise) 
                
              else:
                idxx1 = (idxx1+1)  % len(arr_diff_Second_rise)
                visibility_of_satellite = visibility_of_satellite + time_comm_ISL_aggregation[idxx1]

              #if idxx1 != sink_satellite_id or (idxx1 == sink_satellite_id and check_whether_previos_sink == 0): 
                #print(f"----- idxx1 = {idxx1}, check_sat_idx = {check_sat_idx}")
              check_whether_previos_sink = 1                
              for idx_visit in range(len(arr_diff_Second_rise[idxx1])):
                        #print(f"@@%%%idx_vist = {idx_visit}, @@@idxx1 = {idxx1}")    
                    #print(arr_diff_Second_rise[sink_satellite_id][idx_visit], sink_satellite_time, arr_diff_Second_set[sink_satellite_id][idx_visit])
                        if  check_sat_idx == -1:  
                            #print(f"GGGGGG checkkkkk")
                            if (arr_diff_Second_rise[idxx1][idx_visit] <= visibility_of_satellite  and arr_diff_Second_set[idxx1][idx_visit] >= visibility_of_satellite):                                                
                                                    check_sat_idx = 1

                  
            return idxx1, visibility_of_satellite



    #print(detect_source(arr_diff_Second_rise, arr_diff_Second_set, start_process))
    reception_source = detect_source(arr_diff_Second_rise, arr_diff_Second_set, start_process)   # The first time that one satellite from orbit visit the GS
    #print(f"AAA reception_source = {reception_source}")
    source_satellite_id, source_satellite_time = min(enumerate(reception_source), key=itemgetter(1))
    #print(f"****%%% = {source_satellite_id, source_satellite_time}")
    if source_satellite_time != -1:
        time_start = source_satellite_time  + time_comm_GS_satellite
        #print(f"TTTtime_start = {time_start}")
        pattern_distribution_satellites = source_distribution_time(len(arr_diff_Second_rise), source_satellite_id)
        #print(f"AAA pattern_distribution_satellites = {pattern_distribution_satellites}")
        #print(f"BBB time_comm_ISL_distribution = {time_comm_ISL_distribution}")
        time_reception = Time_Reception(len(arr_diff_Second_rise), time_start, time_comm_ISL_distribution, pattern_distribution_satellites)
        #print(f"RecepTION = {time_reception}")
        time_finish_learning_satellite = [time_reception[idx_sat] + time_training[idx_sat] for idx_sat in range(len(arr_diff_Second_rise))]
        #print(f"111YYY time_finish_learning_satellite = {time_finish_learning_satellite}")
        #print()
        time_sinks = [0 for idx_s in range(len(arr_diff_Second_rise))]
        time_visited_sinks = [0 for idx_s in range(len(arr_diff_Second_rise))]
        pattern_all_sinks = [[] for idx_s in range(len(arr_diff_Second_rise))]
        for idx_sink in range(len(arr_diff_Second_rise)):
                                            pattern = group_pattern_satellites_aggregation(len(arr_diff_Second_rise), idx_sink)
                                            #print(pattern)
                                            time_ready_transmit_sink_GS = time_after_aggregation_func(time_finish_learning_satellite, pattern, time_comm_ISL_aggregation) 
                                            pattern_all_sinks[idx_sink] = pattern
                                            #print(f" %%% time_ready_transmit_sink_GS  = {time_ready_transmit_sink_GS}")      
                                            time_sinks[idx_sink] = max(time_ready_transmit_sink_GS)
                                            
                                            time_visited_sinks[idx_sink] = transmission_time_sink_GS(arr_diff_Second_rise, arr_diff_Second_set, time_sinks[idx_sink], idx_sink)
       
        sink_satellite_id = np.array(time_visited_sinks).argmin()
        sink_satellite_time = time_visited_sinks[sink_satellite_id] 
        #print(f"^^^^ sink_satellite_time  ={sink_satellite_time}")
        #print(f"^^^^ sink_satellite_id  ={sink_satellite_id}")
        pattern_assigned = pattern_all_sinks[sink_satellite_id]
        time_training_delayed = time_training + delay_learning
        time_training_delayed = [time_training[idxxx]+delay_learning[idxxx] for idxxx in range(len(delay_learning))]
        #print(f"YYY delay_learning = {delay_learning}")
        #time_comm_ISL_aggregation_delayed[delay_transmission_index] = time_comm_ISL_aggregation_delayed[delay_transmission_index] + delay_transmission_duration
        time_comm_ISL_aggregation_delayed = [time_comm_ISL_aggregation[iddxx] + delay_transmission[iddxx] for iddxx in range(len(delay_transmission))]
        #print(f"&&& time_comm_ISL_aggregation_delayed = {time_comm_ISL_aggregation_delayed}")
        #print(f"YYY delay_transmission = {delay_transmission}")
        #print(f"LLLLL time_training_delayed = {time_training_delayed}")
        #print(f"CCCC time_comm_ISL_aggregation_delayed = {time_comm_ISL_aggregation_delayed}")
        time_finish_learning_satellite_delayed = [time_reception[idx_sat] + time_training_delayed[idx_sat] for idx_sat in range(len(arr_diff_Second_rise))]
        #print(f"2222YYY time_finish_learning_satellite_delayed  = {time_finish_learning_satellite_delayed}")
        #print(f"000 = {time_after_aggregation_func(time_finish_learning_satellite_delayed, pattern_assigned, time_comm_ISL_aggregation_delayed)}")
        t_now1 = max(time_after_aggregation_func(time_finish_learning_satellite_delayed, pattern_assigned, time_comm_ISL_aggregation_delayed))
        
        #transmission_time_sink_GS(arr_diff_Second_rise, arr_diff_Second_set, time_sinks[idx_sink], idx_sink)
        #print(f" %%% time_after_aggregation_func(time_finish_learning_satellite_delayed, pattern_assigned, time_comm_ISL_aggregation_delayed) = {time_after_aggregation_func(time_finish_learning_satellite_delayed, pattern_assigned, time_comm_ISL_aggregation_delayed)}")
        #print(f"111 t_now = {t_now1}")
        #print(f"222 sink_satellite_time = {sink_satellite_time}")

        ## FAILURE HANDLING APPROACH
        #t_now = 13
        #print(f"sink_satellite_id = {sink_satellite_id}") 
        #time_failure_handling = [0 for idx_sat in range(len(arr_diff_Second_rise))] 
        if t_now1 > sink_satellite_time:
        
                    list_decrease = [[sink_satellite_id] for i in range(len(arr_diff_Second_rise))]
                    list_increase = [[sink_satellite_id] for i in range(len(arr_diff_Second_rise))]
                    time_failure_handling = [0 for idx_sat in range(len(arr_diff_Second_rise))] 
                    for idx_sat in range(len(arr_diff_Second_rise)):
                                                            start = sink_satellite_id
                                                            start1 = copy.deepcopy(start)
                                                            start2 = copy.deepcopy(start)
                                                            end = idx_sat
                                                    
                                                            while start1 != end:
                                                                                                    start1 = (start1 - 1) % len(arr_diff_Second_rise)
                                                                                                    list_decrease[idx_sat].append(start1)
                                                            
                                                            while start2 != end:
                                                                                                    start2 = (start2 + 1) % len(arr_diff_Second_rise)
                                                                                                    list_increase[idx_sat].append(start2)

                    time_failure_handling = [0 for idx_sat in range(len(arr_diff_Second_rise))] 
                    for idxx in range(len(arr_diff_Second_rise)):
                                        time_route1 = t_now1
                                        time_route2 = t_now1
                                        if len(list_increase[idxx]) != 1:
                                                                    for idx_sats in range(len(list_increase[idxx])-1):
                                                                                            time_route1 = time_route1 + time_comm_ISL_aggregation[list_increase[idxx][idx_sats]]
                            #print(f"time_route1 = {time_route1}")

                                        if len(list_decrease[idxx]) != 1:    
                                                                    for idx_sats in range(len(list_decrease[idxx])-1):
                                
                                                                                                time_route2 = time_route2 + time_comm_ISL_aggregation[list_decrease[idxx][idx_sats]]
                                                                                                    #print(f"time_route2 = {time_route2}")
                            
                                        time_failure_handling[idxx] = min(time_route1, time_route2)
                                        
                            

                    time_failure_handling_plausible_time = [1e30 for i in range(len(time_failure_handling))]
                    for idx in range(len(time_failure_handling)):
                                            x = detect_sink_proposed_approach_failure_handling(arr_diff_Second_rise[idx], arr_diff_Second_set[idx], time_failure_handling[idx])
                                            #print(x)
                                            time_failure_handling_plausible_time[idx] = x
                    #print(f"time_failure_handling_plausible_time = {time_failure_handling_plausible_time}")
                    
                    #print(f"time_failure_handling_plausible_time = {time_failure_handling_plausible_time}")
                    sink_id_updated_with_FH = np.array(time_failure_handling_plausible_time).argmin()
                    sink_time_updated_with_FH = time_failure_handling_plausible_time[sink_id_updated_with_FH]
                    #print(f"sink_id_updated_with_FH = {sink_id_updated_with_FH}")


                    time_finish_process_updated_without_FH = copy.copy(t_now1)
                    sink_id_updated_without_FH, sink_time_updated_without_FH = update_time_without_failure_handling(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_id, time_finish_process_updated_without_FH)

                    
                    time_start_process_multi_hop_satellite = copy.copy(t_now1)
                    #print(f"222 time_start_process_multi_hop_satellite = {time_start_process_multi_hop_satellite}")
                    sink_id_updated_multi_hop_FH, sink_time_updated_multi_hop_FH = update_time_multi_hop_FH(arr_diff_Second_rise, arr_diff_Second_set, sink_satellite_id, time_start_process_multi_hop_satellite, time_comm_ISL_aggregation)
                    #print(f"sink_id_updated_multi_hop_FH, sink_time_updated_multi_hop_FH  = {sink_id_updated_multi_hop_FH, sink_time_updated_multi_hop_FH}") 

        elif t_now1 == sink_satellite_time:
            sink_id_updated_with_FH = sink_satellite_id
            sink_time_updated_with_FH = transmission_time_sink_GS(arr_diff_Second_rise, arr_diff_Second_set, t_now1, sink_satellite_id)        
        
        

        #print(f"222 sink_satellite_time = {sink_satellite_time}")






        #print(f"%%% sink_id_updated_with_FH = {sink_id_updated_with_FH}, sink_time_updated_with_FH = {sink_time_updated_with_FH}")
        #print(f"%%% sink_id_updated_without_FH = {sink_id_updated_without_FH}, sink_time_updated_without_FH = {sink_time_updated_without_FH}")
        return t_now1, sink_satellite_time, sink_id_updated_with_FH, sink_time_updated_with_FH, sink_id_updated_without_FH, sink_time_updated_without_FH, sink_id_updated_multi_hop_FH, sink_time_updated_multi_hop_FH
    
    else:
            return []
    
#Delay_training = np.random.gamma(shape = 4, scale = 20, size = 1)

#[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32] , 26, 28, 30, 32, 34, 36, 38, 40 , 22, 24, 26, 28, 30, 32, 34, 36, 38, 40
Num_sat = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
#Num_sat = [4]
#Num_sat = [8]
#time_learning_delay =     [5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1100, 1200, 1500, 1700]
#Num_sat = [10]
#time_learning_delay =     [1000]

Iteration_delay_transmission = 1000
Iteration_delay_training = 1000

time_learning_delay = []
np.random.seed(0)
time_transmission_delay = []
#time_learning_delay =     [1000]

#start_time = [0, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000,13000,14000,16000,18000, 19000, 20000,21000,22000, 25000, 27000, 28000, 29000, 30000, 32000, 33000, 34000, 35000, 37000, 40000, 45000, 47000, 48000, 50000, 60000, 70000, 80000,85000, 90000]
start_time = np.linspace(0, 50000, num=20)
#start_time = [0]
#print(start_time)

#start_time = [0, 20000, 40000, 60000, 75000, 90000]
#start_time = [0, 13000]
#start_time = [0]
#Num_sat = [4]
#time_learning_delay = [500]
#start_time = [9000]
time_comm_GS_satellite = 50






time_train = 480


delay_learning1 = [[] for idx in range(Iteration_delay_transmission)]
delay_transmission1 = [[] for idx in range(Iteration_delay_transmission)]

for idx_time_learning in range(Iteration_delay_transmission):

        for idxx_Sat in range(Num_sat[-1]):
                 #print(index_satellite_delay)
                  delay_learning1[idx_time_learning].extend(np.random.gamma(shape =25, scale = 25, size = 1))
                  #delay_learning1[idx_time_learning].append(0)
                  delay_transmission1[idx_time_learning].extend(np.random.exponential(40, 1))
                  #delay_transmission1[idx_time_learning].append(0)


print(f"delay_learning1 = {delay_learning1}")
print(f"delay_transmission1 = {delay_transmission1}")

def end_to_end_transmission_time(distance, c , Rate, D):
            c = 3e8
            t_propagation = distance / c   # Seconds
            t_transmission = D / Rate
            t_end_to_end = t_propagation + t_transmission
            return t_end_to_end



delay_num_sat_without_FH = np.zeros(len(Num_sat))
delay_num_sat_with_FH = np.zeros(len(Num_sat))
delay_num_sat_multi_hop_with_FH = np.zeros(len(Num_sat))
for idx_sat in range(len(Num_sat)):

    c = 3e8
    Rate = 1 * (10**9)
    D = 500000000000
    h_LEO = 2000e3
    r_E = 6371e3
    Real_dist = np.sin(np.deg2rad(360/(Num_sat[idx_sat]*2))) * (r_E + h_LEO) * 2
    time_isl = end_to_end_transmission_time(Real_dist, c , Rate, D)
    print(f" TTT = {time_isl}")
    time_isl = 50
    sats_rise, sats_set = Extract_satellite_rise_set(Num_sat[idx_sat])
    matrix_time_learning_delay_num_satellite_without_FH = np.zeros((Iteration_delay_transmission, len(start_time)))
    matrix_time_learning_delay_num_satellite_with_FH = np.zeros((Iteration_delay_transmission, len(start_time)))
    matrix_time_learning_delay_num_satellite_multi_hop_FH = np.zeros((Iteration_delay_transmission, len(start_time)))
    time_comm_ISL_distribution = [time_isl for _ in range(Num_sat[idx_sat])]
    time_comm_ISL_aggregation = [time_isl for _ in range(Num_sat[idx_sat])]
    time_training = [time_train for _ in range(Num_sat[idx_sat])]
    for idx_time_learning in range(Iteration_delay_transmission):
        
        delay_learning = delay_learning1[idx_time_learning][0:Num_sat[idx_sat]]
        #print(f"$ delay_learning = {delay_learning}")
        delay_transmission = delay_transmission1[idx_time_learning][0:Num_sat[idx_sat]]
        
        for idx_num_start in range(len(start_time)):
            #print(f"time_learning_delay[idx_time_learning] = {time_learning_delay[idx_time_learning]}")
            #print(f"Num_sat[idx_num_sat] = {Num_sat[idx_num_start]}")
            start_process = start_time[idx_num_start]
            
            t_now1, time_without_delay, idx_with_FH, time_with_FH, idx_without_FH, time_without_FH, id_updated_multi_hop_FH, time_updated_multi_hop_FH = algorithms_delay(sats_rise, sats_set, start_time[idx_num_start], time_comm_ISL_distribution, time_comm_ISL_aggregation, time_training, delay_learning, delay_transmission)
            #print(f"time_without_delay = {time_without_delay}, t_now1 = {t_now1}, time_with_FH = {time_with_FH}, time_without_FH = {time_without_FH}, time_updated_multi_hop_FH = {time_updated_multi_hop_FH}")
            matrix_time_learning_delay_num_satellite_without_FH[idx_time_learning, idx_num_start] = time_without_FH - t_now1
            matrix_time_learning_delay_num_satellite_with_FH[idx_time_learning, idx_num_start] = time_with_FH - t_now1
            matrix_time_learning_delay_num_satellite_multi_hop_FH[idx_time_learning, idx_num_start] = time_updated_multi_hop_FH - t_now1
            #print(f"matrix_time_learning_delay_num_satellite_without_FH = {matrix_time_learning_delay_num_satellite_without_FH}")
    #print(f"$$matrix_time_learning_delay_num_satellite_without_FH = {matrix_time_learning_delay_num_satellite_without_FH}")
    #print(f"$$matrix_time_learning_delay_num_satellite_with_FH = {matrix_time_learning_delay_num_satellite_with_FH}")
    delay_num_sat_without_FH[idx_sat] =  np.mean(matrix_time_learning_delay_num_satellite_without_FH[:, :])  
    delay_num_sat_with_FH[idx_sat] =  np.mean(matrix_time_learning_delay_num_satellite_with_FH[:, :])
    delay_num_sat_multi_hop_with_FH[idx_sat] =  np.mean(matrix_time_learning_delay_num_satellite_multi_hop_FH[:, :])


print(f"delay_num_sat_without_FH = {delay_num_sat_without_FH}")
print(f"delay_num_sat_with_FH = {delay_num_sat_with_FH}")
print(f"delay_num_sat_multi_hop_with_FH = {delay_num_sat_multi_hop_with_FH}")


max_ylim1  = max(delay_num_sat_without_FH)
max_ylim2  = max(delay_num_sat_multi_hop_with_FH)
max_ylim = max(max_ylim1, max_ylim2)

plt.ylim([0, 8000])
plt.xlim([0, 52])
plt.xlabel('Number of satellites')
plt.ylabel('Delay in one orbit')

plt.plot(Num_sat, delay_num_sat_without_FH, label = '3) Without Failure handling (FH)')
plt.plot(Num_sat, delay_num_sat_with_FH, label = '1) Proposed, with Failure handling (FH)')
plt.plot(Num_sat, delay_num_sat_multi_hop_with_FH, label = '2) Clock-wise, with Failure handling (FH)')
plt.legend()
plt.show()

