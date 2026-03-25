# FEDISL with the accurate implementation time
# Implementation FL with intra-link satellites
# In the start point, at first the parameter server sends the initial weights to the satellites when it sees each one


from import_satellite_visitings_time_v7 import vistings
import numpy as np
import copy
import time
class FEDISL:
#, time_aggregation_ISL
    def __init__(self, num_satellite, num_planes, parameter_server, time_calculate_the_sink_node):
        self.num_satellite = num_satellite
        self.num_planes = num_planes
        self.parameter_server = parameter_server
        self.time_calculate_the_sink_node = time_calculate_the_sink_node
        self.time_aggregation_ISL = time_aggregation_ISL


    def queue(self,index,  TIME_AGGREGATE, queue_operation, time_operation, time_now_vector, vector_time_start, vector_time_end, max_value):

      # Only bring the first operation to do and then put in at the end of queue_operations
     first_operation_queue = queue_operation[0][0]
     #print(f" first_operation_queue = {first_operation_queue}")
     if first_operation_queue == 'transmit from GEO to LEO':
            index = [[1000], [1000]]   
     #print(f"inndex = {index}")
     if first_operation_queue != 'aggregate':
      #print(f"index = {index}")
      for idx_orbit in range(len(vector_time_start)):
       
       first_operation_queue = queue_operation[idx_orbit][0]
       print(idx_orbit, first_operation_queue)
       index, A = self.dispatcher(index, first_operation_queue, time_now_vector[idx_orbit], time_operation[idx_orbit], vector_time_start[idx_orbit], vector_time_end[idx_orbit], max_value[idx_orbit],idx_orbit)
       time_now_vector[idx_orbit] = A
       queue_operation[idx_orbit].pop(0)
       queue_operation[idx_orbit].append(first_operation_queue)
      return index, time_now_vector

     elif first_operation_queue == 'aggregate':
             idx_orbit = 100    # Only for having one number for the idx_orbit
             
             index, time_now_vector = self.dispatcher(index, first_operation_queue, time_now_vector, time_operation, vector_time_start, vector_time_end, max_value, idx_orbit)

             #print('TIME_AGGREGATE = ' + str(TIME_AGGREGATE))
             for idx_orbit in range(len(vector_time_start)):
                    first_operation_queue = queue_operation[idx_orbit][0]
                    queue_operation[idx_orbit].pop(0)
                    queue_operation[idx_orbit].append(first_operation_queue)
             return time_now_vector


    def dispatcher(self, index, first_operation_queue, time_now, time_operation, vector_time_start, vector_time_end, max_value, idx_orbit):
        time_operation_ = time_operation[0]

        if first_operation_queue == 'transmit from GEO to LEO':
            # Index of main LEO satellite which receives the weights from GEO
            A = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation_, idx_orbit)    # The information of one orbit
            if A != None:
                index, time_now = A
                #print(f"AAAAindex = {index}")
                return index, time_now



        elif first_operation_queue == 'scatter the weights':    # From source to other satellites
            
            if time_now != None:
                 #time_for_scattering_the_weights = time_operation[1] * int(np.ceil(len(vector_time_start) / 2))
                 time_for_scattering_the_weights = np.zeros(len(vector_time_start))
                 time_scatter_the_weights = np.zeros(len(vector_time_start))
                 time_now1 = []
                 for sat_idx in range(len(vector_time_start)):
                     #print(f"index = {index}")
                     time_for_scattering_the_weights[sat_idx] = time_operation[1] * min(abs(sat_idx-index[idx_orbit][0]), abs((len(vector_time_start[sat_idx])- abs(sat_idx-index[idx_orbit][0]))))
                     #print(f"min(abs(sat_idx-index), abs((len(vector_time_start)- abs(sat_idx-index))) = {abs(sat_idx-index[0])}")
                     #print(f"0000 time_for_scattering_the_weights[sat_idx] = {time_for_scattering_the_weights[sat_idx]}")
                     time_scatter_the_weights[sat_idx] = time_now[-1] + time_for_scattering_the_weights[sat_idx]                     
                     time_now1.append(time_scatter_the_weights[sat_idx])
                                     
                 
                 time_now.append(time_now1)
                 return index, time_now

        elif first_operation_queue == 'Learning':
            
            time_learning = [0 for i in range(len(vector_time_start))]
            #print(f"$$ time_now = {time_now}")
            #print(f"## time_now[-1] = {time_now[-1]}")
            #print(f"## time_operation[2] = {time_operation[2]}")
            time_now1 = []
            #print(f"000 time_now = {time_now}")
            if time_now != None:
                for idx in range(len(vector_time_start)):
                    time_learning[idx] = time_now[-1][idx] + time_operation[2][idx]
                #print(f"###$$ time_learning = {time_learning}")                    
                    if time_learning[idx] <= max_value:
                           time_now1.append(time_learning)
                return index, time_now

        elif first_operation_queue == 'collecting weights in LEO orbit':
                
                time_for_scattering_the_weights = []
                #print(f"555 time_now = {time_now}")
                for kdx in range(len(vector_time_start)):
                    satellite_com_selected_sat = []
                    for j_sat in range(len(vector_time_start)):
                          satellite_com_selected_sat.append(time_now[-1][kdx] + time_operation[1] * min(abs(kdx-j_sat), abs((len(vector_time_start)- abs(kdx-j_sat)))))
                    time_for_scattering_the_weights.append(max(satellite_com_selected_sat))    
                #print(f"666 time_for_scattering_the_weights = {time_for_scattering_the_weights}")
                time_now1 = copy.deepcopy(time_now)
                times = []
                #print(f"666 time_now = {time_now}")
                for hdx in range(len(vector_time_start)):
                    time_operation_ = time_operation[4]
                    time_now.append(time_for_scattering_the_weights[hdx])
                    #print(f"1111time_for_scattering_the_weights[hdx] = {time_for_scattering_the_weights[hdx]}")
                    B = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation_, idx_orbit)
                    #print(f"444 B = {B}")
                    if B  != None:
                      index, A = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation_, idx_orbit)
                      print(f"index = {index}")
                      print(f"A = {A}")
                      if A != None:
                       times.append(A[-1])
                       #print(A)
                      selected_time = min(times)    
                      #print(f"333selected_timetimes = {selected_time}")
                      if time_now != None:
                  
                            time_scatter_the_weights = selected_time 
                            #print(f"222 time_scatter_the_weights = {time_scatter_the_weights}")
                            if time_scatter_the_weights <= max_value:
                                time_now.append(time_scatter_the_weights)
                                return index, time_now

        elif first_operation_queue == 'transmit from LEO to GEO': # From one of the LEOs to PS

            # Index of main one LEO satellite or two LEO satellites which the collecting the data are started from them
            #print(f"time_now = {time_now}")
            
            if time_now != None:
               a1 = len(time_now)
            time_operation_ = time_operation[4]
            A = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation_, idx_orbit)
            if A != None:
                index, time_now = A
            if time_now != None:
               if len(time_now)>a1:
                  return index, time_now

        elif first_operation_queue == 'aggregate':
            #print(f"time_now = {time_now}")
            #index = 1000
            vector_time_aggregate = []
            for idx_orbit in range(len(time_now)):
                if time_now[idx_orbit] != None:
                    vector_time_aggregate.append(time_now[idx_orbit][-1])



            if len(vector_time_aggregate) == len(time_now):
                check_all_satellite_weights_to_PS = 0
                for idx_orbit in range(len(time_now)):
                    if vector_time_aggregate[idx_orbit] <= max_value[idx_orbit]:
                        check_all_satellite_weights_to_PS += 1 #Check if that all satellite gave the weights for aggregate

                if   check_all_satellite_weights_to_PS ==  len(time_now):
                        time_aggregate = max(vector_time_aggregate)
                        time_aggregate = time_aggregate
                        #+ time_operation[0][5]
                        TIME_AGGREGATE.append(time_aggregate)
                        for idx_orbit in range(len(time_now)):
                           time_now[idx_orbit].append(time_aggregate)
                return index, time_now

            elif len(vector_time_aggregate) != len(time_now):
                return None

    def index_satellites_for_receive_from_LEO(self, time_now_temp, vector_time_start, vector_time_end, time_operation_, idx_orbit):

     # for loop in order to derive the visiting satellite and its time
     x1 = 0
     for i in range(len(vector_time_start)):
         if vector_time_start[i] == []:  x1 += 1
     #print('vector_time_start before s = ' + str(vector_time_start))
     condition_vector_start = x1 != len(vector_time_start)

     if condition_vector_start:
        return self.timing_one_orbit(time_now_temp, vector_time_start, vector_time_end, time_operation_, idx_orbit)
     else:
        return None

    def timing_one_orbit(self, time_now_temp, vector_time_start, vector_time_end, time_operation_, idx_orbit):
    # This function selects the index of satellite in one orbit and its time
        time_startt = time.time()
        time_satellites_start = [1e10 for i in range(len(vector_time_start))]   # Adding the applicaple time for satellites in order to select one of them
        time_satellites_end = [0 for i in range(len(vector_time_start))]
            # In the start point
        if time_now_temp == []:
                temp_min = []
                for i in range(len(vector_time_start)):
                    temp_min.append(vector_time_start[i][0])
                time_now_temp.append(min(temp_min))
        vector_control = (time_now_temp[-1], time_now_temp[-1] + time_operation_)
        for idx_satellite in range(len(vector_time_start)):



         if vector_time_start[idx_satellite] != []:
            if vector_control[-1] <= vector_time_end[idx_satellite][-1]:

                if  vector_time_start[idx_satellite] != []:

                       A1 = (vector_time_start[idx_satellite] != [] and vector_control[0] >= vector_time_end[idx_satellite][0])
                       A2 = (vector_time_start[idx_satellite] != [] and ((vector_control[-1]) > (vector_time_end[idx_satellite][0])))
                       A3 = (vector_time_start[idx_satellite] != [] and ((vector_control[-1] - vector_control[0] ) > (vector_time_end[idx_satellite][0] - vector_time_start[idx_satellite][0])))
                       while_condition = A1 or A2 or A3



                       while while_condition == True:
                           del(vector_time_start[idx_satellite][0], vector_time_end[idx_satellite][0])
                           #print('time_start_vector =' + str(vector_time_start[idx_satellite]))
                           #print('NNNNNNNNNNNNN@@@@@@time_start_vector =' + str(vector_time_start))
                           A1 = (vector_time_start[idx_satellite] != [] and vector_control[0] > vector_time_end[idx_satellite][0])
                           A2 = (vector_time_start[idx_satellite] != [] and ((vector_control[-1]) > (vector_time_end[idx_satellite][0])))
                           A3 = (vector_time_start[idx_satellite] != [] and ((vector_control[-1] - vector_control[0] ) > (vector_time_end[idx_satellite][0] - vector_time_start[idx_satellite][0])))
                           #print(A1,A2,A3)
                           while_condition = A1 or A2 or A3



                if  vector_time_start[idx_satellite] != []:
                     if vector_control[0] <= vector_time_start[idx_satellite][0] and  vector_control[0] <= vector_time_end[idx_satellite][-1]:
                             time_now_temp.append(vector_time_start[idx_satellite][0] + ((vector_control[-1] - vector_control[0])))
                     elif vector_control[0] >= vector_time_start[idx_satellite][0] and vector_control[-1] <= vector_time_end[idx_satellite][0] and vector_control[0] <= vector_time_end[idx_satellite][-1]:
                             time_now_temp.append(vector_control[-1])
                     time_satellites_start[idx_satellite] = time_now_temp[-1]
                     time_satellites_end[idx_satellite] = vector_time_end[idx_satellite][0]


        ### Which satellite has the been selected as the one who will receive the weights
        #print(f"time_satellites_start = {time_satellites_start}")
        #print(f"time_satellites_end = {time_satellites_end}")
        min_time = min(time_satellites_start)
        #max_time_end = max(time_satellites_end)
        max_time_duration = [0 for i in range(len(vector_time_start))]
        max_time_duration_and_min_start = [0 for i in range(len(vector_time_start))]
        for idx_sat in range(len(vector_time_start)):
            max_time_duration[idx_sat] = time_satellites_end[idx_sat] - time_satellites_start[idx_sat]
            if time_satellites_start[idx_sat] == min_time:
                max_time_duration_and_min_start[idx_sat] = max_time_duration[idx_sat]

        selected = max(max_time_duration_and_min_start)
        #print(f"max_time_duration_and_min_start = {max_time_duration_and_min_start}")
        #print(f"selected = {selected}")

        if min_time != 1e10:

           #index_min_time.append(time_satellites_start.index(min_time))
           index_min_time[idx_orbit].append(max_time_duration.index(selected))
           #print(f"index_min_time = {index_min_time}")
           time_now_temp.append(min_time)
           duration = time.time() - time_startt
           #print(f"duration = {duration}")
           return  index_min_time, time_now_temp

        else: # If all have the maximum value equal with 1e10
            return None


    def Rate(self, f_c, d, k, T, B, p_t, g_t, g_r, c):
                    loss = (4 * np.pi * f_c * d / c)**2
                    p_n = k * T * B # Noise power
                    #print(f"loss = {loss}")

                    snr = (p_t * g_t * g_r) / (p_n * loss)
                    rate = B * np.log2(1 + snr)
                    #print(f"rate = {rate}")
                    return rate

    def end_to_end_transmission_time(self, distance, c , Rate, D):
            c = 3e8
            t_propagation = distance / c   # Seconds
            t_transmission = D / Rate
            t_end_to_end = t_propagation + t_transmission
            return t_end_to_end


######################  Parameters #########################
num_satellite = 40
num_planes = 5
parameter_server = 'BR-GS' # NP-GS BR-GS  MEO-GS
inc = 60

Data_type = 'CIFAR10'  # CIFAR10   MNIST
if Data_type == 'MNIST':
       time_learning = 60
       Model_bits = 7850 * 8 * 4  # Model parameters size: 7850 * 8 * 4 for MNIST
elif Data_type == 'CIFAR10':
       time_learning = 480
       Model_bits = 122570 * 8 * 4 # Model parameters size: 122570 * 8 * 4 for MNIST


time_operations_aggregation = 0.00015
time_learning = [[1,1,1,1], [2,1,1,1]]
p_t = 10  # Watt, 40 dBm
g_t = 1633   # 6.99 dBi
g_r = 1633
Boltz_fix = 1.380649 * 1e-23    # Boltzman number, Joule / K
T = 354 # Kelvin
B = 500 * 1e6    # Hz
f_c = 20 * 1e9 #Hz
c = 3 * 1e8 # meter/second
'''
p_t = 10  # Watt, 40 dBm
g_t = 5   # 6.99 dBi
g_r = 5
Boltz_fix = 1.380649 * 1e-23    # Boltzman number, Joule / K
T = 354 # Kelvin
B = 20 * 1e6    # Hz
f_c = 2.4 * 1e9 #Hz
c = 3 * 1e8 # meter/second

'''
time_calculate_the_sink_node = 0.00001
time_aggregation_ISL = 0.00001
################################################################

FL = FEDISL(num_satellite, num_planes, parameter_server, time_calculate_the_sink_node)
distance_GS_LEOs1, vector_time_start, vector_time_end, distance_LEO_LEO = vistings(num_satellite, num_planes, parameter_server, inc)


distance_GS_LEOs = copy.deepcopy(distance_GS_LEOs1)
for i in range(len(distance_GS_LEOs1)):
    for k in range(len(distance_GS_LEOs1[i])):
     distance_GS_LEOs[i][k] =  [g * (1e3) for g in distance_GS_LEOs1[i][k]]


#print(f"distance_GS_LEOs = {distance_GS_LEOs}")
#vector_time_start = [[[2,4,12],   [28]],   [[5,10,12], [20]]]
#vector_time_end =   [[[3,10,13], [37]], [[9,11,17], [60]]]
vector_time_start = [[[2,4,12], [1], [1], [1]], [[2,4,12], [12], [1], [1]]] # Threshold = 28
vector_time_end = [[[3,11,17], [17], [17], [17]], [[3,11,17], [17], [17], [17]]]
#print(f"vector_time_start = {vector_time_start}")
#print(f"@@@@ vector_time_start = {vector_time_start[3]}")
#print(f"vector_time_end = {vector_time_end}")
#print(f"distance_LEO_LEO = {distance_LEO_LEO}")

max_end_value = [[] for idx_orbit in range(len(vector_time_start))]
max_value = [[] for idx_orbit in range(len(vector_time_start))]
start_vector_orbit = [[] for idx_orbit in range(len(vector_time_start))]
start_value = [0 for idx_orbit in range(len(vector_time_start))]
for idx_orbit in range(len(vector_time_start)):
     for idx_satellite in range(len(vector_time_start[idx_orbit])):
        max_end_value[idx_orbit].append(vector_time_end[idx_orbit][idx_satellite][-1])
        start_vector_orbit[idx_orbit].append(vector_time_start[idx_orbit][idx_satellite][0])
     max_value[idx_orbit] = max(max_end_value[idx_orbit])
     #print('max_value = ' + str(max_value))
     start_value[idx_orbit] = min(start_vector_orbit[idx_orbit])

#start_point_value = min(start_value)
start_point_value = 0
time_operation_all_orbits = []
queue_operation = []

max_distance_each_satellite = [[0]*(len(vector_time_start[0])) for idx_orbit in range(len(vector_time_start))]
for idx_orbit in range(len(vector_time_start)):
     for idx_satellite in range(len(vector_time_start[0])):
        max_distance_each_satellite[idx_orbit][idx_satellite] = max(distance_GS_LEOs[idx_orbit][idx_satellite])



rate = copy.deepcopy(max_distance_each_satellite)
for idx_orbit in range(len(max_distance_each_satellite)):
    for idx_sat in range(len(max_distance_each_satellite[0])):
        rate[idx_orbit][idx_sat] = FL.Rate(f_c, max_distance_each_satellite[idx_orbit][idx_sat], Boltz_fix, T, B, p_t, g_t, g_r, c)

num_planes = len(vector_time_start)
min_rate_LEO_GEO = [0 for i in range(num_planes)]
for i in range(num_planes):
    min_rate_LEO_GEO[i] = min(rate[i])

max_distance_LEO_GEO = []
for idx_orbit in range(num_planes):
    max_distance_LEO_GEO.append(max(max_distance_each_satellite[idx_orbit]))
##################################################################################################################################
#print(f"$$$ max_distance_LEO_GEO = {max_distance_LEO_GEO}")
################################################ rate for LEO LEO ################################################################

min_rate_LEO_LEO = [0 for i in range(num_planes)]
for idx_orbit in range(num_planes):
    min_rate_LEO_LEO[idx_orbit] = FL.Rate(f_c, distance_LEO_LEO[idx_orbit], Boltz_fix, T, B, p_t, g_t, g_r, c)
##################################################################################################################################
#print(f"$$$ distance_LEO_LEO = {distance_LEO_LEO}")


time_operation_all_orbits = []
for idx_orbit in range(len(vector_time_start)):
   time_operations_transmission_from_GEO_to_LEO = FL.end_to_end_transmission_time(max_distance_LEO_GEO[idx_orbit], c, min_rate_LEO_GEO[idx_orbit], Model_bits)
   #print(f"*****************time_operations_transmission_from_GEO_to_LEO = {time_operations_transmission_from_GEO_to_LEO}")
   time_operations_transmission_from_LEO_to_LEO = FL.end_to_end_transmission_time(distance_LEO_LEO[idx_orbit], c, min_rate_LEO_LEO[idx_orbit], Model_bits)
   time_operation_all_orbits.append([time_operations_transmission_from_GEO_to_LEO, time_operations_transmission_from_LEO_to_LEO,
                                     time_learning[idx_orbit], time_operations_transmission_from_LEO_to_LEO,
                                     time_operations_transmission_from_GEO_to_LEO, time_operations_aggregation])
   queue_operation.append(['transmit from GEO to LEO', 'scatter the weights', 'Learning', 'collecting weights in LEO orbit'
                          , 'transmit from LEO to GEO', 'aggregate'])


#print('time_operation_all_orbits = ' + str(time_operation_all_orbits))
time_now = [[] for idx_orbit in range(len(vector_time_start))]
index_min_time = [[] for idx_orbit in range(len(vector_time_start))]
#print(f"index_min_time  = {index_min_time}")
#TIME_AGGREGATE = [start_point_value]
TIME_AGGREGATE = [0]

A = 0
F = True

index = 1000
while F == True:
       
       index, A = FL.queue(index, TIME_AGGREGATE, queue_operation, time_operation_all_orbits, time_now, vector_time_start, vector_time_end, max_value)
       print(f" %%% index = {index}")
       F = all([A[i] != None for i in range(len(A))])
       if F == True:
         time_now  = A
         print(f"**********  TIME_AGGREGATE = {TIME_AGGREGATE}")

#print('TIME_AGGREGATE after simulation = ' + str(TIME_AGGREGATE))
print(TIME_AGGREGATE)
#print(f"time_operation_all_orbits = {time_operation_all_orbits}")

if Data_type == 'MNIST':
        if parameter_server == 'BR-GS':
           with open('./Results/Time_FedISL_Sync_MNIST_Bremen.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_Bremen =  {TIME_AGGREGATE}')

        elif parameter_server == 'NP-GS':
           with open('./Results/Time_FedISL_Sync_MNIST_NP.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_NP =  {TIME_AGGREGATE}')

        elif parameter_server == 'MEO-GS':
           with open('./Results/Time_FedISL_Sync_MNIST_MEO.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_MEO =  {TIME_AGGREGATE}')
elif Data_type == 'CIFAR10':
        if parameter_server == 'BR-GS':
           with open('./Results/Time_FedISL_Sync_CIFAR10_Bremen.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_Bremen =  {TIME_AGGREGATE}')

        elif parameter_server == 'NP-GS':
           with open('./Results/Time_FedISL_Sync_CIFAR10_NP.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_NP =  {TIME_AGGREGATE}')

        elif parameter_server == 'MEO-GS':
           with open('./Results/Time_FedISL_Sync_CIFAR10_MEO.py', 'w') as f:
              f.write(f'Time_FedISL_Sync_MEO =  {TIME_AGGREGATE}')
