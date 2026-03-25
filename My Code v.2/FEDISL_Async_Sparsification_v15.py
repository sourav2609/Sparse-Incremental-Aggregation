# Written by Nasrin Razmi

# FEDISL-Async with the accurate implementation time
# Implementation FL with intra-link satellites
# In the start point, at first the parameter server sends the initial weights to the satellites when it sees each one

# Description: For each orbit 6 operation in a queue is considered: ['transmit from GEO to LEO', 'scatter the weights', 'Learning', 'collecting weights in LEO orbit', 'transmit from LEO to GEO', 'aggregate']

from import_satellite_visitings_time_v7 import vistings
import numpy as np
import copy
import time
import sys
np.set_printoptions(threshold=sys.maxsize)
class FEDISL_Async:
#, time_aggregation_ISL
    def __init__(self, num_satellite, num_planes, parameter_server, time_calculate_the_sink_node):
        self.num_satellite = num_satellite
        self.num_planes = num_planes
        self.parameter_server = parameter_server
        self.time_calculate_the_sink_node = time_calculate_the_sink_node
        self.time_aggregation_ISL = time_aggregation_ISL

    def queue(self, TIME_AGGREGATE,  INDEX_Aggregate, queue_operation, time_operation, time_now_vector, vector_time_start, vector_time_end, max_value, check_the_first_visit):

      # Only bring the first operation to do and then put in at the end of queue_operations
     first_operation_queue = queue_operation[0][0]


     for idx_orbit in range(len(vector_time_start)):

      if vector_time_start[idx_orbit] != None:
       first_operation_queue = queue_operation[idx_orbit][0]

       A = self.dispatcher(first_operation_queue, time_now_vector[idx_orbit], time_operation[idx_orbit], vector_time_start[idx_orbit], vector_time_end[idx_orbit], max_value[idx_orbit], idx_orbit, INDEX_Aggregate, check_the_first_visit)
       time_now_vector[idx_orbit] = A
       queue_operation[idx_orbit].pop(0)
       queue_operation[idx_orbit].append(first_operation_queue)
     return time_now_vector

    def dispatcher(self, first_operation_queue, time_now, time_operation, vector_time_start, vector_time_end, max_value, idx_orbit,  INDEX_Aggregate, check_the_first_visit):
        #print(f"&&& first_operation_queue = {first_operation_queue}")
        time_operation_ = time_operation[0]


        if first_operation_queue == 'transmit from GEO to LEO':
            # Index of main LEO satellite which receives the weights from GEO
            A = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation[0], idx_orbit)    # The information of one orbit
            if A != None:
                index, time_now = A
                return time_now


        elif first_operation_queue == 'scatter the weights':    # From PS to one of the LEOs
            if time_now != None:
                 time_for_scattering_the_weights = time_operation[1] * int(np.ceil(len(vector_time_start) / 2))
                 #print(f"### time_for_scattering_the_weights = {time_for_scattering_the_weights}")
                 time_scatter_the_weights = time_now[-1] + time_for_scattering_the_weights
                 if time_scatter_the_weights <= max_value:
                   time_now.append(time_scatter_the_weights)
                   return time_now

        elif first_operation_queue == 'Learning':
            if time_now != None:
                time_learning = time_now[-1] + time_operation[2]
                if time_learning <= max_value:
                   time_now.append(time_learning)
                   return time_now

        elif first_operation_queue == 'collecting weights in LEO orbit':

                time_for_scattering_the_weights = time_operation[1] * int(np.ceil(len(vector_time_start) / 2))
                #print(f"### time_for_scattering_the_weights = {time_for_scattering_the_weights}")
                if time_now != None:
                  time_scatter_the_weights = time_now[-1] + time_for_scattering_the_weights
                  if time_scatter_the_weights <= max_value:
                      time_now.append(time_scatter_the_weights)
                      return time_now

        elif first_operation_queue == 'Threshold':
            #print(f"check_the_first_visit = {check_the_first_visit}")
            if time_now != None:
                if check_the_first_visit[idx_orbit] == 0:   # For the first visit
                     time_needed_threshold = time_now[-1]
                else:   # For the next visits, considering the constrainting point: Threshold  or non-visiting
                     time_needed_threshold = time_now[-1] + max((time_operation[4] - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2])), 0)
                     #time_needed_threshold = time_now[-1] + (time_operation[4] - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2]))
                     #print(f"^^^ (time_operation[4] - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2])) = {(time_operation[4] - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2]))}")
                #print(f"YYYYYYtime_threshold = {100000 - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2])}")
                #print((time_operation[4] - (2*time_operation[0] + (2*time_operation[1] * int(np.ceil(len(vector_time_start) / 2))) + time_operation[2])))
                if time_needed_threshold <= max_value:
                   check_the_first_visit[idx_orbit] = 1
                   time_now.append(time_needed_threshold)
                   return time_now

        elif first_operation_queue == 'transmit from LEO to GEO': # From one of the LEOs to PS

            # Index of main one LEO satellite or two LEO satellites which the collecting the data are started from them
            #print(f"time_now = {time_now}")
            if time_now != None:
               a1 = len(time_now)
            time_operation_ = time_operation[4]
            A = self.index_satellites_for_receive_from_LEO(time_now, vector_time_start, vector_time_end, time_operation[0], idx_orbit)
            if A != None:
                index, time_now = A
            if time_now != None:
               if len(time_now)>a1:
                  return time_now

        elif first_operation_queue == 'aggregate':

            vector_time_aggregate = []
            if time_now != None and time_now != []:
                    TIME_AGGREGATE.append(time_now[-1])
                    INDEX_Aggregate.append(idx_orbit)
                    time_now.append(time_now[-1])
            return time_now

    def index_satellites_for_receive_from_LEO(self, time_now_temp, vector_time_start, vector_time_end, time_operation_, idx_orbit):

     # for loop in order to derive the visiting satellite and its time
     x1 = 0
     #print(f"vector_time_start = {vector_time_start}")
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

        #print(f"#### time_now_temp = {time_now_temp}")
        time_satellites_start = [1e10 for i in range(len(vector_time_start))]   # Adding the applicaple time for satellites in order to select one of them
        time_satellites_end = [0 for i in range(len(vector_time_start))]
            # In the start point
        if time_now_temp == []:
                temp_min = []
                for i in range(len(vector_time_start)):
                    temp_min.append(vector_time_start[i][0])
                time_now_temp.append(min(temp_min))
        #print(f"^^^^time_now_temp = {time_now_temp}")
        if time_now_temp != None :
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

               #print(f"duration = {duration}")
               return  index_min_time, time_now_temp

            else: # If all have the maximum value equal with 1e10
                return None

    def Rate(self, f_c, d, k, T, B, p_t, g_t, g_r, c):
                    loss = (4 * np.pi * f_c * d / c)**2
                    print(f"loss = {loss}")
                    p_n = k * T * B # Noise power
                    print(f"p_n = {p_n}")

                    snr = (p_t * g_t * g_r) / (p_n * loss)
                    print(f"snr = {snr}")
                    rate = B * np.log2(1 + snr)
                    print(rate)
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
time_threshold = 10000
################################################################

FL = FEDISL_Async(num_satellite, num_planes, parameter_server,time_calculate_the_sink_node)
#FL = FEDISL(num_satellite, num_planes, parameter_server,time_calculate_the_sink_node, time_aggregation_ISL)

distance_GS_LEOs1, vector_time_start, vector_time_end, distance_LEO_LEO = vistings(num_satellite, num_planes, parameter_server, inc)
#print(f"### vector_time_start  = {vector_time_start }")
#print(f"### vector_time_end  = {vector_time_end }")

distance_GS_LEOs = copy.deepcopy(distance_GS_LEOs1)
for i in range(len(distance_GS_LEOs1)):
    for k in range(len(distance_GS_LEOs1[i])):
     distance_GS_LEOs[i][k] =  [g * (1e3) for g in distance_GS_LEOs1[i][k]]

#vector_time_start = [[[2,4,12], [1], [16], [45]], [[2,4,12], [20], [16], [14], [16],[14], [16], [14]]] # Threshold = 28
#vector_time_end = [[[3,11,17], [12], [17], [67]], [[3,11,17], [60], [17], [14], [16],[14], [16], [14]]]
#vector_time_start = [[[2,4,12], [1], [16], [45]], [[2,4,12], [20], [16], [14]]] # Threshold = 28
#vector_time_end = [[[3,11,17], [12], [17], [67]], [[3,11,17], [60], [17], [14]]]
check_the_first_visit = [0 for idxx in range(len(vector_time_start))]
#vector_time_start = [[[2,4,12], [1], [16], [14], [16],[14], [16], [14]]]
#vector_time_end = [[[3,11,17], [28], [17], [14], [16],[14], [16], [14]]]
#vector_time_start = [[[3,10,12], [4, 13]], [[3]]]
#vector_time_end = [[[4,11,22], [5, 15]], [[4]]]
#print(f"distance_GS_LEOs = {distance_GS_LEOs}")
#print(f"vector_time_start = {vector_time_start[3]}")
#print(f"vector_time_end = {vector_time_end[3]}")
#vector_time_start = [[[7565, 15542, 23756, 67420, 75239, 83268, 91307, 99286, 107352, 151266, 158996, 166994, 175045, 183033, 191045, 235148, 242770, 250722], [6566, 14543, 22659, 66470, 74246, 82260, 90306, 98290, 106322, 150332, 158011, 165987, 174041, 182037, 190033, 234243, 241793, 249718, 257772], [5566, 13546, 21610, 65526, 73256, 81253, 89304, 97293, 105304, 149411, 157030, 164981, 173037, 181041, 189027, 233361, 240822, 248716, 256765], [4565, 12549, 20581, 64593, 72271, 80246, 88301, 96297, 104292, 148506, 156054, 163977, 172031, 180044, 188023, 232527, 239857, 247716, 255758], [3563, 11553, 19563, 63673, 71290, 79241, 87296, 95300, 103286, 147624, 155083, 162975, 171025, 179046, 187023, 195248, 238899, 246720, 254750], [2560, 10556, 18551, 62769, 70314, 78237, 86290, 94303, 102283, 146794, 154118, 161976, 170017, 178047, 186024, 194145, 237947, 245726, 253742], [1555, 9561, 17547, 61891, 69345, 77236, 85285, 93306, 101284, 109502, 153160, 160980, 169010, 177049, 185028, 193095, 237004, 244737, 252735], [550, 8563, 16542, 61068, 68379, 76237, 84278, 92308, 100284, 108403, 152211, 159986, 168003, 176048, 184031, 192064, 236078, 243751, 251729]]]
#vector_time_end = [[[8959, 16814, 24127, 68652, 76631, 84645, 92698, 100619, 108161, 152384, 160378, 168375, 176430, 184403, 192076, 236096, 244126, 252110], [7959, 15850, 23302, 67649, 75634, 83639, 91694, 99643, 107257, 151373, 159382, 167371, 175423, 183418, 191143, 235068, 243129, 251109, 259148], [6957, 14879, 22424, 66643, 74638, 82634, 90689, 98663, 106338, 150355, 158385, 166369, 174415, 182429, 190201, 234022, 242132, 250109, 258140], [5954, 13903, 21520, 65632, 73641, 81631, 89682, 97678, 105405, 149326, 157389, 165368, 173407, 181436, 189251, 232933, 241133, 249110, 257133], [4948, 12923, 20599, 64613, 72645, 80628, 88674, 96688, 104463, 148280, 156391, 164368, 172400, 180440, 188293, 195591, 240133, 248112, 256126], [3941, 11937, 19667, 63585, 71648, 79627, 87667, 95695, 103512, 147188, 155393, 163370, 171392, 179440, 187329, 194775, 239130, 247115, 255120], [2934, 10949, 18731, 62537, 70652, 78628, 86660, 94701, 102554, 109864, 154392, 162372, 170386, 178439, 186359, 193898, 238125, 246119, 254116], [1926, 9956, 17773, 61444, 69653, 77630, 85652, 93700, 101590, 109040, 153391, 161376, 169381, 177435, 185383, 192996, 237115, 245123, 253112]]]

print(f"distance_LEO_LEO = {distance_LEO_LEO}")

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

#print(f"max_distance_each_satellite = {max_distance_each_satellite}")

rate = copy.deepcopy(max_distance_each_satellite)
for idx_orbit in range(len(max_distance_each_satellite)):
    for idx_sat in range(len(max_distance_each_satellite[0])):
        rate[idx_orbit][idx_sat] = FL.Rate(f_c, max_distance_each_satellite[idx_orbit][idx_sat], Boltz_fix, T, B, p_t, g_t, g_r, c)

num_planes = len(vector_time_start)
min_rate_LEO_GEO = [0 for i in range(num_planes)]
for i in range(num_planes):
    min_rate_LEO_GEO[i] = min(rate[i])
#print(f"rate = {rate}")
#print(min_rate_LEO_GEO)

max_distance_LEO_GEO = []
for idx_orbit in range(num_planes):
    max_distance_LEO_GEO.append(max(max_distance_each_satellite[idx_orbit]))
##################################################################################################################################
################################################ rate for LEO LEO ################################################################

min_rate_LEO_LEO = [0 for i in range(num_planes)]
for idx_orbit in range(num_planes):
    min_rate_LEO_LEO[idx_orbit] = FL.Rate(f_c, distance_LEO_LEO[idx_orbit], Boltz_fix, T, B, p_t, g_t, g_r, c)
print(f"min_rate_LEO_LEO = {min_rate_LEO_LEO}")
##################################################################################################################################

time_operation_all_orbits = []
for idx_orbit in range(len(vector_time_start)):
   time_operations_transmission_from_GEO_to_LEO = FL.end_to_end_transmission_time(max_distance_LEO_GEO[idx_orbit], c, min_rate_LEO_GEO[idx_orbit], Model_bits)
   time_operations_transmission_from_LEO_to_GEO = FL.end_to_end_transmission_time(max_distance_LEO_GEO[idx_orbit], c, min_rate_LEO_GEO[idx_orbit], Model_bits)
   #print(f"*****************time_operations_transmission_from_GEO_to_LEO = {time_operations_transmission_from_GEO_to_LEO}")
   time_operations_transmission_from_LEO_to_LEO = FL.end_to_end_transmission_time(distance_LEO_LEO[idx_orbit], c, min_rate_LEO_LEO[idx_orbit], Model_bits)
   time_operation_all_orbits.append([time_operations_transmission_from_GEO_to_LEO, time_operations_transmission_from_LEO_to_LEO,
                                     time_learning, time_operations_transmission_from_LEO_to_LEO, time_threshold,
                                     time_operations_transmission_from_LEO_to_GEO, time_operations_aggregation])
   print(f"111%%% time_operations_transmission_from_GEO_to_LEO = {time_operations_transmission_from_GEO_to_LEO}")
   print(f"222%%% time_operations_transmission_from_LEO_to_LEO = {time_operations_transmission_from_LEO_to_LEO}")
   queue_operation.append(['transmit from GEO to LEO', 'scatter the weights', 'Learning',  'collecting weights in LEO orbit', 'Threshold'
                          , 'transmit from LEO to GEO',  'aggregate'])

print('time_operation_all_orbits = ' + str(time_operation_all_orbits))
time_now = [[] for idx_orbit in range(len(vector_time_start))]
index_min_time = [[] for idx_orbit in range(len(vector_time_start))]
TIME_AGGREGATE = []
INDEX_Aggregate = []

A = 0
F = True
#print(f"queue_operation = {queue_operation}")

while F == True:

       A = FL.queue(TIME_AGGREGATE, INDEX_Aggregate, queue_operation, time_operation_all_orbits, time_now, vector_time_start, vector_time_end, max_value, check_the_first_visit)
       #print(check_the_first_visit)
       F = any([A[i] != None and A[i] != [] for i in range(len(A))])
       #print(f"INDEX_Aggregate = {INDEX_Aggregate}")
       if F == True:
         time_now  = A


print(TIME_AGGREGATE)
indexes = np.argsort(TIME_AGGREGATE)
TIME_AGGREGATE1 = np.array(TIME_AGGREGATE)[np.array(indexes)]
INDEX_Aggregate1 = np.array(INDEX_Aggregate)[np.array(indexes)]
#print('TIME_AGGREGATE_after_simulation = ' + str(TIME_AGGREGATE1.tolist()))
#print('Index_AGGREGATE after simulation = ' + str(INDEX_Aggregate1.tolist()))
#print(len(TIME_AGGREGATE1))
Time_Aggregate_final = [0]
Time_Aggregate_final = np.concatenate((Time_Aggregate_final, TIME_AGGREGATE1))
#print(len(Time_Aggregate_final))
print('TIME_AGGREGATE_after_simulation = ' + str(Time_Aggregate_final.tolist()))
print('Index_AGGREGATE after simulation = ' + str(INDEX_Aggregate1.tolist()))



if Data_type == 'MNIST':
    if parameter_server == 'BR-GS':
       with open('./Results/Time_Index_FedISL_Async_MNIST_Bremen.py', 'w') as f:
          f.write(f'Time_FedISL_Async_Bremen =  {Time_Aggregate_final.tolist()} \n')
          f.write(f'Index_FedISL_Async_Bremen =   {INDEX_Aggregate1.tolist()}')

    elif  parameter_server == 'NP-GS':
        with open('./Results/Time_Index_FedISL_Async_MNIST_NP.py', 'w') as f:
              f.write(f'Time_FedISL_Async_NP =  {Time_Aggregate_final.tolist()} \n')
              f.write(f'Index_FedISL_Async_NP =   {INDEX_Aggregate1.tolist()}')

elif Data_type == 'CIFAR10':
    if parameter_server == 'BR-GS':
       with open('./Results/Time_Index_FedISL_Async_CIFAR10_Bremen.py', 'w') as f:
          f.write(f'Time_FedISL_Async_Bremen =  {Time_Aggregate_final.tolist()} \n')
          f.write(f'Index_FedISL_Async_Bremen =   {INDEX_Aggregate1.tolist()}')

    elif  parameter_server == 'NP-GS':
        with open('./Results/Time_Index_FedISL_Async_CIFAR10_NP.py', 'w') as f:
              f.write(f'Time_FedISL_Async_NP =  {Time_Aggregate_final.tolist()} \n')
              f.write(f'Index_FedISL_Async_NP =   {INDEX_Aggregate1.tolist()}')
