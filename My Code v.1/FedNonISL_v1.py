# FEDNonISL multiple orbits
# Written by Nasrin Razmi
# 13.08.2021
# In the start point, at first the parameter server sends the initial weights to the satellites when it sees each one


from import_satellite_visitings_time_v8 import vistings
import copy
import numpy as np
import matplotlib.pyplot as plt

class FEDNonISL:

    def __init__(self, num_satellite, num_planes, parameter_server):
        self.num_satellite = num_satellite
        self.num_planes = num_planes
        self.parameter_server = parameter_server

    def queue(self, vector_aggregate, cnt_aggregate, queue_operations, time_operations, results, vector_time_append, vector_time_start, vector_time_end):

      all_time = []	# The latest time until now of all satellites will added to this list
      index_all_time = []	# This will add the index of satellites
      for i in range(len(vector_time_append)):
          all_time.append(vector_time_append[i][-1])
          index_all_time.append(i)


      all_time2 = copy.copy(all_time)

      # For controlling that satellites are in the aggregate state?
      queue_first_state = []
      for i in range(len(vector_time_append)):
          queue_first_state.append(queue_operations[i][0]) # Append the state of satellites

          if queue_first_state[i] == 'aggregate':
            all_time[i] = 1000000

            # If all satellites are in the new aggregate state, only add the max time of satellites to the vector_time_append
            if all(x == 1000000 for x in all_time) and all(x == max(cnt_aggregate) for x in cnt_aggregate):
              time_aggregate = max(all_time2)   # This derives the maximum time of the satellites for aggregation
              vector_aggregate.append(time_aggregate +  time_operations[i][3])
              #print('vector_aggregate = ' + str(vector_aggregate))
              #print('vector_time_start = ' + str(vector_time_start))

              for i  in range(len(vector_time_append)):
                  vector_time_append[i].append(time_aggregate)




      # This part helos to proceed the program when one of satellites has more visiting.
      if all(x == 1000000 for x in all_time):
               #print('cnt_aggregate = ' + str(cnt_aggregate))
               all_time = []
               index_all_time = []
               for i in range(len(vector_time_append)):
                   all_time.append(vector_time_append[i][-1])
                   index_all_time.append(i)



      min_time = min(all_time)
      index_min_time = all_time.index(min_time)
      x = queue_first_state[index_min_time]
      #print(x, vector_time_append)
      A = self.dispatcher(x, cnt_aggregate[index_min_time], time_operations[index_min_time], results[index_min_time], vector_time_append[index_min_time], vector_time_start[index_min_time], vector_time_end[index_min_time])
      #print(A)
      if A != None:
          cnt_aggregate[index_min_time], vector_time_append[index_min_time], value, vector_time_start[index_min_time], vector_time_end[index_min_time] = A
          results[index_min_time].append(value)
          #print(vector_time_append)
          temp_queue = copy.copy(queue_operations[index_min_time])
          del(temp_queue[0])
          temp_queue.append(x)
          queue_operations[index_min_time] = temp_queue
          #print('*************vector_time_append = ' + str(vector_time_append))
          return vector_aggregate, cnt_aggregate, queue_operations, results, vector_time_append, vector_time_start, vector_time_end

    # Function dispatcher for handling the tasks
    def dispatcher(self, request_type, cnt_aggregate, time_operations, results, vector_time, vector_time_start, vector_time_end):

      if request_type == 'transmit from GEO to LEO':
          time_now = vector_time[-1]
          time_now_temp = time_now + time_operations[0]  # The process for 'sum' takes 2 seconds
          vector_control = [time_now, time_now_temp]
          #vector_time.append(time_now_temp)
          A = self.calculate_time(vector_control, vector_time, vector_time_start, vector_time_end)
          if A != None:
            vector_time, vector_time_start, vector_time_end = A
            return cnt_aggregate+1, vector_time, self.transmit_from_GEO_LEO(results), vector_time_start, vector_time_end


      elif request_type == 'Learning':
          time_now = vector_time[-1]
          time_now_temp = time_now + time_operations[1]
          if vector_time_end != []:
           if time_now_temp <= vector_time_end[-1]:
             vector_time.append(time_now_temp)
             return cnt_aggregate+1, vector_time, self.Learning(results), vector_time_start, vector_time_end

      elif request_type == 'transmit from LEO to GEO':
          time_now = vector_time[-1]
          time_now_temp = time_now + time_operations[2]  # The process for 'sum' takes 2 seconds
          vector_control = [time_now, time_now_temp]
          #vector_time.append(time_now_temp)
          A = self.calculate_time(vector_control, vector_time, vector_time_start, vector_time_end)
          if A != None:
            vector_time, vector_time_start, vector_time_end = A
            #print('&&&&&&&&&&vector_time'+str(vector_time))
            return cnt_aggregate+1, vector_time, self.transmit_from_LEO_GEO(results), vector_time_start, vector_time_end

      elif request_type == 'aggregate':
          #print('time_now_temp inside aggregate = ' + str(vector_time))
          time_now = vector_time[-1]
          time_now_temp = time_now
          if vector_time_end != []:
            if time_now_temp <= vector_time_end[-1]:
               vector_time.append(time_now_temp)
               return cnt_aggregate+1, vector_time, self.aggregate(results), vector_time_start, vector_time_end

    def transmit_from_GEO_LEO(self, results):
       return results

    def Learning(self, results):
       return results

    def transmit_from_LEO_GEO(self, results):
        return results

    def aggregate(self, results):
        return results

    def calculate_time(self, vector_control, time_now_temp, time_start_vector, time_end_vector):
            #print('vector_control, 117' + str(vector_control))
            #print('time_now_temp, 117' + str(time_now_temp))
            if vector_control[-1] <= time_end_vector[-1]:

                if  time_start_vector != []:

                       A1 = (time_start_vector != [] and vector_control[0] > time_end_vector[0])
                       A2 = (time_start_vector != [] and ((vector_control[-1]) > (time_end_vector[0])))
                       A3 = (time_start_vector != [] and ((vector_control[-1] - vector_control[0] ) > (time_end_vector[0] - time_start_vector[0])))
                       while_condition = A1 or A2 or A3

                       #print(' while_condition = ' + str(while_condition))

                       while while_condition == True:
                           del(time_start_vector[0], time_end_vector[0])
                           #print('time_start_vector =' + str(time_start_vector))

                           A1 = (time_start_vector != [] and vector_control[0] > time_end_vector[0])
                           A2 = (time_start_vector != [] and ((vector_control[-1]) > (time_end_vector[0])))
                           A3 = (time_start_vector != [] and ((vector_control[-1] - vector_control[0] ) > (time_end_vector[0] - time_start_vector[0])))

                           while_condition = A1 or A2 or A3


                if  time_start_vector != []:
                     #print('@@@@@@time_start_vector =' + str(time_start_vector))
                     if vector_control[0] <= time_start_vector[0] and  vector_control[0] <= time_end_vector[-1]:
                             time_now_temp.append(time_start_vector[0] + ((vector_control[-1] - vector_control[0])))
                     elif vector_control[0] >= time_start_vector[0] and vector_control[-1] <= time_end_vector[0] and vector_control[0] <= time_end_vector[-1]:
                             time_now_temp.append(vector_control[-1])

                     #print('#######################time_now_temp' + str(time_now_temp))
                     return  time_now_temp, time_start_vector, time_end_vector
            else:
                if  time_start_vector != []:
                   del(time_start_vector[0], time_end_vector[0])
                return  None


    def Rate(self, f_c, d, k, T, B, p_t, g_t, g_r, c):
                    loss = (4 * np.pi * f_c * d / c)**2
                    p_n = k * T * B # Noise power
                    #print(f"loss = {loss}")

                    snr = (p_t * g_t * g_r) / (p_n * loss)
                    rate = B * np.log2(1 + snr)
                    return rate

    def end_to_end_transmission_time(self, distance, c , Rate, D):
            c = 3e8
            t_propagation = distance / c   # Seconds
            t_transmission = D / Rate
            t_end_to_end = t_propagation + t_transmission
            #print(f"t_end_to_end = {t_end_to_end}")
            return t_end_to_end


######################  Parameters #########################
num_satellite = 40
num_planes = 5
parameter_server = 'BR-GS' # BR-GS  MEO-GS
inc = 85

Data_type = 'MNIST'  # CIFAR10   MNIST
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
################################################################


FL = FEDNonISL(num_satellite, num_planes, parameter_server)
distance_GS_LEOs1, vector_time_start, vector_time_end, distance_LEO_LEO = vistings(num_satellite, num_planes, parameter_server, inc)


distance_GS_LEOs = copy.deepcopy(distance_GS_LEOs1)
for i in range(len(distance_GS_LEOs1)):
    for k in range(len(distance_GS_LEOs1[i])):
     distance_GS_LEOs[i][k] =  [g * (1e3) for g in distance_GS_LEOs1[i][k]]

vector_time_start_temp = copy.deepcopy(vector_time_start)
vector_time_end_temp = copy.deepcopy(vector_time_end)




vector_time_start = []
vector_time_end = []
for i in range(len(vector_time_start_temp)):
	for j in range(len(vector_time_start_temp[i])):
	  vector_time_start.append(vector_time_start_temp[i][j])
	  vector_time_end.append(vector_time_end_temp[i][j])

print('vector_time_start = ' + str(vector_time_start))
print('vector_time_end = ' + str(vector_time_end))
#vector_time_start = [[[2,4,12], [1], [16], [45]], [[2,4,12], [20], [16], [14]]] # Threshold = 28
#vector_time_end = [[[3,11,17], [12], [17], [67]], [[3,11,17], [60], [17], [14]]]
#vector_time_start = [[1,19],[5,20]]
#vector_time_end = [[7,30],[17,35]]

############################################### Rate of GEO-LEO for each orbit #####################################################


#print(f"len(distance_GS_LEOs) = {np.size(distance_GS_LEOs)}")
#print(distance_GS_LEOs)

matrix_to_list_distance = []
for i_idx in range(num_planes):
    for k_idx in range(len(distance_GS_LEOs[i])):
        matrix_to_list_distance.append(distance_GS_LEOs[i_idx][k_idx])

#print(matrix_to_list_distance)

max_distance_LEO_GEO = [0 for idx_sat in range(num_satellite)]
for idx_satellite in range(num_satellite):
        max_distance_LEO_GEO[idx_satellite] = max(matrix_to_list_distance[idx_satellite])

#print(f"max_distance_each_satellite = {max_distance_LEO_GEO}")

#rate = copy.deepcopy(max_distance_each_satellite)
#for idx_orbit in range(len(max_distance_each_satellite)):
#    for idx_sat in range(len(max_distance_each_satellite[0])):
#        rate[idx_orbit][idx_sat] = FL.Rate(f_c, max_distance_each_satellite[idx_orbit][idx_sat], k, T, B, p_t, g_t, g_r, c)


min_rate_LEO_GEO = [0 for i_id in range(num_satellite)]
for idx_sat in range(num_satellite):
    min_rate_LEO_GEO[idx_sat] = FL.Rate(f_c, max_distance_LEO_GEO[idx_sat], Boltz_fix, T, B, p_t, g_t, g_r, c)


##################################################################################################################################
################################################ rate for LEO LEO ################################################################

min_rate_LEO_LEO = [0 for i_sat in range(num_satellite)]
for idx_satellite in range(num_satellite):
    min_rate_LEO_LEO[idx_satellite] = FL.Rate(f_c, distance_LEO_LEO[0], Boltz_fix, T, B, p_t, g_t, g_r, c)

#print(f"min_rate_LEO_LEO = {min_rate_LEO_LEO}")

##################################################################################################################################
time_operations = []
queue_operations = []


time_operations = []
for idx_satellite in range(num_satellite):
   #print(min_rate_LEO_GEO[idx_orbit], idx_orbit)
   time_operations_transmission_from_GEO_to_LEO = FL.end_to_end_transmission_time(max_distance_LEO_GEO[idx_satellite], c, min_rate_LEO_GEO[idx_satellite], Model_bits)
   time_operations_transmission_from_LEO_to_LEO = FL.end_to_end_transmission_time(distance_LEO_LEO[0], c, min_rate_LEO_LEO[idx_satellite], Model_bits)
   time_operations.append([time_operations_transmission_from_GEO_to_LEO, time_learning,
                                     time_operations_transmission_from_GEO_to_LEO, time_operations_aggregation])

   queue_operations.append(['transmit from GEO to LEO', 'Learning', 'transmit from LEO to GEO', 'aggregate'])


#print(queue_operations)
print(f"time_operations = {time_operations}")

#print(time_operations)

#queue_operations = []
vector_time_append = []
results = []
cnt_aggregate = []
for i in range(len(vector_time_start)):
    #queue_operations.append(queue)
    vector_time_append.append([vector_time_start[i][0]])
    results.append([0])
    cnt_aggregate.append(0)


A = 0
vector_aggregate = [0]
vector_start_condition = [[] for i in range(len(vector_time_start))]
while vector_time_start != vector_start_condition and A != None:
  A = FL.queue(vector_aggregate, cnt_aggregate, queue_operations, time_operations, results, vector_time_append, vector_time_start, vector_time_end)
  if A != None:
       vector_aggregate, cnt_aggregate,  request_numbers, results, vector_time_append, vector_time_start, vector_time_end  = A

print('aggregate = ' + str(vector_aggregate))
print('aggregate = ' + str(len(vector_aggregate)))


if Data_type == 'MNIST':
        
        if inc>60:
          if parameter_server == 'BR-GS':
             with open('./Results/Time_FedNonISL_Sync_MNIST_Bremen_walker_star.py', 'w') as f:
                f.write(f"Time_FedNonISL_Sync_Bremen =  {np.array(vector_aggregate).tolist()}")

        elif parameter_server == 'BR-GS':
           with open('./Results/Time_FedNonISL_Sync_MNIST_Bremen.py', 'w') as f:
              f.write(f"Time_FedNonISL_Sync_Bremen =  {np.array(vector_aggregate).tolist()}")

        elif  parameter_server == 'NP-GS':
            with open('./Results/Time_FedNonISL_Sync_MNIST_NP.py', 'w') as f:
                  f.write(f"Time_FedNonISL_Sync_NP =  {np.array(vector_aggregate).tolist()}")


        elif  parameter_server == 'MEO-GS':
            with open('./Results/Time_FedNonISL_Sync_MNIST_MEO.py', 'w') as f:
                  f.write(f'Time_FedNonISL_Sync_MEO =  {np.array(vector_aggregate).tolist()} \n')

if Data_type == 'CIFAR10':
        if inc>60:
          if parameter_server == 'BR-GS':
             with open('./Results/Time_FedNonISL_Sync_CIFAR10_Bremen_walker_star.py', 'w') as f:
                f.write(f"Time_FedNonISL_Sync_Bremen =  {np.array(vector_aggregate).tolist()}")

        elif parameter_server == 'BR-GS':
           with open('./Results/Time_FedNonISL_Sync_CIFAR10_Bremen.py', 'w') as f:
              f.write(f"Time_FedNonISL_Sync_Bremen =  {np.array(vector_aggregate).tolist()}")


        elif  parameter_server == 'NP-GS':
            with open('./Results/Time_FedNonISL_Sync_CIFAR10_NP.py', 'w') as f:
                  f.write(f"Time_FedNonISL_Sync_NP =  {np.array(vector_aggregate).tolist()}")


        elif  parameter_server == 'MEO-GS':
            with open('./Results/Time_FedNonISL_Sync_CIFAR10_MEO.py', 'w') as f:
                  f.write(f'Time_FedNonISL_Sync_MEO =  {np.array(vector_aggregate).tolist()} \n')
