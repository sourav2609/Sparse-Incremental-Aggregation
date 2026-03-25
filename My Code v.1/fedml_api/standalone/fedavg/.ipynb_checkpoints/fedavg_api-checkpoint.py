# Implementing FedAvg with gradients, in this code, at first the received y_i from each orbit are summed and then we derive the sum of sum x_i for each orbit.

import operator
import copy
import logging
import random
from torch import tensor
import numpy as np
import torch
from nltk import flatten
torch.set_printoptions(precision=10)
import wandb
import pandas as pd
import math
import time
torch.set_printoptions(profile="full")
from fedml_api.standalone.fedavg.client import Client
from AMP_FL import AMP_Implement_Sat
from AMP_FL import AMP_Implement_PS
import os
import json
np.random.seed(1)

class FedAvgAPI(object):
   def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self.number_non_zero_entries = []
        self.number_orbits = 1
        self.vectors_local_AMP_each_orbit = [[] for num_orb in range(self.number_orbits)]
        self.sum_transform_vectors_each_orbit = [0 for num_orb in range(self.number_orbits)]
        self.sum_transform_vectorss = [0 for num_orb in range(self.number_orbits)]
        self.error_each_orbit = [0 for num_orb in range(self.number_orbits)]
        self.p = 1
        self.AMP_TopK = [False, True] # % The first entry: AMP algorithm and the second entry: TopK -- > TopK = [False , True]  and AMP = [True, False]
        self.test_acc_Sync = []
        #self.gradients = [[] for idx in range(self.args.comm_round)]

        

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

   def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
            #print('train_data_local_num_dict[client_idx] = ' + str(self.train_data_local_num_dict))
        logging.info("############setup_clients (END)#############")

   def train(self):
        w_global = self.model_trainer.get_model_params()

        ### This for-loop makes a zero-values dictionary with the same size of global parameters.
        residual = copy.deepcopy(w_global)
        for key_idx in range(len(residual)):
            values_key_idx = list(residual.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            residual[list(residual.keys())[key_idx]] = zeros_key_idx


        w_residual_all_users = [residual for i in range(self.args.client_num_per_round)]    # Makes a vector for residual for all users
        
        gradients_save = {}
        for idx_sat_number in range(1, self.args.comm_round):
             key = f'round{idx_sat_number}'
             gradients_save[key] = {}
             for idx_sat in range(1, self.args.client_num_per_round+1):
                   key1 = f'sat_number{idx_sat}'
                   gradients_save[key][key1] = []

        


        round_idx = 0
        self._local_test_on_all_clients(round_idx, 0, 0)    # (round_idx, size_bit, size_data_TopK_method)
        for round_idx in range(1, self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            vectors_local_AMP = []
            vectors_local_Spar = []
            w_global_temp = copy.deepcopy(w_global)
            w_global_temp_to_train = copy.deepcopy(w_global)
            self.number_non_zero_entries = []   # To derive the non-zero elements

            for idx, client in enumerate(self.client_list):

                model_old = copy.deepcopy(w_global_temp)
                gradients = copy.deepcopy(w_global_temp)
                w = client.train(w_global_temp_to_train, round)
                model_new = copy.deepcopy(w)

                for key_idx in range(len(gradients)):   # % Derive the gradients
                   print(key_idx)
                   values_key_idx1 = list(model_old.values())[key_idx]
                   if key_idx == 1:
                      print(values_key_idx1)
                   values_key_idx2 = list(model_new.values())[key_idx]
                   if key_idx == 1:
                      print(values_key_idx2)
                   values_key_idx3 = values_key_idx2 - values_key_idx1
                   if key_idx == 1:
                        print(values_key_idx3)
                   gradients[list(gradients.keys())[key_idx]] = values_key_idx3
                #self.gradients[round_idx-1].append(gradients)
                #Gradient_data = { 'Gradients' : self.gradients}
                #print(f"111$$$ gradient111 : {gradients['linear.weight'][1][600]}")
                #variable_name = 'round' % round_idx
                

                # Convert the NumPy array to a Python list (JSON serializable)
                gradient_copy = copy.deepcopy(gradients)   
                python_list = self.change_weight_vector(gradient_copy)

                numpy_array = python_list.numpy()

                # Convert the NumPy array to a Python list (JSON serializable)
                python_list1 = numpy_array.tolist()
                #print(f"222$$$ gradient111 : {python_list1[1384]}")

                gradients_save[f'round{round_idx}'][f'sat_number{idx+1}'] = python_list1
                #print(f"'self.gradients_save' : {gradients_save}")
                #print(f"#### len(Gradient_data) = {len(gradients[list(gradients.keys())[key_idx]])}")

                with open('parameters_IID_check.json', 'w') as json_file:
                           json.dump(gradients_save, json_file)

                #print(f"gradients.element_size() = {gradients.element_size()}")
                #print(f"gradients.nelement() = {gradients.nelement()}")
                #print(print(x2.element_size() * x2.nelement() * 8))
                w_wo_sparsification = copy.deepcopy(gradients)
                w_residual_user = copy.deepcopy(w_residual_all_users[idx])
                #print(f"1111 w_wo_sparsification = {w_wo_sparsification['linear.weight'][2][200]}")



                if self.AMP_TopK[0] == False and self.AMP_TopK[1] == True:

                    w_sparsified, w_residual_new = self._TopK_Threshold_sparsification_all(w_wo_sparsification, w_residual_user, idx, self.p)
                    w_residual_all_users[idx] = copy.deepcopy(w_residual_new) # Renew the residual values
                    #print(f"2222 w_wo_sparsification = {w_sparsified['linear.weight'][2][200]}")
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w_sparsified)))

                elif self.AMP_TopK[0] == True and self.AMP_TopK[1] == False:

                    w_sparsified, w_residual_new, w_sparsified_AMP_y, N_AMP, n_AMP, k_AMP, AMP_all_entries = self._AMP_Threshold_sparsification_all(w_wo_sparsification, w_residual_user, idx, self.p)
                    w_residual_all_users[idx] = copy.deepcopy(w_residual_new) # Renew the residual values
                    vectors_local_AMP.append(w_sparsified_AMP_y) # This is the y (y =Ax).
                    vectors_local_Spar.append(AMP_all_entries)

                    self.number_non_zero_entries.append(AMP_all_entries)
                    '''
                    if idx == 9:
                      print(f"len(self.number_non_zero_entries) = {len(self.number_non_zero_entries)}")
                      count_non_zeroone_orbit1 = self.count_non_zero_elements(self.number_non_zero_entries[0:5])
                      print(f"@@@count_non_zeroone_orbit1 = {count_non_zeroone_orbit1}")
                      count_non_zeroone_orbit2 = self.count_non_zero_elements(self.number_non_zero_entries[5:10])
                      print(f"###count_non_zeroone_orbit2 = {count_non_zeroone_orbit2}")
                    '''
                    #count_non_zero_all = self.count_non_zero_elements(self.number_non_zero_entries)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w_sparsified_AMP_y)))


            Num_sat_per_orbit = int(len(self.client_list) / self.number_orbits)
            if self.AMP_TopK[0] == True and self.AMP_TopK[1] == False:

               size_data_TopK_method = self._TopK_Alg_Size_MPs(vectors_local_Spar, Num_sat_per_orbit) # The size of uploaded parameters in all orbit
               #print(f"$$ size_data_TopK_method = {size_data_TopK_method}")
               size_data_AMP_method = self._AMP_Alg_Size_MPs(vectors_local_AMP, Num_sat_per_orbit) # The size of uploaded parameters in all orbit
               #print(f"$$ size_data_AMP_method = {size_data_AMP_method}")
               avg_transform_vectors =  self.PS_Op_AMP(vectors_local_AMP, Num_sat_per_orbit, N_AMP, n_AMP)


               w_global = self._aggregate_gradients(w_locals, w_global_temp, avg_transform_vectors, self.AMP_TopK)
               self.model_trainer.set_model_params(w_global)
               if round_idx % self.args.frequency_of_the_test == 0:
                        self._local_test_on_all_clients(round_idx, size_data_AMP_method, size_data_TopK_method)

            elif self.AMP_TopK[0] == False and self.AMP_TopK[1] == True:
                avg_transform_vectors = 0
                w_global = self._aggregate_gradients(w_locals, w_global_temp, avg_transform_vectors, self.AMP_TopK)
                self.model_trainer.set_model_params(w_global)
                if round_idx % self.args.frequency_of_the_test == 0:
                            self._local_test_on_all_clients(round_idx, 0, 0)






   def _TopK_Alg_Size_MPs(self, vectors_local_Spar, Num_sat_per_orbit):

       size_data_TopK_method = 0
       for idx_orb in range(self.number_orbits):    # Size of all satellites in the constellation in TopK
           size_data_TopK_method = size_data_TopK_method + self._size_data_TopK(vectors_local_Spar[idx_orb*Num_sat_per_orbit: idx_orb*Num_sat_per_orbit + Num_sat_per_orbit]) # Each entry of
       return size_data_TopK_method

   def _AMP_Alg_Size_MPs(self, vectors_local_AMP, Num_sat_per_orbit):
       size_data_AMP_method = 0
       for idx_orb in range(self.number_orbits):    # Size of all satellites in the constellation
                  size_data_AMP_method = size_data_AMP_method + self._size_data_AMP(vectors_local_AMP[idx_orb*Num_sat_per_orbit: idx_orb*Num_sat_per_orbit + Num_sat_per_orbit]) # Each entry of
       return size_data_AMP_method

   def PS_Op_AMP(self, vectors_local_AMP, Num_sat_per_orbit, N_AMP, n_AMP):
      for idx_orb in range(self.number_orbits):
              self.vectors_local_AMP_each_orbit[idx_orb] = vectors_local_AMP[idx_orb*Num_sat_per_orbit: idx_orb*Num_sat_per_orbit + Num_sat_per_orbit] # yi from each orbit
              self.sum_transform_vectorss[idx_orb] = self.sum_transform_AMP_(self.vectors_local_AMP_each_orbit[idx_orb]) # Gets sum from \sum_yi
              self.sum_transform_vectors_each_orbit[idx_orb], self.error_each_orbit[idx_orb] = AMP_Implement_PS(self.sum_transform_vectorss[idx_orb], N_AMP, n_AMP) # Gets \avg_xi from \avg_yi (AMP)
              print(f"self.error_each_orbit[idx_orb] = {self.error_each_orbit[idx_orb]}")
      sum_transform_vectors = sum(self.sum_transform_vectors_each_orbit)
      avg_transform_vectors = sum_transform_vectors / self.args.client_num_per_round   #% Average from partial aggregate of all orbits
      return avg_transform_vectors

   def count_non_zero_elements(self, AMP_all_entries):

        #print(f"AMP_all_entries = {AMP_all_entries}")
        #print(f"AMP_all_entries = {len(AMP_all_entries)}")
        x0 = copy.deepcopy(AMP_all_entries)
        a = torch.where(x0[0] != 0.0000000000)[0]
        #print(f"len(x0[0]) = {len(x0[0])}")
        #print(f"len(a) = {len(a)}")
        file1 = open("Original.txt", "w")
        file1.write("%s = %s\n" %("AMP_all_entries", AMP_all_entries))
        file1.close()
        for i in range(1,len(x0)):
           b = torch.where(x0[i] != 0.0000000000)[0]
           #print(f"len(b) = {len(b)}")
           a = torch.cat((b,a))

        unique, counts = torch.unique(a, return_counts=True)    # len(unique) gives the number of k(the number of non-zero) in the whole vector of all clients
        #print(f"len(unique) = {len(unique)}")
        return len(unique)

   def _TopK_Threshold_sparsification_all(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)

            operator1 = operator.add
            w_sum_train_residual = self.sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum resifual and new receive global model
            all_entries = self.change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
            print(f"&&&& len(all_entries) = {len(all_entries)}")                                                                 # % For Examle: w_sum_train_residual = OrderedDict([('linear', tensor([ 1,  5, 13])), ('weight', tensor([12, 14]))])
                                                                            # % all_entries = tensor([ 1.,  5., 13., 12., 14.])

            w_sparsification2 = self.Spar_Threshold(w_sum_train_residual, all_entries, p)   # Makes the received weights global and in the form of global matrix


            operator2 = operator.sub
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            w_residual_new = self.sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsification2, operator2) # % sum resifual and new receive global model

            return w_sparsification2, w_residual_new

   def _AMP_Threshold_sparsification_all(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)

            operator1 = operator.add
            w_sum_train_residual = self.sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum resifual and new receive global model
            all_entries = self.change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
                                                                            # % For Examle: w_sum_train_residual = OrderedDict([('linear', tensor([ 1,  5, 13])), ('weight', tensor([12, 14]))])
                                                                            # % all_entries = tensor([ 1.,  5., 13., 12., 14.])

            w_sparsified = self.Spar_Threshold(w_sum_train_residual, all_entries, p)   # Makes the received weights global and in the form of global matrix


            number_tensor = torch.numel(all_entries)
            number_tensor_accepted = int(number_tensor * p)
            #print(f"&&&& len(all_entries) = {number_tensor_accepted}")
            k = number_tensor_accepted
            #print(f"k = {k}")
            N = number_tensor
            n = int(N/5)
            print(f"n = {n}")
            all_entries_w_sparsification2 = []
            w_sparsification3 = copy.deepcopy(w_sparsified)
            AMP_all_entries = self.change_weight_vector(w_sparsification3)  # A vector with the value pf w_sparsification2
            #print(f"AMP_all_entries = {AMP_all_entries}")
            #t1 = time.time()
            w_sparsified_AMP_y = AMP_Implement_Sat(AMP_all_entries, N, n)    # Each satellite derives y, (y=Ax)
            #t2 = time.time()
            #print(f"^^^ t2-t1 = {t2-t1}")
            operator2 = operator.sub
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            w_residual_new = self.sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsified, operator2) # % sum resifual and new receive global model


            # vector with the form of w_global but with sparsified values, residual, y, N, n, k, same value of w_sparsification2 but sparsified version
            return w_sparsified, w_residual_new, w_sparsified_AMP_y, N, n, k, AMP_all_entries

   def write_vector_in_file(self, file_name, vector):
                   with open(file_name, "w") as f:
                        f.write(str(vector))

   def sum_subtraktion_residual_weight(self, w_sparsification1, w_residual_user, operator):

            w_sum_train_residual = copy.deepcopy(w_residual_user)
            for key_idx in range(len(w_residual_user)): # w_all = w_global + w_residual
                  values_key_idx1 = list(w_sparsification1.values())[key_idx]
                  values_key_idx2 = list(w_residual_user.values())[key_idx]
                  values_key_idx3 = operator(values_key_idx1 , values_key_idx2)
                  w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3
            return w_sum_train_residual

   def change_weight_vector(self, w_sum_train_residual):   # Combine all component of w_sum_train_residual
            all_entries_sep_values = []
            for g in range(len(w_sum_train_residual)):

               values_tensor = list(w_sum_train_residual.values())[g].float()
               flatten_values_tensor = torch.flatten(values_tensor)
               all_entries_sep_values.append(flatten_values_tensor)

            #print(f"all_entries_sep_values = {all_entries_sep_values}")
            all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            return all_entries

   def th_delete(self, tensor, indices): # To delete the indices from tensor
      mask = torch.ones(tensor.numel(), dtype=torch.bool)
      mask[indices] = False
      return tensor[mask]

   def _size_data_AMP(self, w_sparsification):

        x0 = copy.deepcopy(w_sparsification)
        #print(f"!!!!!!! AMPggg len(x0[0]) = {len(x0[0])}")
        number_sat_each_orbit  = len(x0)
        number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
        group_index  = [0 for i in range(number_groups_each_orbit)]
        indexes = torch.arange(number_sat_each_orbit)
        #print(indexes)


        for idx in range(number_groups_each_orbit):

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = self.th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = self.th_delete(indexes, [0])

                #print(group_index)


        group1 = []
        group2 = []
        group_all = []
        len_size_group = 0

        for idxx in range(len(group_index)):
            #print(group1)
            if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
               if group1 != []:

                   y1 = copy.deepcopy(group1)
                   #print(f"$$$y1[-1] = {y1[-1]}")
                   #print(f"@@@x0[group_index[idxx][0]] = {x0[group_index[idxx][0]]}")
                   group1.append(torch.add(y1[-1], x0[group_index[idxx][0]]))
                   #print(f"***group1 = {group1}")

                   y2 = copy.deepcopy(group2)
                   group2.append(torch.add(y2[-1],x0[group_index[idxx][1]]))

               elif group1 == []:

                   group1.append(x0[group_index[idxx][0]])
                   group2.append(x0[group_index[idxx][1]])


            elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:

                y1 = copy.deepcopy(group1)
                group1.append(torch.add(y1[-1],x0[group_index[idxx][0]]))

            elif  idxx == (len(group_index) - 1):

                group_all.append([group1,group2,sum(x0)])
        #print(group_all)
        flattened_list = flatten(group_all)
        #print(flattened_list[0])
        sizee = 0
        for i in range(len(flattened_list)):
                   #print(flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   #flattened_list[i] = list(flattened_list[i])
                   flattened_list[i] = flattened_list[i].long()
                   #print(type(flattened_list[i]))
                   sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   print(f"%%%%%%% = {flattened_list[i].nelement()}")
                   print(f"$$$$$$$ = {flattened_list[i].element_size()}")

        #print(f"AMP flattened_list[i].element_size() = {flattened_list[i].element_size()}")
        print(f" sizee = {sizee}")
        return sizee

   def _size_data_TopK(self,w_sparsification):


            if self.p != 1:
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 0
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())
                        #print(f"x1 = {x1}")
                    #print(f"###values = {len(values[0])}")
                    #print(f"###values = {w_sparsification.element_size()}")
                    #print(f"$$$indices = {indices}")
            elif self.p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 1000000
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())
                        #print(f"x1 = {x1}")
                    #print(f"###values = {w_sparsification.element_size()}")

            #################################################

            x0 = copy.deepcopy(values)
            number_sat_each_orbit  = len(x0)
            number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
            group_index  = [0 for i in range(number_groups_each_orbit)]
            indexes = torch.arange(number_sat_each_orbit)



            for idx in range(number_groups_each_orbit): # For example, [0,1] which 0 is from one group and 1 from the other group

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = self.th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = self.th_delete(indexes, [0])




            group1 = []
            group2 = []
            group_all = []
            group1_ind = []
            group2_ind = []
            group_all_ind = []
            len_size_group = 0

            for idxx in range(len(group_index)):
                if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
                   #print(group_index[idxx])
                   if group1 != []:

                       y1 = copy.deepcopy(group1)
                       y1_ind = copy.deepcopy(group1_ind)


                       group1.append([y1[-1], x0[group_index[idxx][0]]])
                       same1_ind1 = [x for x in y1_ind[-1] if x in indices[group_index[idxx][0]]]
                       same1_ind2 = [x for x in indices[group_index[idxx][0]] if x in y1_ind[-1]]
                       same11_ind = same1_ind1 + same1_ind2
                       same1_ind = []
                       [same1_ind.append(item) for item in same11_ind if item not in same1_ind]

                       not_same2_ind1 = [x for x in y1_ind[-1] if x not in indices[group_index[idxx][0]]]
                       not_same2_ind2 = [x for x in indices[group_index[idxx][0]] if x not in y1_ind[-1]]
                       not_same22_ind = not_same2_ind1 + not_same2_ind2
                       not_same2_ind = []
                       [not_same2_ind.append(item) for item in not_same22_ind if item not in not_same2_ind]
                       all_index = same1_ind + not_same2_ind
                       group1_ind.append(all_index)
                       for k in range(len(group1_ind)):
                           for k1 in range(len(group1_ind[k])):
                             #print(f"111&&& group1_ind = {group1_ind}")
                             if group1_ind[k][k1] == []:
                                    del(group1_ind[k])



                       y2 = copy.deepcopy(group2)
                       y2_ind = copy.deepcopy(group2_ind)
                       group2.append([y2[-1], x0[group_index[idxx][1]]])
                       same1_ind1 = [x for x in y2_ind[-1] if x in indices[group_index[idxx][1]]]
                       same1_ind2 = [x for x in indices[group_index[idxx][1]] if x in y2_ind[-1]]
                       same11_ind = same1_ind1 + same1_ind2
                       same1_ind = []
                       [same1_ind.append(item) for item in same11_ind if item not in same1_ind]
                       not_same2_ind1 = [x for x in y2_ind[-1] if x not in indices[group_index[idxx][1]]]
                       not_same2_ind2 = [x for x in indices[group_index[idxx][1]] if x not in y2_ind[-1]]
                       not_same22_ind = not_same2_ind1 + not_same2_ind2
                       not_same2_ind = []
                       [not_same2_ind.append(item) for item in not_same22_ind if item not in not_same2_ind]
                       all_ind = same1_ind + not_same2_ind
                       group2_ind.append(all_ind)


                   elif group1 == []:

                       group1.append(x0[group_index[idxx][0]])
                       group1_ind.append(indices[group_index[idxx][0]])
                       #print(f"%%%%group1_ind = {x0[group_index[idxx][0]]}")
                       group2.append(x0[group_index[idxx][1]])
                       group2_ind.append(indices[group_index[idxx][1]])
                       #print(f"%%%%group2_ind = {x0[group_index[idxx][1]]}")


                elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                    #print(f"AAA = {group1_ind}")
                    y1 = copy.deepcopy(group1)
                    y1_ind = copy.deepcopy(group1_ind)
                    group1.append([y1[-1], x0[group_index[idxx][0]]])
                    #print(group1)

                    index_same1 = [x for x in y1_ind[-1] if x in indices[group_index[idxx][0]]]
                    index_same2 = [x for x in indices[group_index[idxx][0]] if x in y1_ind[-1]]
                    index11_same = index_same1 + index_same2
                    index1_same = []
                    [index1_same.append(item) for item in index11_same if item not in index1_same]
                    #print(index1_same)
                    index_not_same1 = [x for x in y1_ind[-1] if x not in indices[group_index[idxx][0]]]
                    index_not_same2 = [x for x in indices[group_index[idxx][0]] if x not in y1_ind[-1]]
                    index11_not_same = index_not_same1 + index_not_same2
                    index1_not_same = []
                    [index1_not_same.append(item) for item in index11_not_same if item not in index1_not_same]

                    jj4 = index1_same + index1_not_same


                    jj = max(jj4)
                    group1_ind.append(jj4)
                    #print(f"CCCC = {group1_ind}")

                elif  idxx == (len(group_index) - 1):
                    group_all.append([group1, group2, x0])
                    flatten_group1_ind =flatten(group1_ind)
                    #print(f"((( ))) = {flatten_group1_ind}")

                    flatten_group1_ind = list(set(flatten_group1_ind))

                    flatten_group2_ind =flatten(group2_ind)
                    flatten_group2_ind = list(set(flatten_group2_ind))
                    #print(f"###((( ))) = {flatten_group2_ind}")
                    index_not_same1 = [x for x in flatten_group2_ind if x not in flatten_group1_ind]
                    flatten_group1_ind = flatten_group1_ind + index_not_same1



                    indices = flatten(indices)
                    indices = list(set(indices)) # remove the duplicated items
                    jj2_ind = [x for x in indices if x not in flatten_group1_ind]
                    flatten_group1_ind = flatten_group1_ind + jj2_ind

                    #group_all.append([group1, group2, x0])
                    group_all_ind.append([group1_ind, group2_ind,flatten_group1_ind])
                    #group_all_ind.append([group1_ind, group2_ind, indices])

            #print(f"# %%%%%%%%%%%%% = {group_all_ind}")
            flattened_list = flatten(group_all)
            #print(flattened_list)

            flattened_list_same_index = []

            for i in range(len(flattened_list)):
                flattened_list_same_index.extend(flattened_list[i].tolist())
                #print(f"**** flattened_list[i] = {len(flattened_list[i])}")

            flattened_list_ind = flatten(group_all_ind)
            #print(f"flattened_list_ind = {flattened_list_ind}")
            sizee_value = 0
            sizee_ind = 0
            flattened_list_ind_tensor = torch.tensor(flattened_list_ind)
            flattened_list_ind_tensor1 = torch.tensor(flattened_list_ind_tensor)
            sizee_value = flattened_list_ind_tensor1.element_size() * flattened_list_ind_tensor1.nelement() * 8
            #print(type(flattened_list_ind_tensor1))
            #print(f"TopK flattened_list[i].element_size() = {flattened_list_ind_tensor1.element_size()}")
            sizee_position = np.log2(7850) * flattened_list_ind_tensor1.nelement() # for each Position entry , we need log2(7850)
            #print(sizee_position)
            #print(f"TopK flattened_list[i].element_size() = {flattened_list_ind_tensor1.nelement()}")
            size_transmission = sizee_value + sizee_position

            return size_transmission

   def x():
            '''
            def _size_data_spar(self, w_sparsification):

                x1 = copy.deepcopy(w_sparsification)
                #print(f"!!!!!!! SPARggg len(x0[0]) = {len(x1[0])}")
                #print(f"x0[0] = {x0[0]}")
                #print(f"((( len(x1) = {len(x1)}")
                values = []
                indices = []
                for k in range(len(x1)):
                    b = x1[k] != 0
                    values.append(x1[k][b])
                    indices.append(b.nonzero().tolist())
                    #print(f"x1 = {x1}")
                    print(f"###values = {len(values[-1])}")
                    print(f"$$$indices = {len(indices[-1])}")


                x0 = copy.deepcopy(values)
                number_sat_each_orbit  = len(x0)
                number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
                group_index  = [0 for i in range(number_groups_each_orbit)]
                indexes = torch.arange(number_sat_each_orbit)
                #print(indexes)


                for idx in range(number_groups_each_orbit):

                    if len(indexes) > 2:
                        z = [indexes[0],indexes[1]]
                        group_index[idx] = z
                        indexes = self.th_delete(indexes, [0,1])
                    elif len(indexes) <= 2:
                        z = [indexes[0]]
                        group_index[idx] = z
                        indexes = self.th_delete(indexes, [0])

                #print(group_index)



                group1 = []
                group2 = []
                group_all = []
                group1_ind = []
                group2_ind = []
                group_all_ind = []
                len_size_group = 0

                for idxx in range(len(group_index)):
                    #print(group1)
                    if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
                       if group1 != []:

                           y1 = copy.deepcopy(group1)
                           y1_ind = copy.deepcopy(group1_ind)
                           #print(f"^^**^y1 = {y1[-1]}")
                           #print(f"^^**^y1 = {np.shape(y1)}")
                           group1.append([y1[-1], x0[group_index[idxx][0]]])
                           group1_ind.append([y1_ind[-1], indices[group_index[idxx][0]]])

                           y2 = copy.deepcopy(group2)
                           y2_ind = copy.deepcopy(group2_ind)
                           group2.append([y2[-1], x0[group_index[idxx][1]]])
                           group2_ind.append([y2_ind[-1], indices[group_index[idxx][1]]])

                       elif group1 == []:

                           group1.append(x0[group_index[idxx][0]])
                           group1_ind.append(indices[group_index[idxx][0]])
                           group2.append(x0[group_index[idxx][1]])
                           group2_ind.append(indices[group_index[idxx][1]])


                    elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:

                        y1 = copy.deepcopy(group1)
                        y1_ind = copy.deepcopy(group1_ind)
                        group1.append([y1[-1], x0[group_index[idxx][0]]])
                        group1_ind.append([y1_ind[-1], indices[group_index[idxx][0]]])

                    elif  idxx == (len(group_index) - 1):
                        group_all.append([group1, group2, x0])
                        group_all_ind.append([group1_ind, group2_ind, indices])

                #print(group_all)
                #print(group_all_ind)
                flattened_list = flatten(group_all)
                #print(flattened_list)

                flattened_list_same_index = []
                for i in range(len(flattened_list)):
                    flattened_list_same_index.extend(flattened_list[i].tolist())


                flattened_list_ind = flatten(group_all_ind)
                #print(flattened_list_ind)
                #print(len(flattened_list_ind))
                #print(len(flattened_list_same_index))
                #print(flattened_list_ind)
                sizee_value = 0
                sizee_ind = 0
                flattened_list_ind_tensor = torch.tensor(flattened_list_ind)
                sizee_ind = flattened_list_ind_tensor.element_size() * flattened_list_ind_tensor.nelement() * 8
                #print(flattened_list_ind_tensor.element_size())
                flattened_list_same_index_tensor = torch.tensor(flattened_list_same_index, dtype=torch.float64)
                #flattened_list_same_index_tensor = torch.double(flattened_list_same_index_tensor)
                sizee_list_values = flattened_list_same_index_tensor.element_size() * flattened_list_same_index_tensor.nelement() * 8
                #print(flattened_list_same_index_tensor.element_size())

                sizee = sizee_ind + sizee_list_values
                print(f" 888 sizee_ind = {sizee_ind}")
                print(f" 999 sizee_list_values = {sizee_list_values}")
                return sizee


            def Spar_Threshold(self, w_sum_train_residual, all_entries, p): # % Makes the vector sparse

                w_sparsification2 = copy.deepcopy(w_sum_train_residual)
                w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
                for g in range(len(w_sum_train_residual_main)):
                                     w_each_dic = list(w_sum_train_residual_main.values())[g]   # % Values of each dict_key
                                     w_each_dic_zero = torch.zeros(w_each_dic.shape)
                                     #print(f"$$w_each_dic_zero  = {w_each_dic_zero}")
                                     all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
                                     number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
                                     sort_flatten_matrix = torch.sort(all_entries_abs)
                                     number_tensor_accepted = int(number_tensor * p)
                                     #print(f"number_tensor_accepted = {number_tensor_accepted}")
                                     if number_tensor_accepted == 0:
                                         threshold = 100000
                                     else:
                                         threshold = sort_flatten_matrix[0][number_tensor - number_tensor_accepted]

                                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(w_each_dic)>=threshold, w_each_dic.double() , w_each_dic_zero.double())
                return w_sparsification2
            '''

   def Spar_Threshold(self, w_sum_train_residual, all_entries, p): # % Makes the vector sparse
        #print(f"w_sum_train_residual = {w_sum_train_residual}")
        #print(f"all_entries = {all_entries}")
        #time0 = time.time()
        w_sparsification2 = copy.deepcopy(w_sum_train_residual)
        all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
        number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
        zero_number_tensor = torch.zeros(number_tensor)
        ordered, sort_flatten_matrix_index = torch.sort(all_entries_abs)
        number_tensor_accepted = int(number_tensor * p)

        selected_entries_values = ordered[number_tensor-number_tensor_accepted:]

        selected_entries_index = sort_flatten_matrix_index[number_tensor-number_tensor_accepted:]


        values_updated = torch.zeros(number_tensor)
        '''
        for i in range(number_tensor):
            for j in range(len(selected_entries_index)):
              if i == selected_entries_index[j]:
                values_updated[i] = all_entries[i]
        '''

        for j in selected_entries_index:
          values_updated[j] = all_entries[j]

        #time1 = time.time()
        #print(f"time = {time1 - time0}")


        w_sparsification3 = copy.deepcopy(w_sparsification2)

        k1 = 0

        for i in range(len(list(w_sparsification2.keys()))): # This for changes the full local vector to matrix in the same shape of weights

                            x1 = list(w_sparsification2.keys())[i]

                            if len(w_sparsification2[x1].size()) != 1:
                              for j in range(len(w_sparsification2[x1])):

                                z = len(w_sparsification2[x1][j])
                                w_sparsification3[x1][j] = values_updated[k1:k1+z]
                                k1 = k1+z

                            elif len(w_sparsification3[x1].size()) == 1: # This if is for the bias part

                                  z = len(w_sparsification3[x1])
                                  w_sparsification3[x1] = values_updated[k1:k1+z]
                                  k1 = k1+z


        #print(f"w_sparsification3 = {w_sparsification3}")

        return w_sparsification3

   def sum_transform_AMP_(self, vectors_local_AMP):

        sum_output_transform_AMP = sum(vectors_local_AMP)
        #avg_output_transform_AMP = sum_output_transform_AMP / len(vectors_local_AMP)
        return sum_output_transform_AMP

   def _aggregate_gradients(self, w_locals, w_global, avg_transform_vectors, AMP_SPAR):


        if self.AMP_TopK[0] == True and self.AMP_TopK[1] == False:  # Only put the averaged values in avg_transform_vectors in the form of orderdict same as w_global
                Grad = copy.deepcopy(w_global)

                #print(f"!!!!! len(sum_transform_vectors) = {sum_transform_vectors}")
                k1 = 0

                for i in range(len(list(w_global.keys()))): # This for changes the full local vector to matrix in the same shape of weights

                    x1 = list(w_global.keys())[i]

                    if len(w_global[x1].size()) != 1:
                      for j in range(len(w_global[x1])):

                        z = len(w_global[x1][j])
                        Grad[x1][j] = tensor(avg_transform_vectors[k1:k1+z])
                        k1 = k1+z

                    elif len(Grad[x1].size()) == 1: # This if is for the bias part

                          z = len(Grad[x1])
                          Grad[x1] = tensor(avg_transform_vectors[k1:k1+z])
                          k1 = k1+z
        elif self.AMP_TopK[0] == False and self.AMP_TopK[1] == True:    # Completely derives the average of gradients of weights
                       #print('Hi')
                       training_num = 0
                       for idx in range(len(w_locals)):
                           (sample_num, averaged_params) = w_locals[idx]
                           training_num += sample_num

                       (sample_num, Grad) = w_locals[0]
                       #print('averaged_params: '+str(averaged_params))
                       #print('averaged_params.keys(): '+str(averaged_params.keys()))
                       for k in Grad.keys():
                           #print('k: '+str(k))
                           for i in range(0, len(w_locals)):
                               local_sample_number, local_model_params = w_locals[i]
                               w = local_sample_number / training_num
                               if i == 0:
                                   Grad[k] = local_model_params[k] * w
                               else:
                                   Grad[k] += local_model_params[k] * w

        #print(f"1111111### w_local = {w_local}")
        w_global_new = copy.deepcopy(w_global)
        for key_idx in range(len(w_global)):    # This for derives the new global parameters by using the gradients and old global weights
              values_key_idx1 = list(w_global.values())[key_idx]
              values_key_idx2 = list(Grad.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_global_new[list(w_global_new.keys())[key_idx]] = values_key_idx3


        return w_global_new

   def _local_test_on_all_clients(self, round_idx, size_bit, size_data_TopK_method):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        #temp_client0   = copy.deepcopy(self.client_list[0])
        #client = self.client_list[0]

        for idx2, client2 in enumerate(self.client_list):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            #if  idx2 < 5:
            if self.test_data_local_dict[idx2] is None:
                continue
            client2.update_local_dataset(idx2, self.train_data_local_dict[idx2],
                                        self.test_data_local_dict[idx2],
                                        self.train_data_local_num_dict[idx2])
            #print('self.train_data_local_num_dict[idx2] = '+str(self.train_data_local_num_dict[idx2]))
            #print('self.test_data_local_num_dict[idx2] = '+str(self.test_data_local_num_dict[idx2]))
            #print('data = '+str(self.test_data_local_dict[idx2][0]))
            # train data
            #train_local_metrics = client2.local_test(False)
            #train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            #train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            #train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client2.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))


            """
            Note: CI environment is CPU-based computing.
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        #train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        #train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        #stats = {'training_acc': train_acc, 'training_loss': train_loss}
        #wandb.log({"Train/Acc": train_acc, "round": round_idx})
        #wandb.log({"Train/Loss": train_loss, "round": round_idx})
        #logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)


        stats_upload_size = {"Upload_Bits_each_Orbit_AMP": size_bit}
        wandb.log({"Upload_Bits_each_Orbit_AMP": size_bit, "round": round_idx})
        logging.info(stats_upload_size)


        stats_upload_size = {"Upload_Bits_each_Orbit_Spar": size_data_TopK_method}
        wandb.log({"Upload_Bits_each_Orbit_Spar": size_data_TopK_method, "round": round_idx})
        logging.info(stats_upload_size)

        '''
        self.test_acc_Sync.append(test_acc)
        if round_idx == self.args.round-1:
                            with open('../../../FedISL_Sync_ACC_results.py', 'w') as f:
                                 f.write(f'FedISL_Sync_ACC_results =  {self.test_acc_Sync}')
        '''
