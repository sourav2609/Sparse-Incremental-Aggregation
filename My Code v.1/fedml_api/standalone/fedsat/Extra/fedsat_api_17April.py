### Approach 3 with sparsification, w^{n+1} = w^{n}+{D_k/D}*g_k(w_k^{n_k})
import copy
import logging
import random
import operator
import numpy as np
import torch
import wandb
import time
torch.set_printoptions(precision=10)
torch.set_printoptions(profile="full")

from fedml_api.standalone.fedsat.client import Client
from Time_Index_FedISL_Async_MNIST_Bremen import Index_FedISL_Async_Bremen
from Time_Index_FedISL_Async_MNIST_Bremen import Time_FedISL_Async_Bremen


class fedsat(object):
    def __init__(self, dataset, device, args, model_trainer, initial_weights_matrix):
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
        self.initial_weights_matrix = initial_weights_matrix
        #print('lennn = ' + str(len(self.initial_weights_matrix)))
        self.test_acc_time = []
        self.test_acc_round = []
        self.test_time = []
        self.test_round = []
        #self.visting_orbit_index = [0,1,2,3,4]
        #self.visting_orbit_index = Index_FedISL_Async_Bremen
        #Index_FedISL_Async_Bremen = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,]
        self.visting_orbit_index = Index_FedISL_Async_Bremen
        self.visting_orbit_time = Time_FedISL_Async_Bremen
        #self.visting_orbit_index = np.array(self.visting_orbit_index)
        #print(f"$$self.visting_orbit_index = {self.visting_orbit_index}")
        self.num_satellite = 40
        self.num_orbit = 5
        self.p = 1
        self.acc_FedISL = []
        #self.start_visting_satellite_GS_index = visibility_satellite_GS_matrix()

        #self.difference_time = satellite_GS_matrix_time_difference()

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx0 in range(self.args.client_num_per_round):
            c = Client(client_idx0, train_data_local_dict[client_idx0], test_data_local_dict[client_idx0],
                       train_data_local_num_dict[client_idx0], self.args, self.device, model_trainer)
            #print(f"***********test_data_local_dict[client_idx0] = {train_data_local_dict[client_idx0].dataset.target}")
            #logging.info({len(train_data_local_dict[client_idx0].dataset.target)})
            #logging.info({len(test_data_local_dict[client_idx0].dataset.target)})
            self.client_list.append(c)
        #print('self.client_list = '+str(self.client_list))
        #logging.info("############setup_clients (END)#############")



    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights

        # To have the same initial weight for start as set up in the main_fed

        w_global = self.model_trainer.get_model_params()



        w_locals = []   # w_local has the w_local of each satellite
        #w_locals_new = []
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        w_residual_all_users = []
        index_presence_sat_old = [0 for idx in range(self.num_satellite)]
        index_presence_sat_new = [0 for idx in range(self.num_satellite)]
        gradient_matrix_new = []
        num_sat_each_orbit = int(self.args.client_num_per_round / self.num_orbit)
        satellites_index_orbit = {}
        for k in range(self.num_orbit):
              satellites_index_orbit[str(k)] = range(num_sat_each_orbit*k, num_sat_each_orbit*k+num_sat_each_orbit)


        first_values = copy.deepcopy(w_global)
        for key_idx in range(len(first_values)):
            values_key_idx = list(first_values.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            first_values[list(first_values.keys())[key_idx]] = zeros_key_idx


        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals
          w_locals.append((0, copy.deepcopy(first_values))) # w_local has the w_local of each satellite
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS
          w_residual_all_users.append(copy.deepcopy(first_values))
          gradient_matrix_new.append(copy.deepcopy(first_values))


        self._local_test_on_all_clients(0, x_axes = "round")
        for comm_round in range(self.args.comm_round):

            index_orbit1 = self.visting_orbit_index[comm_round]  #Which satellites will be active in this time
            satellite_idxes_in_orbit = satellites_index_orbit[str(index_orbit1)]     
            index_presence_sat_new[index_orbit1] = 1       
            #print(f"### w_global = {w_global['linear.weight'][0][12]}")
            for i_sat, client2 in enumerate(self.client_list):

                if i_sat in satellite_idxes_in_orbit:
                       if i_sat == 31:
                          print(f"1111 w_global_matrix[i_sat] = {w_global_matrix[i_sat]['linear.weight'][0][12]}")
                       w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_pass_sparsification = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_pass_without_train = copy.deepcopy(w_global_matrix)

                       w_global_temp = copy.deepcopy(w_global)

                       w = client2.train(w_global_matrix_pass_train, comm_round)
                       w_trained_copy = copy.deepcopy(w)
                       if i_sat == 31:
                          print(f"2222 w = {w['linear.weight'][0][12]}")                       
                       for key_idx in range(len(w_trained_copy)):   # % Derive the new gradients (g_k(w_k^{n_k+1,I})

                           values_key_idx1 = list(w_global_matrix_pass_sparsification.values())[key_idx]
                           values_key_idx2 = list(w_trained_copy.values())[key_idx]
                           values_key_idx3 = values_key_idx2 - values_key_idx1
                           gradient_matrix_new[i_sat][list(gradient_matrix_new[i_sat].keys())[key_idx]] = values_key_idx3

                        
                       #w_wo_sparsification = copy.deepcopy(gradient_matrix_new[i_sat])
                       w_global = copy.deepcopy(w_global_temp)
                       #w_residual_user = copy.deepcopy(w_residual_all_users[i_sat])
                       #w_sparsified, w_residual_new = self._TopK_Threshold_sparsification_all(w_wo_sparsification, w_residual_user, i_sat, self.p)
                       w_residual_all_users[i_sat] = copy.deepcopy(w_residual_all_users[i_sat])
                       w = copy.deepcopy(gradient_matrix_new[i_sat])


                       w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                       #w_global_matrix2 = copy.deepcopy(w_global_matrix)
                       w_locals[i_sat] = (client2.get_sample_number(), copy.deepcopy(w))

            w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
            for k11 in range(self.num_satellite):
                if k11 == 31:
                          (x1,y1) = w_locals_pass_to_aggregate[k11]
                          print(f"333 w = {y1['linear.weight'][0][12]}")            
            '''
            x_sat = []
            for k11 in range(self.num_satellite):
                (local_sample_number1, local_model_params1) = w_locals[k11]
                x_sat.append(local_model_params1['linear.weight'][0][12])  
            print(f"$$$$ satellite_idxes_in_orbit = {satellite_idxes_in_orbit}")
            print(f"%%% x_sat = {x_sat}") 
            sum_x_sat = 0
            for k2 in satellite_idxes_in_orbit:
                (local_sample_number1, local_model_params1) = w_locals[k2]
                sum_x_sat = sum_x_sat +  local_model_params1['linear.weight'][0][12]   
            print(f"%%% sum_x_sat = {sum_x_sat}") 
            '''
            w_global = self._aggregate(w_locals_pass_to_aggregate, w_global_temp, satellite_idxes_in_orbit, comm_round, index_presence_sat_old, index_presence_sat_new)
            index_presence_sat_old = copy.deepcopy(index_presence_sat_new)
            #print(f"222 index_presence_sat_old = {index_presence_sat_old}")
            #print(f" w_global = {w_global}")
            self.model_trainer.set_model_params(w_global)
            #print(f" GG w_global_matrix before = {w_global_matrix}")
            #if j_sat in satellite_idxes_in_orbit:
            for j_sat in satellite_idxes_in_orbit:                
                          w_global_matrix[j_sat] = copy.deepcopy(w_global)
                          

            #print(f" GG w_global_matrix after = {w_global_matrix}")
            if comm_round % self.args.frequency_of_the_test == 0:
               self._local_test_on_all_clients(comm_round+1, x_axes = "round")


    def _aggregate(self, w_locals, w_global_temp, satellite_idxes_in_orbit, comm_round, index_presence_sat_old, index_presence_sat_new):

        for l in range(len(satellite_idxes_in_orbit)):
            if satellite_idxes_in_orbit[l] == 31:
                (sample_nnum, parameters) = w_locals[satellite_idxes_in_orbit[l]]
                print(f"555 w_locals = {parameters['linear.weight'][0][12]}")
        

        training_num = 0
        for idx2 in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx2]
            if sample_num != 0:
               training_num += (sample_num)

        (sample_num_new, averaged_params_new) = w_locals[satellite_idxes_in_orbit[0]]

        #print(f"********satellite_idxes_in_orbit = {satellite_idxes_in_orbit}")
    
        for k in averaged_params_new.keys():
            temp_sum_1 = 0
            for i1 in satellite_idxes_in_orbit:

                     (local_sample_number, local_model_params) = w_locals[i1]
                     temp_sum_1 += 1
                     w = (local_sample_number) / training_num
                     w = 1 / 40
                     #print(f"&&& *** w = {w}")
                     if temp_sum_1 == 1:
                        averaged_params_new[k] = local_model_params[k] * w
                     else:
                        averaged_params_new[k] += local_model_params[k] * w

        #print(f"*** averaged_params_new = {averaged_params_new['linear.weight'][0][12]}")
        num_old_non_zero = [1 for i in index_presence_sat_old if i == 1]
        num_new_non_zero = [1 for i in index_presence_sat_new if i == 1]
        new_weight_multiply = (len(num_old_non_zero) ) / (len(num_new_non_zero))
        
        if comm_round == 0:
            new_weight_multiply = 1
        ##print(f"^^^new_weight_multiply = {new_weight_multiply}")    
        #print(f"##@!&&&&new_weight_multiply = {new_weight_multiply}")

        first_values = copy.deepcopy(w_global_temp)
        for key_idx in range(len(first_values)):
                                values_key_idx = list(first_values.values())[key_idx]
                                zeros_key_idx = torch.zeros(values_key_idx.shape)
                                first_values[list(first_values.keys())[key_idx]] = zeros_key_idx

        w_global_temp_new = copy.deepcopy(first_values)
        #print(f"666 w_global_temp = {w_global_temp['linear.weight'][0][12]}")
        for k in w_global_temp_new.keys():
                                #w_global_temp_new[k] = (new_weight_multiply*w_global_temp[k]) + averaged_params_new[k]
                                #if comm_round == 0:
                                #    w_global_temp_new[k] = averaged_params_new[k]
                                #else: 
                                      w_global_temp_new[k] = (w_global_temp[k]) + averaged_params_new[k]     
        #print(f"777w_global_temp_new = {w_global_temp_new['linear.weight'][0][12]}")
        return w_global_temp_new

    def _TopK_Threshold_sparsification_all(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

           w_sparsification1 = copy.deepcopy(w_sparsification)
           w_sparsification2 = copy.deepcopy(w_sparsification)

           operator1 = operator.add
           w_sum_train_residual = self.sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum residual and new local gradients
           all_entries = self.change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
           #print(f"&&&& len(all_entries) = {len(all_entries)}")                                                                 # % For Examle: w_sum_train_residual = OrderedDict([('linear', tensor([ 1,  5, 13])), ('weight', tensor([12, 14]))])
                                                                           # % all_entries = tensor([ 1.,  5., 13., 12., 14.])

           w_sparsification2 = self.Spar_Threshold(w_sum_train_residual, all_entries, p)   # Makes the received weights global and in the form of global matrix


           operator2 = operator.sub
           w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
           w_residual_new = self.sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsification2, operator2) # % sum resifual and new receive global model

           return w_sparsification2, w_residual_new

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

            
            #print(f"^^^^ all_entries_sep_values = {all_entries_sep_values}")

            all_entries_sep_values_cat = torch.tensor([])
            for k in range(len(all_entries_sep_values)):
                     all_entries_sep_values_cat = torch.cat((all_entries_sep_values_cat ,all_entries_sep_values[k]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            #print(f"333 all_entries = {all_entries}")    

            return all_entries

    def nested_matrix(self, lst, levels1, levels2, rows, columns, matrix):
        
        lst1= lst.tolist()
        #if len(lst1) > 20:
        #    print(f"lst1 = {lst1[0], lst1[20], lst1[-1]}")
        for l1 in range(levels1):
                    if levels1 == 1:
                        levels1 = 0
                    for i in range(levels2):
                        if levels2 == 1:
                         levels2 = 0
                        for j in range(rows):
                          if rows == 1:
                             rows = 0
                            
                          for k in range(columns):
                              if columns == 1:
                                  columns = 0
                              index = (l1 * levels2 * rows * columns) + (i * rows * columns) + (j * columns) + k 
                            
                              if levels1 != 0 and levels2 != 0 and rows != 0 and columns != 0:
                                        matrix[l1][i][j][k] = lst1[index]                       
                              elif levels1 == 0 and levels2 != 0 and rows != 0 and columns != 0:
                                        matrix[i][j][k] = lst1[index]
                              elif levels1 == 0 and levels2 == 0 and rows != 0 and columns != 0:
                                        matrix[j][k] = lst1[index]  
                              elif levels1 == 0 and levels2 == 0 and rows == 0 and columns != 0:
                                        matrix[k] = lst1[index] 
                                        #print(f"000matrix = {matrix}")
                              elif levels1 == 0 and levels2 == 0 and rows == 0 and columns == 0:
                                        print(f"000matrix = {matrix}")
                                        matrix = torch.tensor(lst1)
                                        print(f"111lst1 = {lst1}")
                                        print(f"111matrix = {matrix}")
        #if torch.numel(matrix) > 20:
        #   print(f"$$$ torch.flatten(matrix) = {torch.flatten(matrix)[0], torch.flatten(matrix)[20], torch.flatten(matrix)[-1]}")      
        return matrix

    def Spar_Threshold(self, w_sum_train_residual, all_entries, p): # % Makes the vector sparse
        
        w_sparsification2 = copy.deepcopy(w_sum_train_residual)
        all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
        number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
        zero_number_tensor = torch.zeros(number_tensor)
        ordered, sort_flatten_matrix_index = torch.sort(all_entries_abs)
        number_tensor_accepted = int(number_tensor * p)
        selected_entries_values = ordered[number_tensor-number_tensor_accepted:]
        selected_entries_index = sort_flatten_matrix_index[number_tensor-number_tensor_accepted:]


        values_updated = torch.zeros(number_tensor)
        for j in selected_entries_index:
          values_updated[j] = all_entries[j]

        first_values = copy.deepcopy(w_sparsification2)
        for key_idx in range(len(first_values)):
                    values_key_idx = list(first_values.values())[key_idx]
                    zeros_key_idx = torch.zeros(values_key_idx.shape)
                    first_values[list(first_values.keys())[key_idx]] = zeros_key_idx


        w_sparsification3 = copy.deepcopy(first_values)

        k1 = 0
        for i in range(len(list(w_sparsification2.keys()))): # This for changes the full local vector to matrix in the same shape of weights

                            x1 = list(w_sparsification2.keys())[i]           
                            size_x1 = w_sparsification2[x1].size()
                            #print(f"size_x1 = {size_x1}")
                            if len(size_x1) == 2:
                                    size_x1 = (1, 1, size_x1[0],size_x1[1])
                            elif   len(size_x1) == 3:  
                                size_x1 = (1, size_x1[0],size_x1[1], size_x1[2])
                            elif len(size_x1) == 1:
                                    size_x1 = (1, 1, 1, size_x1[0])
                            elif  len(size_x1) == 0: 
                                        size_x1 = (1, 1, 1, 1)
                            z = size_x1[0] * size_x1[1] * size_x1[2] * size_x1[3]
                            lst = values_updated[k1:k1+z]
                            w_sparsification3[x1] = copy.deepcopy(self.nested_matrix(lst, size_x1[0], size_x1[1], size_x1[2], size_x1[3], w_sparsification2[x1]))
                            
                            k1 = k1+z
                            

        return w_sparsification3

    def _local_test_on_all_clients(self, idx, x_axes):
        #print('x_axes = ' + str(x_axes))
        logging.info("################local_test_on_all_clients : {}".format(idx))

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


        #Users_test_5_10 = Users_test[5:10]
        #print('Users_test = '+str(Users_test_5_10))
        b1 = []
        for idx2, client3 in enumerate(self.client_list):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            #if  idx2 < 5:
            b1.append(idx2)
            if self.test_data_local_dict[idx2] is None:
                continue
            client3.update_local_dataset(idx2, self.train_data_local_dict[idx2],
                                        self.test_data_local_dict[idx2],
                                        self.train_data_local_num_dict[idx2])

            #print('=== test_data === {}:{} '.format(idx2, self.test_data_local_dict[idx2][0]))

            # train data
            #train_local_metrics = client2.local_test(False)
            #train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            #train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            #train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))


            # test data
            test_local_metrics = client3.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing.
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        #print(f"%%%%%%% b1 = {b1}")
        # test on training dataset
        #train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        #train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        #print('train_loss = ' + str(train_loss))
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])


        #stats_train = {'training_acc': train_acc, 'training_loss': train_loss}
        #logging.info(stats_train)
        stats_test = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats_test)
        wandb.log({"Test/Acc": test_acc, "round": idx})
        wandb.log({"Test/Loss": test_loss, "round": idx})


        if self.visting_orbit_index == Index_FedISL_Async_Bremen:
                if x_axes=="time":

                  wandb.log({"Test/Acc": test_acc, "time": idx})
                  wandb.log({"Test/Loss": test_loss, "time": idx})
                  self.acc_FedISL.append(test_acc)
                  with open('../../../Results/FedISL_Async_ACCC_Bremenn.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_results_Bremen =  {self.acc_FedISL}')

                elif x_axes=="round":
                          wandb.log({"Test/Acc": test_acc, "time": idx})
                          wandb.log({"Test/Loss": test_loss, "time": idx})
                          self.acc_FedISL.append(test_acc)
                          with open('../../../Results/FedISL_Async_ACC_Bremen_app3.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_Bremen =  {self.acc_FedISL}')

        elif self.visting_orbit_index == Index_FedISL_Async_NP:
                if x_axes=="time":

                  wandb.log({"Test/Acc": test_acc, "time": idx})
                  wandb.log({"Test/Loss": test_loss, "time": idx})
                  self.acc_FedISL.append(test_acc)
                  with open('../../../Results/FedISL_Async_ACCC_NP.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_results_NP =  {self.acc_FedISL}')

                elif x_axes=="round":
                          wandb.log({"Test/Acc": test_acc, "round": idx})
                          wandb.log({"Test/Loss": test_loss, "round": idx})
                          self.acc_FedISL.append(test_acc)
                          with open('../../../Results/FedISL_Async_ACC_NP.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_NP =  {self.acc_FedISL}')
