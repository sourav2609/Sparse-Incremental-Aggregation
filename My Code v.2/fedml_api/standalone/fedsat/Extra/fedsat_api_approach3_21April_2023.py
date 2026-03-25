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
        #Index_FedISL_Async_Bremen = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4, 0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
        
        self.visting_orbit_index = Index_FedISL_Async_Bremen
        self.visting_orbit_time = Time_FedISL_Async_Bremen
        #self.visting_orbit_index = np.array(self.visting_orbit_index)
        print(f"$$self.visting_orbit_index = {self.visting_orbit_index}")
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

        number_samples_each_client = []
        for idx_sat, idx_client in enumerate(self.client_list):
             number_samples_each_client.append(idx_client.get_sample_number())

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
            check_w_global1 = []  
            print(f"### {w_global.keys()}")
            for k11 in range(self.num_satellite):
                check_w_global1.append((w_global_matrix[k11]['linear.bias'][0], k11))
            print(f"111 check_w_global1 = {check_w_global1}")

            w_global_temp = copy.deepcopy(w_global)
            for i_sat, client2 in enumerate(self.client_list):

                if i_sat in satellite_idxes_in_orbit:

                       w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_pass_sparsification = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_pass_without_train = copy.deepcopy(w_global_matrix)

                       

                       w = client2.train(w_global_matrix_pass_train, comm_round)
                       w_trained_copy = copy.deepcopy(w)
                       #if i_sat == 31:
                       #   print(f"2222 w_trained_copy = {w_trained_copy['linear.weight'][0][12]}")                       
                    
                       for key_idx in range(len(w_trained_copy)):   # % Derive the new gradients (g_k(w_k^{n_k+1,I})

                           values_key_idx1 = list(w_global_matrix_pass_sparsification.values())[key_idx]
                           values_key_idx2 = list(w_trained_copy.values())[key_idx]
                           values_key_idx3 = values_key_idx2 - values_key_idx1
                           gradient_matrix_new[i_sat][list(gradient_matrix_new[i_sat].keys())[key_idx]] = values_key_idx3

                       #if i_sat == 31:
                       #   print(f"3333 gradient_matrix_new = {gradient_matrix_new[i_sat]['linear.weight'][0][12]}")  
                       #w_global = copy.deepcopy(w_global_temp)
                       #w_residual_all_users[i_sat] = copy.deepcopy(w_residual_all_users[i_sat])
                       w1 = copy.deepcopy(gradient_matrix_new[i_sat])
                       w_locals[i_sat] = (client2.get_sample_number(), copy.deepcopy(w1))

            w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
            #for k11 in range(self.num_satellite):
            #    if k11 == 31:
            #              (x1,y1) = w_locals_pass_to_aggregate[k11]
                          #print(f"333 w = {y1['linear.weight'][0][12]}")            
            
            
            w_global = self._aggregate(w_locals_pass_to_aggregate, w_global_temp, satellite_idxes_in_orbit, comm_round, index_presence_sat_old, index_presence_sat_new, number_samples_each_client)
            print(f"000 w_global = {w_global['linear.bias'][0]}")
            index_presence_sat_old = copy.deepcopy(index_presence_sat_new)
            self.model_trainer.set_model_params(w_global)
            for j_sat in satellite_idxes_in_orbit:                
                          w_global_matrix[j_sat] = copy.deepcopy(w_global)
                          
                          #if j_sat == 31:
                          #   print(f"555 w_global_matrix = {w_global_matrix[j_sat]['linear.weight'][0][12]}")  
                          #         
            check_w_global2 = []  
            for k11 in range(self.num_satellite):
                check_w_global2.append((w_global_matrix[k11]['linear.bias'][0], k11))
            print(f"222 check_w_global2 = {check_w_global2}")
            if comm_round % self.args.frequency_of_the_test == 0:
               self._local_test_on_all_clients(comm_round+1, x_axes = "round")


    def _aggregate(self, w_locals, w_global_temp, satellite_idxes_in_orbit, comm_round, index_presence_sat_old, index_presence_sat_new, number_samples_each_client):

        for l in range(len(satellite_idxes_in_orbit)):
            if satellite_idxes_in_orbit[l] == 31:
                (sample_nnum, parameters) = w_locals[satellite_idxes_in_orbit[l]]
                print(f"555 w_locals = {parameters['linear.weight'][0][12]}")
        

        training_num = 0
        for idx2 in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx2]
            if sample_num != 0:
               training_num += (sample_num)
        
        #training_num = 0
        #for i3 in range(len(number_samples_each_client)):
        #    training_num += number_samples_each_client[i3]


        (sample_num_new, averaged_params_new) = w_locals[satellite_idxes_in_orbit[0]]

        #print(f"********satellite_idxes_in_orbit = {satellite_idxes_in_orbit}")
        xw = []
        for k in averaged_params_new.keys():
            temp_sum_1 = 0
            
            for i1 in satellite_idxes_in_orbit:
                     xw.append(i1)
                     (local_sample_number, local_model_params) = w_locals[i1]
                     temp_sum_1 += 1
                     #w = (local_sample_number) / training_num
                     
                     w = number_samples_each_client[i1] / training_num
                     #w = 1 / 40
                     print(f"&&& *** w = {w}")
                     if temp_sum_1 == 1:
                        averaged_params_new[k] = local_model_params[k] * w
                     else:
                        averaged_params_new[k] += local_model_params[k] * w
        print(xw)
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

        '''
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
        '''