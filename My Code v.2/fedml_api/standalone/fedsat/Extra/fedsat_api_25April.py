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
from Results.Time_Index_FedISL_Async_MNIST_Bremen import Index_FedISL_Async_Bremen
from Results.Time_Index_FedISL_Async_MNIST_Bremen import Time_FedISL_Async_Bremen


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
        self.test_acc_time = []
        self.test_acc_round = []
        self.test_time = []
        self.test_round = []
        self.visting_orbit_index = Index_FedISL_Async_Bremen
        self.visting_orbit_time = Time_FedISL_Async_Bremen
        print(f"$$self.visting_orbit_index = {self.visting_orbit_index}")
        self.num_satellite = 40
        self.num_orbit = 5
        self.p = 1
        self.acc_FedISL = []

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx0 in range(self.args.client_num_per_round):
            c = Client(client_idx0, train_data_local_dict[client_idx0], test_data_local_dict[client_idx0],
                       train_data_local_num_dict[client_idx0], self.args, self.device, model_trainer)
            self.client_list.append(c)



    def train(self):

        w_global = self.model_trainer.get_model_params()

        number_samples_each_client = []
        for idx_sat, idx_client in enumerate(self.client_list):
             number_samples_each_client.append(idx_client.get_sample_number())

        w_locals = []   # w_local has the w_local of each satellite
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        w_residual_all_users = []
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


        for _ in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals
          w_locals.append((0, copy.deepcopy(first_values))) # w_local has the w_local of each satellite
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS
          w_residual_all_users.append(copy.deepcopy(first_values))
          gradient_matrix_new.append(copy.deepcopy(first_values))

        Train_Time = [[] for k in range(self.num_orbit)]
        self._local_test_on_all_clients(0, x_axes = "round")
        for comm_round in range(self.args.comm_round):

            index_orbit1 = self.visting_orbit_index[comm_round]  #Which satellites will be active in this time
            satellite_idxes_in_orbit = satellites_index_orbit[str(index_orbit1)]            
            w_global_temp = copy.deepcopy(w_global)

            for i_sat, client2 in enumerate(self.client_list):
                if i_sat in satellite_idxes_in_orbit:

                       w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_wo_train = copy.deepcopy(w_global_matrix[i_sat])   
                       #time1 = time.time()                    
                       w = client2.train(w_global_matrix_pass_train, comm_round)
                       #time2 = time.time()
                       #time_diff = time2-time1
                       #Train_Time[index_orbit1].append(time_diff)
                       w_trained_copy = copy.deepcopy(w)
                       
                       gradient_matrix_new[i_sat] = self.gradient_calc(w_trained_copy, w_global_matrix_wo_train)                    
                       gradient_sat = copy.deepcopy(gradient_matrix_new[i_sat])
                       w_locals[i_sat] = (client2.get_sample_number(), copy.deepcopy(gradient_sat))
            #print(f"Time required for Train = {Train_Time}")



            w_locals_pass_to_aggregate = copy.deepcopy(w_locals)           
            w_global = self._aggregate(w_locals_pass_to_aggregate, w_global_temp, satellite_idxes_in_orbit)
            self.model_trainer.set_model_params(w_global)

            for j_sat in satellite_idxes_in_orbit:                
                          w_global_matrix[j_sat] = copy.deepcopy(w_global)
                          
            if comm_round % self.args.frequency_of_the_test == 0:
               self._local_test_on_all_clients(comm_round+1, x_axes = "round")

    def gradient_calc(self, w_trained_copy, w_global_matrix_wo_train):

                 gard_temp = copy.deepcopy(w_trained_copy) 
                 for key_idx in range(len(w_trained_copy)):   # % Derive the new gradients (g_k(w_k^{n_k+1,I})

                           values_key_idx1 = list(w_global_matrix_wo_train.values())[key_idx]
                           values_key_idx2 = list(w_trained_copy.values())[key_idx]
                           values_key_idx3 = values_key_idx2 - values_key_idx1
                           gard_temp[list(gard_temp.keys())[key_idx]] = values_key_idx3

                 return gard_temp


    def _aggregate(self, w_locals, w_global_temp, satellite_idxes_in_orbit):



        training_num = 0
        for idx2 in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx2]
            if sample_num != 0:
               training_num += (sample_num)
        
        (sample_num_new, averaged_params_new) = w_locals[satellite_idxes_in_orbit[0]]


        for k in averaged_params_new.keys():
            temp_sum_1 = 0
            
            for i1 in satellite_idxes_in_orbit:
                     (local_sample_number, local_model_params) = w_locals[i1]
                     temp_sum_1 += 1
                     w = (local_sample_number) / training_num
                     if temp_sum_1 == 1:
                        averaged_params_new[k] = local_model_params[k] * w
                     else:
                        averaged_params_new[k] += local_model_params[k] * w

        w_global_temp_new = copy.deepcopy(w_global_temp)
        for k in w_global_temp_new.keys():
                    w_global_temp_new[k] = (w_global_temp[k]) + averaged_params_new[k] 
        #print(f"eee111 w_global_temp_new = {w_global_temp_new['linear.bias'][0]}")            
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

        
        if self.visting_orbit_index == Index_FedISL_Async_Bremen:
                          
                          wandb.log({"Test/Acc": test_acc, "time": idx})
                          wandb.log({"Test/Loss": test_loss, "time": idx})
                          self.acc_FedISL.append(test_acc)
                          with open('../../../Results/FedISL_Async_ACC_MNIST_Bremen.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_MNIST_Bremen =  {self.acc_FedISL}')

        elif self.visting_orbit_index == Index_FedISL_Async_NP:

                          wandb.log({"Test/Acc": test_acc, "round": idx})
                          wandb.log({"Test/Loss": test_loss, "round": idx})
                          self.acc_FedISL.append(test_acc)
                          with open('../../../Results/FedISL_Async_ACC_MNIST_NP.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_NP =  {self.acc_FedISL}')
        