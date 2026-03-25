### Approach 2, Sparsification, Satellite gradient the (w_k^{n_k,I}-w_k^{n_k-1,I})
import copy
import logging
import random
import operator

import numpy as np
import torch
import wandb
import time

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
        self.visting_orbit_index = Index_FedISL_Async_Bremen
        self.visting_orbit_time = Time_FedISL_Async_Bremen
        #self.visting_orbit_index = np.array(self.visting_orbit_index)
        print(f"$$self.visting_orbit_index = {self.visting_orbit_index}")
        self.num_satellite = 40
        self.num_orbit = 5
        self.acc_FedISL = []
        self.p = 1
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

        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        gradient_matrix = []
        gradient_matrix_spar = []
        w_residual_all_users = []
        w_local_last = []   # w_local has the w_local of each satellite
        w_local_new = []   # w_local has the w_local of each satellite
        client_indexes = range(self.args.client_num_per_round)
        matrix_idx = []
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round) # This vector is defined for controlling that each satellite
        update_epoch_for_round_figure = 0
        update_epoch = 0    # Shows how many satellite have seen the GS
        sample_number = []
        delta_weight = []

        index_presence_sat_old = [0 for idx in range(self.num_orbit)]
        index_presence_sat_new = [0 for idx in range(self.num_orbit)]

        first_values = copy.deepcopy(w_global)
        for key_idx in range(len(first_values)):
            values_key_idx = list(first_values.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            first_values[list(first_values.keys())[key_idx]] = zeros_key_idx

        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS
          gradient_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS
          gradient_matrix_spar.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS
          w_residual_all_users.append(copy.deepcopy(first_values))   #w_global_matrix has the w_global that each satellite has received fron the GS
          w_local_last.append(copy.deepcopy(first_values))
          w_local_new.append(copy.deepcopy(first_values))
          sample_number.append(0)


        self._local_test_on_all_clients(0, x_axes = "round")
        for comm_round in range(self.args.comm_round):

            index_orbit1 = self.visting_orbit_index[comm_round]  #Which satellites will be active in this time
            index_orbit = [index_orbit1]
            index_presence_sat_new[index_orbit1] = 1
            num_sat_each_orbit = int(self.args.client_num_per_round / self.num_orbit)
            w_global_withoutchange = copy.deepcopy(w_global)
            len_group = len(index_orbit)
            satellite_idxes_in_orbit = []
            #print(f"111$$$ w_global = {w_global}")
            for k_orbit in range(len_group):
                num = []
                for j in range(num_sat_each_orbit*index_orbit[k_orbit], num_sat_each_orbit*index_orbit[k_orbit]+num_sat_each_orbit):
                    num.append(j)
                satellite_idxes_in_orbit = satellite_idxes_in_orbit + num

            for i_sat, client2 in enumerate(self.client_list):
                if i_sat in satellite_idxes_in_orbit:

                       w_save_global1 = copy.deepcopy(w_global)
                       model_old = copy.deepcopy(w_local_last[i_sat])
                       #if i_sat == 31:
                       #       print(f"111 model_old = {model_old}")
                       gradients_for_new = copy.deepcopy(gradient_matrix[i_sat])
                       w_global_temp_to_train = copy.deepcopy(w_global_matrix[i_sat])
                       w = client2.train(w_global_temp_to_train, comm_round)
                       model_new = copy.deepcopy(w)
                       #if i_sat == 31:
                       #      print(f"222 model_old = {model_new}")
                       w_local_new[i_sat] = copy.deepcopy(model_new)
                       sample_number[i_sat] = copy.deepcopy(client2.get_sample_number())
                       #print(f"000$$$ w_global = {w_global}")

                       for key_idx in range(len(gradient_matrix[i_sat])):   # % Derive the new gradients (g_k(w_k^{n_k+1,I})
                           values_key_idx1 = list(model_old.values())[key_idx]
                           values_key_idx2 = list(model_new.values())[key_idx]
                           values_key_idx3 = values_key_idx2 - values_key_idx1
                           gradient_matrix[i_sat][list(gradient_matrix[i_sat].keys())[key_idx]] = values_key_idx3
                       #if i_sat == satellite_idxes_in_orbit[0]:
                       #print(f"111gradient_matrix_new[i_sat] == {gradient_matrix_new[i_sat]}")
                       w_wo_sparsification = copy.deepcopy(gradient_matrix[i_sat])
                       w_residual_user = copy.deepcopy(w_residual_all_users[i_sat])
                       w_sparsified, w_residual_new = self._TopK_Threshold_sparsification_all(w_wo_sparsification, w_residual_user, i_sat, self.p)
                       gradient_matrix_spar[i_sat] = copy.deepcopy(w_sparsified)
                       w_residual_all_users[i_sat] = copy.deepcopy(w_residual_new)
                       #if i_sat == 31:
                       #      print(f"000 gradient_matrix_spar[i_sat] = {gradient_matrix_spar[i_sat]}")


                       #if i_sat == 31:


            w_global = copy.deepcopy(w_global_withoutchange)
            #print(f"222$$$ w_global = {w_global}")
            w_global_pass_to_aggregate = copy.deepcopy(w_global)
            gradient_pass_to_aggregate = copy.deepcopy(gradient_matrix_spar)


            w_global = self._aggregate(w_global_pass_to_aggregate, gradient_pass_to_aggregate, satellite_idxes_in_orbit, sample_number, comm_round, index_presence_sat_old, index_presence_sat_new)
            index_presence_sat_old = copy.deepcopy(index_presence_sat_new)
            self.model_trainer.set_model_params(w_global)
            for j_sat, client1 in enumerate(self.client_list):
                if j_sat in satellite_idxes_in_orbit:

                          w_local_last[j_sat] = copy.deepcopy(w_local_new[j_sat])
                          w_global_matrix[j_sat] = copy.deepcopy(w_global)
                          #if j_sat == 31:
                          #      print(f"$$$ w_local_new[j_sat] = {w_local_new[j_sat]}")
                          #      print(f"%%% w_local_last[j_sat] = {w_local_last[j_sat]}")

            if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
               self._local_test_on_all_clients(comm_round+1, x_axes = "round")



    def _aggregate(self, w_global, gradient, satellite_idxes_in_orbit, sample_num, comm_round, index_presence_sat_old, index_presence_sat_new):

                ######### This part is for deriving the sum of sample numbers
                #print(f"111 w_global = {w_global}")
                #print(f"222 delta_gradient = {delta_gradient}")
                #print(f"333 delta_weight = {delta_weight}")

                #for idx2 in range(len(satellite_idxes_in_orbit)):
                #    if satellite_idxes_in_orbit[idx2] == 31:
                #        print(f"444delta_gradient[i_sat] == {delta_gradient[satellite_idxes_in_orbit[idx2]]}")

                training_num = 0
                for idx2 in range(len(sample_num)):
                    if sample_num[idx2] != 0:
                      training_num += (sample_num[idx2])

                first_values = copy.deepcopy(w_global)
                for key_idx in range(len(first_values)):
                    values_key_idx = list(first_values.values())[key_idx]
                    zeros_key_idx = torch.zeros(values_key_idx.shape)
                    first_values[list(first_values.keys())[key_idx]] = zeros_key_idx

                averaged_params_new = copy.deepcopy(first_values)

                for k in averaged_params_new.keys():
                    temp_sum_1 = 0
                    #print(f"********satellite_idxes_in_orbit = {satellite_idxes_in_orbit}")
                    for i1 in satellite_idxes_in_orbit:

                             temp_sum_1 += 1
                             w = sample_num[i1] / training_num

                             if temp_sum_1 == 1:
                                averaged_params_new[k] = gradient[i1][k] * w
                             else:
                                averaged_params_new[k] += gradient[i1][k] * w

                ################### This part is for getting average.
                new_update_global = copy.deepcopy(first_values)
                num_old_non_zero = [1 for i in index_presence_sat_old if i == 1]
                num_new_non_zero = [1 for i in index_presence_sat_new if i == 1]
                new_weight_multiply = (len(num_old_non_zero) ) / (len(num_new_non_zero))
                for k in new_update_global.keys():

                        new_update_global[k] = w_global[k]*new_weight_multiply + averaged_params_new[k]

                #print(f"444 new_update_global = {new_update_global}")
                return new_update_global

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

            #print(f"all_entries_sep_values = {all_entries_sep_values}")
            all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            return all_entries

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
        for j in selected_entries_index:
          values_updated[j] = all_entries[j]

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



        if self.visting_orbit_index == Index_FedISL_Async_Bremen:
                if x_axes=="time":

                  wandb.log({"Test/Acc": test_acc, "time": idx})
                  wandb.log({"Test/Loss": test_loss, "time": idx})
                  self.acc_FedISL.append(test_acc)
                  with open('../../../Results/FedISL_Async_ACCC_Bremenn.py', 'w') as f:
                                 f.write(f'FedISL_Async_ACC_results_Bremen =  {self.acc_FedISL}')

                elif x_axes=="round":
                          wandb.log({"Test/Acc": test_acc, "round": idx})
                          wandb.log({"Test/Loss": test_loss, "round": idx})
                          self.acc_FedISL.append(test_acc)
                          with open('../../../Results/FedISL_Async_ACC_Bremen.py', 'w') as f:
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
