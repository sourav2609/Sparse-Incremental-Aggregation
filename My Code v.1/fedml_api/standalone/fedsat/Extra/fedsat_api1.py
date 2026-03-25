
import copy
import logging
import random

import numpy as np
import torch
import wandb
import time

from fedml_api.standalone.fedsat.client import Client
from FedISL_Async_results_Bremen import Time_FedISL_Async_Bremen
#from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_simulation_v02_05 import visibility_satellite_GS_matrix
#from fedml_api.standalone.FedSat_setup.Walker_Delta_two_shells import visibility_satellite_GS_matrix
#from fedml_api.standalone.FedSat_setup.Walker_Delta_two_shells import satellite_GS_matrix_time_difference

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
        self.visting_orbit_index = Time_FedISL_Async_Bremen
        #self.visting_orbit_index = np.array(self.visting_orbit_index)
        self.num_satellite = 40
        self.num_orbit = 5
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
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)


        w_locals = []   # w_local has the w_local of each satellite
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        client_indexes = range(self.args.client_num_per_round)
        matrix_idx = []
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round) # This vector is defined for controlling that each satellite
        update_epoch_for_round_figure = 0
        update_epoch = 0    # Shows how many satellite have seen the GS



        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals
          w_locals.append((0, 0)) # w_local has the w_local of each satellite
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS


        for comm_round in range(self.args.comm_round):

            index_orbit = self.visting_orbit_index[comm_round]  #Which satellites will be active in this time
            #print(f"index_orbit = {index_orbit}")
            #print(f" II index_orbit = {index_orbit}")
            if comm_round == 0:
                self._local_test_on_all_clients(comm_round, x_axes = "comm_round")

            num_sat_each_orbit = int(self.args.client_num_per_round / self.num_orbit)
            satellite_idxes_in_orbit = range(index_orbit * num_sat_each_orbit, index_orbit * num_sat_each_orbit + num_sat_each_orbit)
            #print(f" && satellite_idxes_in_orbit = {satellite_idxes_in_orbit}")
            #x = [0 for jj in range(self.num_orbit)]
            for i_sat, client in enumerate(self.client_list):

                if i_sat in satellite_idxes_in_orbit:
                       #x[index_orbit] = x[index_orbit] + 1
                       w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[i_sat])
                       w_global_matrix_pass_without_train = copy.deepcopy(w_global_matrix)
                       w_global_temp = copy.deepcopy(w_global)

                       w = client.train(w_global_matrix_pass_train, w_global_matrix_pass_train)
                       w_global = copy.deepcopy(w_global_temp)

                       w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                       w_locals[i_sat] = (client.get_sample_number(), copy.deepcopy(w))




            w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
            #print(f" WWW w_locals_pass_to_aggregate = {w_locals_pass_to_aggregate}")
            w_global = self._aggregate(w_locals_pass_to_aggregate)
            #print(f" w_global = {w_global}")
            self.model_trainer.set_model_params(w_global)
            #print(f" GG w_global_matrix before = {w_global_matrix}")
            for j_sat, client in enumerate(self.client_list):
                if j_sat in satellite_idxes_in_orbit:
                          w_global_matrix[j_sat] = copy.deepcopy(w_global)
            #print(f" GG w_global_matrix after = {w_global_matrix}")
            if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
               self._local_test_on_all_clients(comm_round, x_axes = "comm_round")



    # FedavgPlannedAsync
    def _aggregate(self, w_locals):

            ######### This part is for deriving the sum of sample numbers
            training_num = 0
            for idx2 in range(len(w_locals)):
                (sample_num, averaged_params) = w_locals[idx2]
                #print(f"sample_num = {sample_num}")


                if sample_num != 0:
                  training_num += (sample_num)


            #################### This part is for deriving the suitable idx.
            x_temp = 0
            for idx in range(len(w_locals)):

             (sample_num_temp, averaged_params_temp) = w_locals[idx]
             if sample_num_temp != 0 and x_temp == 0:
              x_temp = x_temp + 1

              (sample_num, averaged_params) = w_locals[idx]
              Number_GS_satellite_contact = 1

            ################### This part is for getting average.
            for k in averaged_params.keys():
                augmentation_vector = []
                temp_sum = 0
                for i in range(0, len(w_locals)):

                  (local_sample_number, local_model_params) = w_locals[i]
                  Number_GS_satellite_contact = 1
                  if local_sample_number != 0:
                    temp_sum += 1

                    w = (local_sample_number) / training_num
                    augmentation_vector.append(w)
                    #print(augmentation_vector)
                    if temp_sum == 1:
                        averaged_params[k] = local_model_params[k] * w
                        #print('local_model_params[k]'+str(local_model_params[k].size()))
                    else:
                        averaged_params[k] += local_model_params[k] * w


            #print('averaged_params after = '+str(averaged_params))
            return averaged_params





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

            #print('=== test_data === {}:{} '.format(idx2, self.test_data_local_dict[idx2][0]))

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
        #print('train_loss = ' + str(train_loss))
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])


        #stats_train = {'training_acc': train_acc, 'training_loss': train_loss}
        #logging.info(stats_train)
        stats_test = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats_test)




        if x_axes=="time":
          #self.test_acc_time.append(test_acc)
          #self.test_time.append(idx)

          #wandb.log({"Train/Acc": train_acc, "time": idx})
          #wandb.log({"Train/Loss": train_loss, "time": idx})
          wandb.log({"Test/Acc": test_acc, "time": idx})
          wandb.log({"Test/Loss": test_loss, "time": idx})

        elif x_axes=="comm_round":
                  #self.test_acc_round.append(test_acc)
                  #self.test_round.append(idx)

                  #wandb.log({"Train/Acc": train_acc, "update_epoch": idx})
                  #wandb.log({"Train/Loss": train_loss, "update_epoch": idx})
                  wandb.log({"Test/Acc": test_acc, "comm_round": idx})
                  wandb.log({"Test/Loss": test_loss, "comm_round": idx})
                  self.acc_FedISL.append(test_acc)
                  #print(f"self.acc_FedISL = {self.acc_FedISL}")
                  if idx == self.args.comm_round-1:
                    with open('../../../FedISL_Async_ACC_results.py', 'w') as f:
                         f.write(f'FedISL_Async_ACC_results =  {self.acc_FedISL}')
