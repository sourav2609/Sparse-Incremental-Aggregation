
import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedavgPlannedAsync.client import Client
#from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_simulation_v02_05 import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import satellite_GS_matrix_time_difference

class FedAvgPlannedAsync(object):
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
        self.start_visting_satellite_GS_index = visibility_satellite_GS_matrix()
        self.difference_time = satellite_GS_matrix_time_difference()

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx0 in range(self.args.client_num_per_round):
            c = Client(client_idx0, train_data_local_dict[client_idx0], test_data_local_dict[client_idx0],
                       train_data_local_num_dict[client_idx0], self.args, self.device, model_trainer)
            self.client_list.append(c)
        #print('self.client_list = '+str(self.client_list))
        logging.info("############setup_clients (END)#############")


    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights

        # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)

        #print('w_global very first = '+str(w_global))
        #print(f'len(self.start_visting_satellite_GS_index) = {len(self.start_visting_satellite_GS_index)}')
        # Derive the number of visiting for each satellite
        Number_GS_satellite_contact = np.zeros(self.args.client_num_per_round)
        for j in range(len(self.start_visting_satellite_GS_index)):
          for i in range(self.args.client_num_per_round):
           if self.start_visting_satellite_GS_index[j] == i:
              Number_GS_satellite_contact[i] = Number_GS_satellite_contact[i]+1
        #Number_GS_satellite_contact = [5668, 5668, 5668, 5668,5668,7622,7622,7622,7622,7622]
        #Number_GS_satellite_contact # [5. 4. 4. 4. 4. 9. 8. 7. 8. 7.]
        #Number_GS_satellite_contact = [15378. 15382. 15376. 15382. 15374. 29258. 29334. 29302. 29307. 29334.]
        ##Number_GS_satellite_contact = [15378, 15378, 15378, 15378, 15378, 29307, 29307, 29307, 29307, 29307]
        #print(" # Number_GS_satellite_contact # "+str(Number_GS_satellite_contact))


        client_indexes = range(self.args.client_num_per_round)
        #logging.info("#Initial w_global#")
        w_locals = []   # w_local has the w_local of each satellite
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        client_indexes = range(self.args.client_num_per_round)
        matrix_idx = []
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round) # This vector is defined for controlling that each satellite
        update_epoch_for_round_figure = 0
        update_epoch = 0    # Shows how many satellite have seen the GS

        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals

          w_locals.append((0, 0,  0)) # w_local has the w_local of each satellite
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS

        time1 = self.difference_time[0]
        for comm_round in range(self.args.comm_round):


            index_satellites = self.start_visting_satellite_GS_index[update_epoch]  #Which satellites will be active in this time


            if update_epoch == 0 and comm_round == 0:
                self._local_test_on_all_clients(update_epoch_for_round_figure,  x_axes = "update_epoch")
                #self._local_test_on_all_clients(time1,  x_axes = "time")


            for idx, client in enumerate(self.client_list):

                if index_satellites == idx:
                   vector_for_initial_weight_control[idx] = vector_for_initial_weight_control[idx] + 1
                   update_epoch = update_epoch + 1
                   time1 = time1 + self.difference_time[update_epoch]
                # The first visiting is for getting the initial weights from GS and after that for sending locals and receiving global weights
                if index_satellites == idx and vector_for_initial_weight_control[idx] > 1:
                   #print('idx = '+str(idx))
                   update_epoch_for_round_figure = update_epoch_for_round_figure  + 1
                   #print('idx = ' + str(idx) + ' vector_for_initial_weight_control[idx] = ' + str(vector_for_initial_weight_control))
                   matrix_idx.append(idx)
                   w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[idx])

                   w_global_matrix_pass_without_train = copy.deepcopy(w_global_matrix)
                   w_global_temp = copy.deepcopy(w_global)
                   #print('w_global_matrix_pass_train = '+str(w_global_matrix_pass_train))
                   w = client.train(w_global_matrix_pass_train, w_global_temp)

                   w_global = copy.deepcopy(w_global_temp)
                   w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                   w_locals[idx] = (client.get_sample_number(), Number_GS_satellite_contact[idx],  copy.deepcopy(w))
                   w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
                   #for param in w_global.parameters():
                    #           print(param.grad)
                   #print(f"dir(w_global) = {w_global.parameters()}")
                   w_global = self._aggregate(w_locals_pass_to_aggregate)
                   #print(f"w_locals={w_locals}")
                   #print(f"w_global = {w_global}")

                   self.model_trainer.set_model_params(w_global)
                   w_global_matrix[idx] = copy.deepcopy(w_global)

                   if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                        self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes = "update_epoch")


            #print(f"w_global_matrix = {w_locals}")
            self._local_test_on_all_clients(time1, x_axes = "time")




    # FedavgPlannedAsync
    def _aggregate(self, w_locals):

            training_num = 0
            for idx2 in range(len(w_locals)):
                (sample_num, Number_GS_satellite_contact,  averaged_params) = w_locals[idx2]

                if sample_num != 0:
                  training_num += (sample_num)

            #print('training_num = '+str(training_num))
            x_temp = 0
            for idx in range(len(w_locals)):

             (sample_num_temp, Number_GS_satellite_contact_temp,  averaged_params_temp) = w_locals[idx]
             if sample_num_temp != 0 and x_temp == 0:
              x_temp = x_temp + 1

              (sample_num, Number_GS_satellite_contact, averaged_params) = w_locals[idx]


            time_period_weight = 0
            number_participated_satellite = 0
            for idx_sat in range(0, len(w_locals)):
              (local_sample_number, Number_GS_satellite_contact, local_model_params) = w_locals[idx_sat]
              if local_sample_number != 0:
                   number_participated_satellite += 1
                   time_period_weight += (Number_GS_satellite_contact)



            time_period_weight = (time_period_weight/number_participated_satellite) #(1/T1+1/T2)/K
            for k in averaged_params.keys():
                augmentation_vector = []
                temp_sum = 0
                for i in range(0, len(w_locals)):

                  (local_sample_number, Number_GS_satellite_contact, local_model_params) = w_locals[i]
                  if local_sample_number != 0:
                    temp_sum += 1

                    #w = (local_sample_number/Number_GS_satellite_contact) / training_num   %For main fedsat+
                    w = (local_sample_number*time_period_weight) / (training_num*(Number_GS_satellite_contact))
                    #w = (local_sample_number*time_period_weight) / (training_num*(1/Number_GS_satellite_contact))
                    augmentation_vector.append(w)
                    if temp_sum == 1:
                        averaged_params[k] = local_model_params[k] * w
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
            train_local_metrics = client2.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))


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
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        #print('train_loss = ' + str(train_loss))
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats_train = {'training_acc': train_acc, 'training_loss': train_loss}
        logging.info(stats_train)
        stats_test = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats_test)




        if x_axes=="time":
          self.test_acc_time.append(test_acc)
          self.test_time.append(idx)

          wandb.log({"Train/Acc": train_acc, "time": idx})
          wandb.log({"Train/Loss": train_loss, "time": idx})
          wandb.log({"Test/Acc": test_acc, "time": idx})
          wandb.log({"Test/Loss": test_loss, "time": idx})
        elif x_axes=="update_epoch":
                  self.test_acc_round.append(test_acc)
                  self.test_round.append(idx)

                  wandb.log({"Train/Acc": train_acc, "update_epoch": idx})
                  wandb.log({"Train/Loss": train_loss, "update_epoch": idx})
                  wandb.log({"Test/Acc": test_acc, "update_epoch": idx})
                  wandb.log({"Test/Loss": test_loss, "update_epoch": idx})
