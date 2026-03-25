
import copy
import logging
import random

import numpy as np
import torch
import wandb
import collections

from fedml_api.standalone.fedavgPlannedAsync.client import Client
#from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_simulation_v02_05 import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import satellite_GS_matrix_time_difference

class fedavgPlannedAsync_withoutaug(object):
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
        print('lennn = ' + str(len(self.initial_weights_matrix)))
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
        print(f'len(self.initial_weights_matrix) = {len(self.initial_weights_matrix)}')
        # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)


        print(f'len(self.start_visting_satellite_GS_index) = {len(self.start_visting_satellite_GS_index)}')
        # Derive the number of visiting for each satellite
        Number_GS_satellite_contact = np.zeros(self.args.client_num_per_round)
        for j in range(len(self.start_visting_satellite_GS_index)):
          for i in range(self.args.client_num_per_round):
           if self.start_visting_satellite_GS_index[j] == i:
              Number_GS_satellite_contact[i] = Number_GS_satellite_contact[i]+1
        print(f"self.start_visting_satellite_GS_index = {self.start_visting_satellite_GS_index}")

        client_indexes = range(self.args.client_num_per_round)
        #logging.info("#Initial w_global#")
        w_locals = []   # w_local has the w_local of each satellite
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        client_indexes = range(self.args.client_num_per_round)
        matrix_idx = []
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round) # This vector is defined for controlling that each satellite
        update_epoch_for_round_figure = 0
        update_epoch = 0    # Shows how many satellite have seen the GS


        window_size = 20

        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals

          w_locals.append((0, 0,  0)) # w_local has the w_local of each satellite
          w_global_matrix.append(copy.deepcopy(w_global))   #w_global_matrix has the w_global that each satellite has received fron the GS

        time1 = self.difference_time[0]
        vector_counter = np.zeros(10)
        win_idx = 0
        for comm_round in range(self.args.comm_round):

            if update_epoch % window_size == 0 and update_epoch != 0:
                            win_idx = win_idx + 1
                            vector_counter = np.zeros(10)

            index_satellites = self.start_visting_satellite_GS_index[update_epoch]  #Which satellites will be active in this time

            vector_satellite_idxs_window = self.start_visting_satellite_GS_index[win_idx*window_size:(win_idx+1)*window_size]


            sat_visiting_count = collections.Counter(vector_satellite_idxs_window)
            print(list(sat_visiting_count.keys()))
            mean_sat_visiting_count = np.mean(list(sat_visiting_count.values()))
            #
            #for idx_windows in range(len(vector_satellite_idxs_window)):
            for idx_sat in range(10):
                if index_satellites == idx_sat:
                    vector_counter[idx_sat] +=1






            if update_epoch == 0 and comm_round == 0:
                self._local_test_on_all_clients(update_epoch_for_round_figure,  x_axes = "update_epoch")
                #self._local_test_on_all_clients(time1,  x_axes = "time")


            for idx, client in enumerate(self.client_list):

                if index_satellites == idx:
                   vector_for_initial_weight_control[idx] = vector_for_initial_weight_control[idx] + 1
                   update_epoch = update_epoch + 1
                   time1 = time1 + self.difference_time[update_epoch]
                if   vector_counter[idx] <= mean_sat_visiting_count:

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
                   #w = client.train(w_global_matrix_pass_train, w_global_temp)
                   #w = client.train(w_global_matrix_pass_train)
                   w = client.train(w_global_matrix_pass_train, w_global_temp)
                   w_global = copy.deepcopy(w_global_temp)
                   w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                   w_locals[idx] = (client.get_sample_number(), Number_GS_satellite_contact[idx],  copy.deepcopy(w))
                   w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
                   w_global = self._aggregate(w_locals_pass_to_aggregate)
                   print(f"w_global = {w_global}")
                   #print(f"w_locals={w_locals}")
                   #print(f"w_global = {w_global}")

                   self.model_trainer.set_model_params(w_global)
                   w_global_matrix[idx] = copy.deepcopy(w_global)

                   if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                        self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes = "update_epoch")


            #print(f"w_global_matrix = {w_locals}")
            self.model_trainer.set_model_params(w_global)
            self._local_test_on_all_clients(time1, x_axes = "time")

    '''
    #main
    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights
        print(f'len(self.initial_weights_matrix) = {len(self.initial_weights_matrix)}')
        # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)

        #print('w_global very first = '+str(w_global))
        print(f'len(self.start_visting_satellite_GS_index) = {len(self.start_visting_satellite_GS_index)}')
        # Derive the number of visiting for each satellite
        Number_GS_satellite_contact = np.zeros(self.args.client_num_per_round)
        for j in range(len(self.start_visting_satellite_GS_index)):
          for i in range(self.args.client_num_per_round):
           if self.start_visting_satellite_GS_index[j] == i:
              Number_GS_satellite_contact[i] = Number_GS_satellite_contact[i]+1
        #Number_GS_satellite_contact = [5668, 5668, 5668, 5668,5668,7622,7622,7622,7622,7622]
        #Number_GS_satellite_contact # [5. 4. 4. 4. 4. 9. 8. 7. 8. 7.]
        #Number_GS_satellite_contact = [15378. 15382. 15376. 15382. 15374. 29258. 29334. 29302. 29307. 29334.]
        #Number_GS_satellite_contact = [15378, 15378, 15378, 15378, 15378, 29307, 29307, 29307, 29307, 29307]
        #print(" # Number_GS_satellite_contact # "+str(Number_GS_satellite_contact))
        print(f"self.start_visting_satellite_GS_index = {self.start_visting_satellite_GS_index}")

        client_indexes = range(self.args.client_num_per_round)
        #logging.info("#Initial w_global#")
        w_locals = []   # w_local has the w_local of each satellite
        w_global_matrix = [] #w_global_matrix has the w_global that each satellite has received fron the GS
        client_indexes = range(self.args.client_num_per_round)
        matrix_idx = []
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round) # This vector is defined for controlling that each satellite
        update_epoch_for_round_figure = 0
        update_epoch = 0    # Shows how many satellite have seen the GS


        #indexes = []
        #for i in range(10):
        #        indexes.append(list(range(10)))

        #ddd = sum(indexes,[])
        #ddd.sort()
        #self.start_visting_satellite_GS_index = ddd

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
                   #w = client.train(w_global_matrix_pass_train, w_global_temp)
                   #w = client.train(w_global_matrix_pass_train)
                   w = client.train(w_global_matrix_pass_train, w_global_temp)
                   w_global = copy.deepcopy(w_global_temp)
                   w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                   w_locals[idx] = (client.get_sample_number(), Number_GS_satellite_contact[idx],  copy.deepcopy(w))
                   w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
                   w_global = self._aggregate(w_locals_pass_to_aggregate)
                   print(f"w_global = {w_global}")
                   #print(f"w_locals={w_locals}")
                   #print(f"w_global = {w_global}")

                   self.model_trainer.set_model_params(w_global)
                   w_global_matrix[idx] = copy.deepcopy(w_global)

                   if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                        self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes = "update_epoch")


            #print(f"w_global_matrix = {w_locals}")
            self._local_test_on_all_clients(time1, x_axes = "time")
    '''

    '''
    ## Start point collects the weights and then start
    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights

        # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)

        print('w_global very first = '+str(w_global))

        # Derive the number of visiting for each satellite
        Number_GS_satellite_contact = np.zeros(self.args.client_num_per_round)
        for j in range(len(self.start_visting_satellite_GS_index)):
          for i in range(self.args.client_num_per_round):
            if self.start_visting_satellite_GS_index[j] == i:
              Number_GS_satellite_contact[i] = Number_GS_satellite_contact[i]+1

        print(" # Number_GS_satellite_contact # "+str(Number_GS_satellite_contact))



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

        t0 = self.difference_time[0]
        time1 = t0  # time1 shows the time from the start of all satellites process

        #indexes = []
        #for i in range(5):
        #    indexes.append(list(range(10)))

        #ddd = sum(indexes,[])
        #ddd.sort()
        #for ii in range(len(self.start_visting_satellite_GS_index))
        #self.start_visting_satellite_GS_index = ddd
        improve_vec = np.zeros(10)
        gg_check = 0
        print(f"self.start_visting_satellite_GS_index = {self.start_visting_satellite_GS_index}")
        for comm_round in range(self.args.comm_round):


            index_satellites = self.start_visting_satellite_GS_index[update_epoch]  #Which satellites will be active in this time

            #print('update_epoch = '+str(update_epoch))
            if update_epoch == 0 and comm_round == 0:
                self._local_test_on_all_clients(update_epoch_for_round_figure,  x_axes = "update_epoch")
                #self._local_test_on_all_clients(time1,  x_axes = "time")


            for idx, client in enumerate(self.client_list):

                if index_satellites == idx:

                   vector_for_initial_weight_control[idx] = vector_for_initial_weight_control[idx] + 1
                   update_epoch = update_epoch + 1
                   time1 = time1 + self.difference_time[update_epoch]
                   if gg_check ==0 and improve_vec[idx]<=1:
                       improve_vec[idx] = improve_vec[idx] + 1
                   print(f"improve_vec = {improve_vec}")

                # The first visiting is for getting the initial weights from GS and after that for sending locals and receiving global weights
                if index_satellites == idx and vector_for_initial_weight_control[idx] > 1:
                #if index_satellites == idx and vector_for_initial_weight_control[idx] > 1:


                   if improve_vec[idx] == 0 or improve_vec[idx] == 2:
                       update_epoch_for_round_figure = update_epoch_for_round_figure  + 1
                       matrix_idx.append(idx)
                       w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[idx])
                       w_global_matrix_pass_without_train = copy.deepcopy(w_global_matrix)
                       w_global_temp = copy.deepcopy(w_global)
                       w = client.train(w_global_matrix_pass_train, w_global_temp)


                       w_global = copy.deepcopy(w_global_temp)
                       w_global_matrix = copy.deepcopy(w_global_matrix_pass_without_train)
                       w_locals[idx] = (client.get_sample_number(), Number_GS_satellite_contact[idx],  copy.deepcopy(w))
                       w_locals_pass_to_aggregate = copy.deepcopy(w_locals)

                       checck_sum_point1 = 0
                       checck_sum_point2 = 0
                       for iddxx in range(10):
                           if improve_vec[iddxx] == 2:
                               checck_sum_point1 = checck_sum_point1 + 1
                           elif improve_vec[iddxx] == 0:
                               checck_sum_point2 = checck_sum_point2 + 1

                       print(f"checck_sum_point1 = {checck_sum_point1}")
                       print(f"checck_sum_point2 = {checck_sum_point1}")
                       if  checck_sum_point1 == 10:

                         print(f"**********************improve_vec = {improve_vec}")
                         improve_vec = np.zeros(10)


                         w_global = self._aggregate(w_locals_pass_to_aggregate)
                         self.model_trainer.set_model_params(w_global)
                         w_global_matrix[idx] = copy.deepcopy(w_global)
                         gg_check = gg_check + 1

                         if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                            self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes = "update_epoch")

                       elif checck_sum_point2 == 10:
                        print(improve_vec)
                        print('HIiiiiiiiiiiiiiiiiiiiiiii')
                        #w_locals[idx] = (client.get_sample_number(), Number_GS_satellite_contact[idx],  copy.deepcopy(w))
                        #w_locals_pass_to_aggregate = copy.deepcopy(w_locals)
                        w_global = self._aggregate(w_locals_pass_to_aggregate)
                        self.model_trainer.set_model_params(w_global)
                        w_global_matrix[idx] = copy.deepcopy(w_global)
                        if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                           self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes = "update_epoch")


            print(f"w_global =  {w_global}")
            self.model_trainer.set_model_params(w_global)
            self._local_test_on_all_clients(time1, x_axes = "time")
    '''


    # FedavgPlannedAsync
    def _aggregate(self, w_locals):

            ######### This part is for deriving the sum of sample numbers
            training_num = 0
            for idx2 in range(len(w_locals)):
                (sample_num, Number_GS_satellite_contact,  averaged_params) = w_locals[idx2]

                if sample_num != 0:
                  Number_GS_satellite_contact = 1
                  training_num += (sample_num/Number_GS_satellite_contact)


            #################### This part is for deriving the suitable idx.
            x_temp = 0
            for idx in range(len(w_locals)):

             (sample_num_temp, Number_GS_satellite_contact_temp,  averaged_params_temp) = w_locals[idx]
             if sample_num_temp != 0 and x_temp == 0:
              x_temp = x_temp + 1

              (sample_num, Number_GS_satellite_contact, averaged_params) = w_locals[idx]
              Number_GS_satellite_contact = 1

            ################### This part is for getting average.
            for k in averaged_params.keys():
                augmentation_vector = []
                temp_sum = 0
                for i in range(0, len(w_locals)):

                  (local_sample_number, Number_GS_satellite_contact, local_model_params) = w_locals[i]
                  Number_GS_satellite_contact = 1
                  if local_sample_number != 0:
                    temp_sum += 1

                    w = (local_sample_number/Number_GS_satellite_contact) / training_num
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
        print('x_axes = ' + str(x_axes))
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
        print('train_loss = ' + str(train_loss))
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
        '''
        elif x_axes=="round":
          self.test_acc_round.append(test_acc)
          self.test_round.append(idx)

          wandb.log({"Train/Acc": train_acc, "round": idx})
          wandb.log({"Train/Loss": train_loss, "round": idx})
          wandb.log({"Test/Acc": test_acc, "round": idx})
          wandb.log({"Test/Loss": test_loss, "round": idx})
        '''


        #stats = {'training_acc': train_acc, 'training_loss': train_loss}
        #wandb.log({"Train/Acc": train_acc, "round": round_idx})
        #wandb.log({"Train/Loss": train_loss, "round": round_idx})
        #logging.info(stats)

        #stats = {'test_acc': test_acc, 'test_loss': test_loss}
        #wandb.log({"Test/Acc": test_acc, "round": round_idx})
        #wandb.log({"Test/Loss": test_loss, "round": round_idx})
        #logging.info(stats)





        '''
        if len(self.test_acc_time) == len(self.start_visting_satellite_GS_index) * len(self.initial_weights_matrix):
            print('self.test_acc_time: ' + str(self.test_acc_time))
            print('self.test_acc_round: '  + str(self.test_acc_round))
            print('self.test_time: '  + str(self.test_time))
            print('self.test_round: '  + str(self.test_round))
            len_time = int(len(self.test_time) / len(self.initial_weights_matrix))
            len_round = int(len(self.test_round) / len(self.initial_weights_matrix))
            time_idx = self.test_time[0:len_time]
            round_idx = self.test_round[0:len_round]

            len_each_initial_weight_time = int(len(self.test_acc_time) / len(self.initial_weights_matrix))
            len_each_initial_weight_round = int(len(self.test_acc_round) / len(self.initial_weights_matrix))

            test_acc_time_all_initialize_weights = []
            test_acc_round_all_initialize_weights = []
            for idx_weight in range(len(self.initial_weights_matrix)):
                test_acc_time_all_initialize_weights.append(self.test_acc_time[idx_weight*len_each_initial_weight_time: idx_weight*len_each_initial_weight_time+len_each_initial_weight_time])
                test_acc_round_all_initialize_weights.append(self.test_acc_round[idx_weight*len_each_initial_weight_round: idx_weight*len_each_initial_weight_time+len_each_initial_weight_round])


            dict_acc_round = {}
            dict_acc_time = {}


            for i in range(len(test_acc_round_all_initialize_weights)):
              dict_acc_round['initial_round_'+str(i)] = {'acc_round': 0 , 'round': 0}
            dict_acc_round['average_initial_weights_round'] = {'acc_round': 0 , 'round': 0}



            for i in range(len(test_acc_time_all_initialize_weights)):
              dict_acc_time['initial_time_'+str(i)] = {'acc_time': 0 , 'time': 0}
            dict_acc_time['average_initial_weights_time'] = {'acc_time': 0 , 'time': 0}



            for i in range(len(test_acc_round_all_initialize_weights[0])):  #for loop for round considering
                for j in range(len(test_acc_round_all_initialize_weights)):   #for loop for the umber of initial weights

                   dict_acc_round['initial_round_'+str(j)]['acc_round'] = test_acc_round_all_initialize_weights[j][i]
                   dict_acc_round['initial_round_'+str(j)]['round'] = round_idx[i]
                average_vector_round = []
                for j in range(len(test_acc_round_all_initialize_weights)):
                     average_vector_round.append(dict_acc_round['initial_round_'+str(j)]['acc_round'])
                dict_acc_round['average_initial_weights_round']['acc_round'] = sum(average_vector_round)/len(average_vector_round)
                dict_acc_round['average_initial_weights_round']['round'] = round_idx[i]

                print(dict_acc_round)
                wandb.log(dict_acc_round)
                   #print('dict_acc_round[list(dict_acc_round.keys())[j]] = ' + str(dict_acc_round[list(dict_acc_round.keys())[j]]))
                   #dict_acc_round[list(dict_acc_round.keys())[i]][list(list(dict_acc_round.values())[0].keys())[0]] = test_acc_round_all_initialize_weights[j][i]
                   #dict_acc_round[list(dict_acc_round.keys())[i]][list(list(dict_acc_round.values())[0].keys())[1]] = round_idx[i]



            print('print(dict_acc_round) = ' + str(dict_acc_round))
            for i in range(len(test_acc_time_all_initialize_weights[0])):  #for loop for round considering
                for j in range(len(test_acc_time_all_initialize_weights)):   #for loop for the umber of initial weights

                   dict_acc_time['initial_time_'+str(j)]['acc_time'] = test_acc_time_all_initialize_weights[j][i]
                   dict_acc_time['initial_time_'+str(j)]['time'] = time_idx[i]
                average_vector_time = []
                for j in range(len(test_acc_time_all_initialize_weights)):
                     average_vector_time.append(dict_acc_time['initial_time_'+str(j)]['acc_time'])
                dict_acc_time['average_initial_weights_time']['acc_time'] = sum(average_vector_time)/len(average_vector_time)
                dict_acc_time['average_initial_weights_time']['time'] = time_idx[i]

                print(dict_acc_time)
                wandb.log(dict_acc_time)

                   #dict_acc_time[list(dict_acc_time.keys())[i]][list(list(dict_acc_time.values())[0].keys())[0]] = test_acc_time_all_initialize_weights[j][i]
                   #dict_acc_time[list(dict_acc_time.keys())[i]][list(list(dict_acc_time.values())[0].keys())[1]] = time_idx[i]
             '''



        '''
        for ii in range(3):
         if x_axes=="time" and ii == initiall:
            wandb.log({"Train/Acc": train_acc, "satellites_GS_time (seconds)": idx})
            wandb.log({"Train/Loss": train_loss, "satellites_GS_time (seconds)": idx})
            wandb.log({"Test/Acc": test_acc, "ii": ii, "satellites_GS_time (seconds)": idx})
            print("=== time  === {}: {}".format(idx, test_acc))
            wandb.log({"Test/Loss": test_loss, "satellites_GS_time (seconds)": idx})
         elif x_axes=="update_epoch" and ii == initiall:
            wandb.log({"Train/Acc": train_acc, "round": idx})
            wandb.log({"Train/Loss": train_loss, "round": idx})
            wandb.log({"Test/Acc": test_acc, "ii": ii, "round": idx})
            print("=== round === {}: {}".format(idx, test_acc))
            wandb.log({"Test/Loss": test_loss, "ii": ii, "round": idx})
        '''
