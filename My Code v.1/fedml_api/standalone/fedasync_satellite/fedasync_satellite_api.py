import copy
import logging
import random

import numpy as np
#import tensorflow as tf
import torch
import wandb

from fedml_api.standalone.fedavgPlannedAsync.client import Client
#from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_simulation_v02_05 import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import visibility_satellite_GS_matrix
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import satellite_GS_matrix_time_difference
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import satellite_rise_difference_for_fedasync
from fedml_api.standalone.fedavgPlannedAsync.Walker_Delta_two_shells import Period_fedasync

class FedAsyncAPI_satellite(object):    #class FedAvgAPI_satellite(object):
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
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

## This part is the steps for learning in all communication rounds
    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights

            # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
             w_global = self.model_trainer.get_model_params()
        else:
             w_global = self.initial_weights_matrix[initiall]
             self.model_trainer.set_model_params(w_global)

        #print('w_global very first = '+str(w_global))

        #Satellite visiting information

        satellite_rise_time_difference_for_fedasync = satellite_rise_difference_for_fedasync()
        #print('satellite_rise_time_difference_for_fedasync : ' + str(satellite_rise_time_difference_for_fedasync))
        T_o_max = Period_fedasync()
        #print('self.start_visting_satellite_GS_index = ' + str(self.start_visting_satellite_GS_index))
        #print('self.difference_time = '+str(self.difference_time))



        # Derive the number of visiting for each satellite
        Number_GS_satellite_contact = np.zeros(self.args.client_num_per_round)
        for j in range(len(self.start_visting_satellite_GS_index)):
          for i in range(self.args.client_num_per_round):
            if self.start_visting_satellite_GS_index[j] == i:
              Number_GS_satellite_contact[i] = Number_GS_satellite_contact[i]+1




        # define the required matrixes for weights
        update_epoch_for_round_figure = 0
        client_indexes = range(self.args.client_num_per_round)
        logging.info("satellite_indexes = " + str(client_indexes))
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round)
        w_locals = []
        w_global_matrix = []
        update_epoch_client = np.zeros(self.args.client_num_per_round)
        update_epoch = 0
        time_difference_initial = np.zeros(self.args.client_num_per_round)


        for idx1 in range(self.args.client_num_per_round): # Create an initial version of these matrix: w_global_matrix , w_locals
          w_locals.append((update_epoch_client[idx1], time_difference_initial[idx1],  copy.deepcopy(w_global)))
          w_global_matrix.append(copy.deepcopy(w_global))

        for comm_round in range(self.args.comm_round):



            index_satellites = self.start_visting_satellite_GS_index[update_epoch]  #Which satellites will be active in this time
            t0 = self.difference_time[0]
            if update_epoch == 0:
                time1 = t0
            else:
                time1 = time1 + self.difference_time[update_epoch]

            #print('update_epoch = '+str(update_epoch))
            if update_epoch == 0:
               self._local_test_on_all_clients(update_epoch_for_round_figure,  x_axes = "update_epoch")


            for idx, client in enumerate(self.client_list):
                # update dataset
                if index_satellites == idx:
                   vector_for_initial_weight_control[idx] = vector_for_initial_weight_control[idx] + 1
                   update_epoch = update_epoch + 1

                if index_satellites == idx and vector_for_initial_weight_control[idx] > 1:    #This denotes this satellite has started to see the GS

                   update_epoch_for_round_figure = update_epoch_for_round_figure  + 1
                   w_global_matrix_pass_train = copy.deepcopy(w_global_matrix[idx])
                   w_global_matrix_temp = copy.deepcopy(w_global_matrix)
                   w_global_temp = copy.deepcopy(w_global)
                   #print('w_global_matrix_pass_train = ' + str(w_global_matrix_pass_train))
                   w = client.train(w_global_matrix_pass_train)
                   #print('w_local = '+str(w))
                   w_global = copy.deepcopy(w_global_temp)
                   w_global_matrix = copy.deepcopy(w_global_matrix_temp)

                   #print('vector_for_initial_weight_control[idx] = ' + str(vector_for_initial_weight_control[idx]))
                   #print('idx = ' + str(idx) +'satellite_rise_time_difference_for_fedasync[int(vector_for_initial_weight_control[idx])-2] = ' + str(satellite_rise_time_difference_for_fedasync[idx][int(vector_for_initial_weight_control[idx])-2]))
                   w_locals[idx] = (update_epoch_client[idx], satellite_rise_time_difference_for_fedasync[idx][int(vector_for_initial_weight_control[idx])-2],  copy.deepcopy(w))
                   #print('idx = ' + str(idx) + 'satellite_rise_time_difference_for_fedasync[idx][int(vector_for_initial_weight_control[idx])-2] = '+str(satellite_rise_time_difference_for_fedasync[idx][int(vector_for_initial_weight_control[idx])-2]))
                   w_local_one  = copy.deepcopy(w_locals[idx])
                   #print('update_epoch_client' +str(update_epoch_client))
                   #print('staleness' +str(update_epoch_client[idx]-update_epoch))
                   update_epoch_client_one = copy.deepcopy(update_epoch_client[idx])
                   #print("update_epoch_client before = "+ str(update_epoch_client))
                   w_global = self._updater(w_global, w_local_one, update_epoch_client_one, update_epoch, T_o_max)
                   #print('w_global = '+str(w_global))
                   #print('w_global_matrix_before = '+str(w_global_matrix))
                   update_epoch_client[idx] = update_epoch + 1    #update the update_time of this satellite
                   #print("update_epoch_client after = "+ str(update_epoch_client))
                   w_global_current = copy.deepcopy(w_global)
                   w_global_matrix[idx] = copy.deepcopy(w_global)

                   #print('w_global_matrix_after_updater (line 129)= '+str(w_global_matrix))
                   self.model_trainer.set_model_params(w_global_current)

                    # test results
                    # at last round
                   if update_epoch_for_round_figure % self.args.frequency_of_the_test == 0:
                        self._local_test_on_all_clients(update_epoch_for_round_figure, x_axes="update_epoch")




            # test results in terms of rounds
            #if time == self.args.comm_round - 1 and time1>self.difference_time[0]:
            self._local_test_on_all_clients(time1, x_axes = "time")




    def _hinged_staleness(self, time_mapping_staleness, T_o_max):

         epsilon = 1/100
         #T_o_max = 10000000000000000000000 #Without hinged function
         b = (1 + epsilon) * T_o_max
         a = 5 * b
         #a = 5 * b

         if time_mapping_staleness < b:
             s_a_b = 1
         else:
             #print('time_mapping_staleness-b = '+str(time_mapping_staleness-b))
             s_a_b = 1/((a*(time_mapping_staleness-b))+1)
         return s_a_b




    def _updater(self, w_global_update_time_all_satellites, w_local_one, update_epoch_client_one, update_epoch, T_o_max):

            #(update_epoch_client,  w_local) = w_local_one
            (update_epoch_client, satellite_rise_time_difference_for_fedasync_one,  w_local) = w_local_one

            staleness= update_epoch - update_epoch_client_one
            #print("staleness = "+ str(staleness))
            #print('satellite_rise_time_difference_for_fedasync_one = ' + str(satellite_rise_time_difference_for_fedasync_one))
            staleness_hinged_effect = self._hinged_staleness(satellite_rise_time_difference_for_fedasync_one, T_o_max)
            adaptive_mixing_hyperparameter = (self.args.alpha_prim) * staleness_hinged_effect
            #print('adaptive_mixing_hyperparameter' +str(adaptive_mixing_hyperparameter)+'satellite_rise_time_difference_for_fedasync_one = ' + str(satellite_rise_time_difference_for_fedasync_one))
            #print('adaptive_mixing_hyperparameter'+str(adaptive_mixing_hyperparameter))

            #print('w_global: '+str(w_global))
            #print('w_global.keys(): '+str(w_global.keys()))
            for k in w_global_update_time_all_satellites.keys():

                 #print('k: '+str(k))
                 #print('w_global[k] before (line 201)= '+str(w_global[k]))
                 #print('w_local[k] before (line 202)= '+str(w_local[k]))
                 w_global_update_time_all_satellites[k]=((1-adaptive_mixing_hyperparameter)*w_global_update_time_all_satellites[k])+(adaptive_mixing_hyperparameter*w_local[k])
                 #print('w_global[k] after (line 203)= '+str(w_global[k]))

            return w_global_update_time_all_satellites

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
        elif x_axes=="update_epoch":
          self.test_acc_round.append(test_acc)
          self.test_round.append(idx)



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
