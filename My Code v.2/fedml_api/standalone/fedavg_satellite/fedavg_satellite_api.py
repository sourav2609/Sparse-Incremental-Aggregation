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

class FedAvg_satellite(object):
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

    def train(self):

     for initiall in range(len(self.initial_weights_matrix)):   # For loop to do the learning with multi initial-weights

        # To have the same initial weight for start as set up in the main_fed
        if initiall == 0:
         w_global = self.model_trainer.get_model_params()
        else:
         w_global = self.initial_weights_matrix[initiall]
         self.model_trainer.set_model_params(w_global)


        # Define parameters
        fedavg_satellites_visit_index = np.zeros(self.args.client_num_per_round)
        update_epoch = 0
        update_epoch_training = 0
        client_indexes = range(self.args.client_num_per_round)
        vector_for_initial_weight_control = np.zeros(self.args.client_num_per_round)


        time1 = self.difference_time[0]
        for comm_round in range(self.args.comm_round):

            index_satellites = self.start_visting_satellite_GS_index[update_epoch]  #Which satellites will be active in this time

            #t0 = self.difference_time[0]
            #if update_epoch == 0:
            #    time1 = t0
            #else:
            #    time1 = time1 + self.difference_time[update_epoch]

            if update_epoch == 0 and comm_round == 0:
                self._local_test_on_all_clients(update_epoch, x_axes = "update_epoch")

            for idx1 in range(self.args.client_num_per_round):
                if idx1 == index_satellites:

                 vector_for_initial_weight_control[idx1] = vector_for_initial_weight_control[idx1] + 1
                 update_epoch = update_epoch + 1
                 time1 = time1 + self.difference_time[update_epoch]

                 if fedavg_satellites_visit_index[idx1] == 0:   # This means that satellite had the visiting with GS and then it has sent his weights to the GS
                    fedavg_satellites_visit_index[idx1] = 1


                 if np.sum(fedavg_satellites_visit_index) == self.args.client_num_per_round and np.all(vector_for_initial_weight_control > 1):  #>1 because in the first round, all satellite only receive the weights

                   update_epoch_training = update_epoch_training + 1 # Increase the update_epoch_training because all satellites have sent their updates to the GS
                   fedavg_satellites_visit_index = np.zeros(self.args.client_num_per_round) # To have the zeros matrix for the next round to update them

                   if  (update_epoch_training-1) % 2 == 0:   #  The update will occur 1 time in two full of A vector because one time for sending and second for updating

                    w_locals=[]
                    for idx, client in enumerate(self.client_list):

                        w_global_matrix_pass_train = copy.deepcopy(w_global)
                        w_global_main = copy.deepcopy(w_global)
                        w_new = client.train(w_global_matrix_pass_train, w_global_main)
                        w_global = copy.deepcopy(w_global_main)
                        w_locals.append((client.get_sample_number(), copy.deepcopy(w_new)))


                    w_global = self._aggregate(w_locals)                    #print('w_global'+str(w_global))
                    w_global_current = copy.deepcopy(w_global)
                    self.model_trainer.set_model_params(w_global_current)
                    if (update_epoch_training == 1) or (update_epoch_training %2 ==1):
                       if update_epoch_training == 1:
                         self._local_test_on_all_clients(update_epoch_training, x_axes="update_epoch")
                       else:
                           update_round =  (int(update_epoch_training/2)) +1
                           self._local_test_on_all_clients(update_round, x_axes="update_epoch")


            self._local_test_on_all_clients(time1, x_axes = "time")




    def _aggregate(self, w_locals):
            training_num = 0
            for idx in range(len(w_locals)):
                (sample_num, averaged_params) = w_locals[idx]
                training_num += sample_num

            (sample_num, averaged_params) = w_locals[0]
            for k in averaged_params.keys():
                for i in range(0, len(w_locals)):
                    local_sample_number, local_model_params = w_locals[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w
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
        if x_axes == "time":
          self.test_acc_time.append(test_acc)
          self.test_time.append(idx)
        elif x_axes == "update_epoch":
          self.test_acc_round.append(test_acc)
          self.test_round.append(idx)


        if len(self.test_acc_time) == len(self.start_visting_satellite_GS_index) * len(self.initial_weights_matrix):    #
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
            print('len_each_initial_weight_time'+str(len_each_initial_weight_time))
            print('len_each_initial_weight_round'+str(len_each_initial_weight_round))
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
