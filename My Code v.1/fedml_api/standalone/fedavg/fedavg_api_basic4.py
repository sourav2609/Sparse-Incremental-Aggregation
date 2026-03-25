import copy
import logging
import random
import operator
import numpy as np
import torch
import wandb
import time
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=10)
from fedml_api.standalone.fedavg.client import Client


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
        self.acc_FedISL = []
        self.model_trainer = model_trainer
        self.p = 1
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round): 
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            #print(f"%%%%%%%% = {np.array(train_data_local_num_dict[client_idx]).itemsize}")
            self.client_list.append(c)
            #print('train_data_local_num_dict[client_idx] = ' + str(self.train_data_local_num_dict))
        #logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        #print('w_global_first:'+str(w_global))
        ### This for-loop makes a zero-values dictionary with the same size of global parameters.
        residual = copy.deepcopy(w_global)
        for key_idx in range(len(residual)):
            values_key_idx = list(residual.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            residual[list(residual.keys())[key_idx]] = zeros_key_idx


        w_residual_all_users = [residual for i in range(self.args.client_num_per_round)]    # Makes a vector for residual for all users #S tthis repeates residual for self.agrs.client_num_per_round

        round_idx = 0
        self._local_test_on_all_clients(round_idx)                              #S for each round run tests on all clients

        for round_idx in range(1, self.args.comm_round):                        #S Federated Learning: Global round

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []                                                       #S for storing the local weights/parameters
            w_locals_gradients = []                                             #S for storing the local gradients

            client_indexes = range(self.args.client_num_per_round)              #S create indices, i.e., 0,1,...,self.args.client_num_per_round
            logging.info("client_indexes = " + str(client_indexes))
            #print('w_global before = '+str(w_global))
            w_global_temp = copy.deepcopy(w_global)                             #S in the running of the code w_global may change as observed by Nasrin, that's why store and retrieve this important information
            for idx, client in enumerate(self.client_list):                     #S for each client run this loop

               
                model_old = copy.deepcopy(w_global_temp)
                gradients = copy.deepcopy(w_global_temp)                        #S for every client its changing
                w_global_temp_to_train = copy.deepcopy(w_global)                
                w = client.train(w_global_temp_to_train, round_idx)             #S to train set_parameters (global) -> train -> get_parameters 
                model_new = copy.deepcopy(w)
                
                #print(f"^%$$$ idx = {idx}")
                for key_idx in range(len(gradients)):                           #S  Derive the gradients

                   values_key_idx1 = list(model_old.values())[key_idx]
                   values_key_idx2 = list(model_new.values())[key_idx]
                   values_key_idx3 = values_key_idx2 - values_key_idx1
                   gradients[list(gradients.keys())[key_idx]] = values_key_idx3
                   #print(f"client")
                print(f"in basic 3-with TopK sparsification")
                w_wo_sparsification = copy.deepcopy(gradients)                  #S original gradients
                w_residual_user = copy.deepcopy(w_residual_all_users[idx])
                w_sparsified, w_residual_new = self._sparsificationAlgorithm('TopK', w_wo_sparsification, w_residual_user, idx, self.p)   #S algoName to be decided based on desired algorithm
                
                w_residual_all_users[idx] = copy.deepcopy(w_residual_new) # Renew the residual values
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_sparsified)))
                print(f"w_sparsified = {w_sparsified}")
                #w_locals_gradients.append((client.get_sample_number(), copy.deepcopy(w_sparsified))) #S store the gradients
                #w_locals.append((client.get_sample_number(), copy.deepcopy(w))) #S append the weights in w_locals for aggregating in what way? stacking?, what is meant by sample_number?

            #w_global = self._aggregate(w_locals) 
            w_global = copy.deepcopy(w_global_temp)                             #S aggregate/average the values to obtain new global value?
            w_global = self._aggregate_gradients(w_locals, w_global)
            self.model_trainer.set_model_params(w_global)                       #S we are setting the model with new global value at the PS to check the testing accuracy?

            self._local_test_on_all_clients(round_idx)                          #S check test accuracy, etc parameters for the training accuracy at a given global iterations







    def _aggregate(self, w_locals):             #S This function implements the aggregation stratagy in the Parameter Server (PS)
        #print(f"w_locals = {w_locals}")        #S would like to understand in terms of formula?
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        #print('averaged_params: '+str(averaged_params))
        #print('averaged_params.keys(): '+str(averaged_params.keys()))
        for k in averaged_params.keys():        #S .keys() returns the labels for the weights, i.e., averaged _params?
            #print('k: '+str(k))
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_gradients(self, w_locals_gradients, w_global):             #S This function implements the aggregation stratagy in the Parameter Server (PS) with gradients
        #print(f"w_locals = {w_locals}")                                      #S would like to understand in terms of formula?
        training_num = 0
        for idx in range(len(w_locals_gradients)):
            (sample_num, averaged_params) = w_locals_gradients[idx]
            training_num = training_num + sample_num

        (sample_num, averaged_params) = w_locals_gradients[0]
        #print('averaged_params: '+str(averaged_params))
        #print('averaged_params.keys(): '+str(averaged_params.keys()))
        for k in averaged_params.keys():        #S .keys() returns the labels for the weights, i.e., averaged _params?
            #print('k: '+str(k))
            for i in range(0, len(w_locals_gradients)):
                local_sample_number, local_model_params = w_locals_gradients[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        for key in range(0, len(averaged_params)):
            w_global[list(w_global.keys())[key]] = w_global[list(w_global.keys())[key]] + averaged_params[list(w_global.keys())[key]]              #S check this very carefully
        return w_global



    def _local_test_on_all_clients(self, round_idx):    #S to test the metrics of training

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

        for idx2, client2 in enumerate(self.client_list):     #S self.client_list contains the set of all clients
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

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        self.acc_FedISL.append(test_acc)
        logging.info(stats)




    def _sparsificationAlgorithm(self, algoName, w_wo_sparsification, w_residual_user, idx, p):  #S a general sparsification algorthim
            
            if algoName == 'TopK':
                w_sparsification1 = copy.deepcopy(w_wo_sparsification)
                w_sparsification2 = copy.deepcopy(w_wo_sparsification)
                
                operator1 = operator.add
                w_sum_train_residual = self.sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum resifual and new receive global model
                all_entries = self.change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
                #print(f"&&&& len(all_entries) = {len(all_entries)}")                                                                 # % For Examle: w_sum_train_residual = OrderedDict([('linear', tensor([ 1,  5, 13])), ('weight', tensor([12, 14]))])
                                                                            # % all_entries = tensor([ 1.,  5., 13., 12., 14.])
                print(f"in TopK: p = {p}")
                w_sparsification2 = self.Spar_Threshold(w_sum_train_residual, all_entries, p)   # Makes the received weights global and in the form of global matrix

                operator2 = operator.sub
                w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
                w_residual_new = self.sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsification2, operator2) # % sum resifual and new receive global model
                #print(f"In the code")
                return w_sparsification2, w_residual_new
            
            if algoName == 'RE_Sparse':
                 print(f"in RE")
                 return 0


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
        print(f"values_updated = {values_updated}")
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






        
