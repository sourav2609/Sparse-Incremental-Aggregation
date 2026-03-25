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
        self.p = args.spar_ratio                        # This denotes the sparsification ratio
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round): 
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
            #print('train_data_local_num_dict[client_idx] = ' + str(self.train_data_local_num_dict))
        #logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        #print('w_global_first:'+str(w_global))
        ### This for-loop makes a zero-values dictionary with the same size of global parameters.
        residual = copy.deepcopy(w_global)    #Formatting
        for key_idx in range(len(residual)):
            values_key_idx = list(residual.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            residual[list(residual.keys())[key_idx]] = zeros_key_idx
        w_residual_all_users = [residual for i in range(self.args.client_num_per_round)]    # Makes a vector for residual for all users #S tthis repeates residual for self.agrs.client_num_per_round
        #print(f"w_global = {w_global.keys()}")
        round_idx = 0
        print(f"Algo: Proposed Algorithm")

        # Calculating K value in Top-K for Sparsification
        size_model = torch.numel(self.change_weight_vector(w_global))  #change_weight_vector appends and flatten the dict into an array and numel spits out size of array
        K = int(self.p*size_model)  #Let's say fixing the budget for each client for CL-SIA

        print(f"K value = {K}")

        self._local_test_on_all_clients(round_idx,  [0], self.args.client_num_per_round, 0, 0)                              #For each round run tests on all clients
        commCostVec = []
        normCostVec = []
        for round_idx in range(1, self.args.comm_round):                        #Global round
            nonZerosOfAggregate = []                                          # Varible for storing ||\gamma_k||_0 in the notes
            commCost = 0
            data_trans_norm = 0                #Container for normalized data
            norm_val = 0                       #The value with which we should normalize
            
            logging.info("################Communication round : {}".format(round_idx))

            total_sample_num = 0                                               #This calculates the total sample for averaging at the parameter server
          

            client_indexes = range(self.args.client_num_per_round)              #Create indices, i.e., 0,1,...,self.args.client_num_per_round
            logging.info("client_indexes = " + str(client_indexes))
            #print('w_global before = '+str(w_global))
            w_global_temp = copy.deepcopy(w_global)                             #In the running of the code w_global may change as observed by Nasrin, that's why store and retrieve this important information       
            for idx, client in enumerate(self.client_list):                     #For each client run this loop

               
                model_old = copy.deepcopy(w_global_temp)
                gradients = copy.deepcopy(w_global_temp)                        #For every client its changing
                w_global_temp_to_train = copy.deepcopy(w_global)                
                w = client.train(w_global_temp_to_train, round_idx)             #Get local parameters
                model_new = copy.deepcopy(w)                                    #To train set_parameters (global) -> train -> get_parameters 
                if idx == 0:                                                    #Only for the first client
                    aggregated_value = copy.deepcopy(w)                         #Formatting For storing aggregated values, which is independent for each round

                for key_idx in range(len(gradients)):                           #Derive the gradients

                   values_key_idx1 = list(model_old.values())[key_idx]
                   values_key_idx2 = list(model_new.values())[key_idx]
                   values_key_idx3 = values_key_idx2 - values_key_idx1
                   gradients[list(gradients.keys())[key_idx]] = client.get_sample_number()*values_key_idx3
                   #gradients[list(gradients.keys())[key_idx]] = values_key_idx3
                   if idx == 0:
                    aggregated_value[list(aggregated_value.keys())[key_idx]] = torch.zeros(aggregated_value[list(aggregated_value.keys())[key_idx]].size())  #For the first client: Make the tensor with zero values



                w_residual_user = copy.deepcopy(w_residual_all_users[idx])      #Obtain the initial residual

                ## Error Feedback || Error-compensated gradients <-- Gradients + error
                gradients_tilde = copy.deepcopy(aggregated_value) #Formatting
                for key in range(len(gradients_tilde)):
                     gradients_tilde[list(gradients_tilde.keys())[key]] = list(gradients.values())[key] + list(w_residual_user.values())[key]  

                ## Incremental aggregation || Aggregated value <-- Error-compensated gradients + Aggregated value
                for key in range(len(aggregated_value)):
                     aggregated_value[list(aggregated_value.keys())[key]] = list(gradients_tilde.values())[key] + list(aggregated_value.values())[key] 

                ##  Sparsification || Proposed sparsification methods
                aggregated_value_with_sparsification = copy.deepcopy(gradients_tilde) #Formatting
                aggregated_value_with_sparsification = self.sparsifyMain(aggregated_value, K, 'Proposed')
                #print(f"nonzeros after spars. = {self.countNonZeroValues(aggregated_value_with_sparsification)}")

                ## Update error
                w_residual_new = copy.deepcopy(w_residual_user)
                for key in range(len(w_residual_user)):
                     w_residual_new[list(w_residual_new.keys())[key]] = list(aggregated_value.values())[key] - list(aggregated_value_with_sparsification.values())[key]
                w_residual_all_users[idx] = copy.deepcopy(w_residual_new)       # Renew the residual values
                total_sample_num = total_sample_num + client.get_sample_number()   

                ## Renew the aggregated values
                for key in range(len(aggregated_value)):
                    aggregated_value[list(aggregated_value.keys())[key]] = list(aggregated_value_with_sparsification.values())[key]   
                ## Calculation for data transmission
                tmp = self.countNonZeroValues(aggregated_value)
                #print(f"num non-zero (should be 78.6) = {tmp}")
                if idx == 0:   #For the first client
                     norm_val = tmp*(13+32)                

                commCost = commCost + (tmp*13 + tmp*32)    ## ceil(log2(d)) = 13 and \omega = 32  
                #print(f"values = {(tmp*13 + tmp*32)}")
                data_trans_norm = data_trans_norm + ((tmp*13 + tmp*32) )/norm_val
                #print(f"type of temp = {type(tmp)}")
                ## Calculation of non zero values of aggregate for each clients
                nonZerosOfAggregate.append(tmp)
                #print(f"nonZerosOfAggregate = {nonZerosOfAggregate}")
               

            # Till here all clients has sparsified and aggregated the values, now Parameter Server will add them to obtain the final w_global for the next round
            w_global = copy.deepcopy(w_global_temp)                             # Obtain the global parameters
            w_global =  self.parameterServer(w_global, aggregated_value, total_sample_num) # Update the w_global at the parameter server
            self.model_trainer.set_model_params(w_global)                       # We are setting the model with new global value at the PS to check the testing accuracy?
            #print(f"data_trans_norm = {data_trans_norm}")
            commCostVec.append(commCost)
            normCostVec.append(data_trans_norm)

            self._local_test_on_all_clients(round_idx, nonZerosOfAggregate, self.args.client_num_per_round, commCost, data_trans_norm)                          #Check test accuracy, etc parameters for the training accuracy at a given global iterations

        
    def sparsifyAndAggregate(self, gradients, w_residual_user, sample_number, aggregated_value):
        #Error Feedback
        gradients_tilde = copy.deepcopy(gradients)  #Formatting
        for key in range(len(gradients)):
            val_idx1 = list(gradients.values())[key]
            val_idx2 = list(w_residual_user.values())[key]
            val_idx3 = sample_number*val_idx1 + val_idx2
            gradients_tilde[list(gradients_tilde.keys())[key]] = val_idx3

        ###Incoming Sparsification Mask
        incomingMask = self.getMask(aggregated_value)    #Get mask from aggregated value

        #Sparsification of accumulated gradients and local gradients
        gradients_sparsified = self.sparsifyMain(gradients_tilde, K)   #This function implements sparsification

        ###Local sparsification mask
        localMask = self.getMask(gradients_sparsified)
        finalMask = self.addMask(incomingMask, localMask) #Obtain final mask

        ###Sparsify
        finalSparsifiedGradient = self.Hadamard(gradients_tilde, finalMask)

        ###Update error
        w_residual_user_new = copy.deepcopy(w_residual_user) #Formatting
        for key in range(len(w_residual_user_new)):
            w_residual_user_new[list(w_residual_user_new.keys())[key]] = list(gradients_tilde.values())[key] - list(finalSparsifiedGradient.values())[key]

        ###Incremental aggregation IA
        aggregated_value_new = copy.deepcopy(aggregated_value) #Formatting
        for key in range(len(aggregated_value_new)):
            aggregated_value_new[list(aggregated_value_new.keys())[key]] = list(finalSparsifiedGradient.values())[key] + list(aggregated_value.values())[key]

        return aggregated_value_new, w_residual_user_new

    def parameterServer(self, w_global, aggregated_value, total_sample_num):    #This function does the job of Parameter Server
        w_global_new = copy.deepcopy(w_global)                                  #This is just for formatting
        for key_idx in range(len(w_global)):                                    #This just adds
            values_key_idx1 = list(w_global.values())[key_idx]
            values_key_idx2 = list(aggregated_value.values())[key_idx]
            values_key_idx3 = values_key_idx1 + (1/total_sample_num)*values_key_idx2
            w_global_new[list(w_global_new.keys())[key_idx]] = values_key_idx3
        return w_global_new

    def getMask(self, incomingValue):
        mask = copy.deepcopy(incomingValue)   #Formatting
        for key in range(len(mask)):
            #print(f"list(incomingValue.values())[key] ={list(incomingValue.values())[key]}")
            temp = torch.zeros(mask[list(mask.keys())[key]].size())                       #This is tensor of zeros which will help in comparing with incomingValue to generate boolean True/False
            mask_boolean = torch.eq(incomingValue[list(incomingValue.keys())[key]], temp) #Generate boolean value
            mask[list(mask.keys())[key]] = torch.where(mask_boolean, 0, 1)                #False --> 1 and True--> 0
        return mask
        
    def addMask(self, mask1, mask2):        #XOR addition of mask
        newMask = copy.deepcopy(mask1)   #Formatting
        for key in range(len(newMask)):
            val1 = list(mask1.values())[key] > 0  #These are tensors make them boolean
            val2 = list(mask2.values())[key] > 0
            logical_or = torch.logical_or(val1, val2)
            newMask[list(newMask.keys())[key]] = torch.where(logical_or, 1, 0)
        return newMask        

    def Hadamard(self, val_1, mask):
        newValue = copy.deepcopy(val_1)   #Formatting: Let the keys be same as the first input
        for key in range(len(newValue)):
            newValue[list(newValue.keys())[key]] = torch.mul(list(val_1.values())[key], list(mask.values())[key])
        return newValue
    
    def sparsifyMain(self, gradients, K, algoName):   #Put the sparfication algorithm here
        #Make the residual as all zeros as it is not required for now
        w_residual_user = copy.deepcopy(gradients)     #Formatting
        for key in range(len(w_residual_user)):
             w_residual_user[list(w_residual_user.keys())[key]] = torch.zeros(w_residual_user[list(w_residual_user.keys())[key]].size())    #Make every entry of residual as zero  #This needs to be a tensor of zeros

        w_sparsified = self._sparsificationAlgorithm(algoName, gradients, w_residual_user, K) 
        return w_sparsified
    




    def _sparsificationAlgorithm(self, algoName, w_wo_sparsification, w_residual_user, K):  #Sparsification algorthim
            
            if algoName == 'TopK':
                w_sparsification1 = copy.deepcopy(w_wo_sparsification)
                w_sparsification2 = copy.deepcopy(w_wo_sparsification)
                
                operator1 = operator.add
                w_sum_train_residual = self.sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum resifual and new receive global model
                all_entries = self.change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
                w_sparsification2 = self.Spar_Threshold(w_sum_train_residual, all_entries, K)   # Makes the received weights global and in the form of global matrix

                operator2 = operator.sub
                w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
                w_residual_new = self.sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsification2, operator2) # % sum resifual and new receive global model
                return w_sparsification2      #We need only the sparsified value not residual
            
            elif algoName == 'Proposed':
                #print(f"Proposed Algorithm")
                w_sparsified = copy.deepcopy(w_wo_sparsification) # Formatting
                w_sparsified_flat = self.change_weight_vector(w_sparsified)
                w_sparsified_flat_abs = torch.abs(w_sparsified_flat) 
                num_tensor = torch.numel(w_sparsified_flat_abs)

                # First lets spilt them into K partitions 
                G = num_tensor//K  # Only quotient
                #print(f"G = {G}")
                indices = []
                budget = 0  # 1 element per partition

                for i in range(1, K+1): # Goes 1 ... K
                    if i != K:
                        partition_i = w_sparsified_flat_abs[(i-1)*G: i*G-1]  # ith partition
                        #print(f"partiton_i = {partition_i}")                        
                        value, ind = torch.sort(partition_i, descending=True)

                        if (budget == 0):
                            if (value[0] != 0):
                                indices.append((i-1)*G + ind[0]) # only 1 element is selected per partition
                                budget = 0 # reinitialize
                            else:
                                budget = budget + 1    
                        else:  
                            if (value[0] == 0):  # if the first element is zero then, all the budget goes to next partition
                                budget = budget + 1
                            else:
                                for bud in range(0, budget+1):
                                    if value[bud] !=0:
                                       indices.append((i-1)*G + ind[bud-1])
                                       budget = budget - 1
                        #print(f"budget = {budget}")
                            
                    else:   # last partition composed of all remaining
                        partition_i = w_sparsified_flat_abs[(i-1)*G:]
                        value, ind = torch.sort(partition_i, descending=True)
                        for bud in range(0, budget+1):
                            indices.append((i-1)*G + ind[bud]) # only 1 element is selected per partition || Its the last partition, thus I dont care


                values_updated = torch.zeros(num_tensor)
                for j in indices:
                    values_updated[j] = w_sparsified_flat[j]

                w_sparsification3 = copy.deepcopy(w_sparsified) # Formatting

                k1 = 0

                for i in range(len(list(w_sparsification3.keys()))): # This for changes the full local vector to matrix in the same shape of weights

                    x1 = list(w_sparsification3.keys())[i]

                    if len(w_sparsification3[x1].size()) != 1:
                        for j in range(len(w_sparsification3[x1])):

                            z = len(w_sparsification3[x1][j])
                            w_sparsification3[x1][j] = values_updated[k1:k1+z]
                            k1 = k1+z

                    elif len(w_sparsification3[x1].size()) == 1: # This if is for the bias part

                        z = len(w_sparsification3[x1])
                        w_sparsification3[x1] = values_updated[k1:k1+z]
                        k1 = k1+z

                return w_sparsification3






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

    def Spar_Threshold(self, w_sum_train_residual, all_entries, K): # % Makes the vector sparse

        w_sparsification2 = copy.deepcopy(w_sum_train_residual)
        all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
        number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
        ordered, sort_flatten_matrix_index = torch.sort(all_entries_abs)
        number_tensor_accepted = K  # I should change here to select how many values are accepted in Top-K

        selected_entries_index = sort_flatten_matrix_index[number_tensor-number_tensor_accepted:]  # slice K elements

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


        return w_sparsification3

    def _local_test_on_all_clients(self, round_idx, nonZerosOfAggregate, totalClients, commCost, data_trans_norm):    #S to test the metrics of training

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
        print(f"round idx in local_test = {round_idx}")
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        #wandb.log({"Averaged num of non-zero values per round": np.sum(nonZerosOfAggregate)/totalClients, "round": round_idx})
        #wandb.log({"Total communication cost (bits)": commCost, "round": round_idx})
        #wandb.log({"Total communication cost [normalized]": data_trans_norm, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        self.acc_FedISL.append(test_acc)
        logging.info(stats)

    def countNonZeroValues(self, input):    #This function returns the count for non-zero elements
         count = 0
         for key in range(len(input)):
              val1 = list(input.values())[key]
              val2 = torch.count_nonzero(val1) #Counts all non-zero in the tensor and returns a tensor
              count = count + val2.item()      #Get the .item() out of the tensor
         return count     


            