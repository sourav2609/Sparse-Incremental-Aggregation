# Implementing FedAvg with gradients, in this code, at first the received y_i from each orbit are summed and then we derive the sum of sum x_i for each orbit.


import copy
import logging
import random
from torch import tensor
import numpy as np
import torch
torch.set_printoptions(precision=10)
import wandb
torch.set_printoptions(profile="full")

from fedml_api.standalone.fedavg.client import Client
from AMP_FL import AMP_Implement_Sat
from AMP_FL import AMP_Implement_PS
np.random.seed(1)

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

        self.model_trainer = model_trainer
        self.number_non_zero_entries = []
        self.number_orbits = 2
        self.vectors_local_AMP_each_orbit = [[] for num_orb in range(self.number_orbits)]
        self.sum_transform_vectors_each_orbit = [0 for num_orb in range(self.number_orbits)]
        self.sum_transform_vectorss = [0 for num_orb in range(self.number_orbits)]
        self.error_each_orbit = [0 for num_orb in range(self.number_orbits)]

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
            #print('train_data_local_num_dict[client_idx] = ' + str(self.train_data_local_num_dict))
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()


        ### This for-loop makes a zero-values dictionary with the same size of global parameters.
        residual = copy.deepcopy(w_global)
        for key_idx in range(len(residual)):
            values_key_idx = list(residual.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            residual[list(residual.keys())[key_idx]] = zeros_key_idx


        w_residual_all_users = [residual for i in range(self.args.client_num_per_round)]    # Makes a vector for residual for all users

        round_idx = 0
        self._local_test_on_all_clients(round_idx)
        for round_idx in range(1, self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            vectors_local_AMP = []
            w_global_temp = copy.deepcopy(w_global)
            w_global_temp_to_train = copy.deepcopy(w_global)
            self.number_non_zero_entries = []   # In order to derive the non-zero elements
            for idx, client in enumerate(self.client_list):

                model_old = copy.deepcopy(w_global_temp)
                gradients = copy.deepcopy(w_global_temp)
                w = client.train(w_global_temp_to_train, w_global_temp_to_train)
                model_new = copy.deepcopy(w)


                for key_idx in range(len(gradients)):

                   values_key_idx1 = list(model_old.values())[key_idx]
                   values_key_idx2 = list(model_new.values())[key_idx]
                   values_key_idx3 = values_key_idx2 - values_key_idx1
                   gradients[list(gradients.keys())[key_idx]] = values_key_idx3


                w_wo_sparsification = copy.deepcopy(gradients)
                w_residual_user = copy.deepcopy(w_residual_all_users[idx])
                #print(f"w_residual_user = {w_residual_user}")
                p = 0.1


                #w_sparsification1, w_residual_new = self._digital_sparsification(w_sparsification, w_residual_user, idx)
                #w_sparsification1, w_residual_new = self._Strom_Threshold_sparsification_all(w_sparsification, w_residual_user, idx, p)
                #w_sparsification1, w_residual_new, w_sparsified_transmission = self._AMP_sparsification_all(w_sparsification, w_residual_user, idx, p)
                w_sparsification1, w_residual_new, w_sparsified_transmission, N_AMP, n_AMP, k_AMP, AMP_all_entries = self._AMP_Threshold_sparsification_all(w_wo_sparsification, w_residual_user, idx, p)
                #w_sparsification1, w_residual_new = self._Strom_Threshold_sparsification(w_sparsification, w_residual_user, idx, p)
                w_residual_all_users[idx] = copy.deepcopy(w_residual_new) # Renew the residual values
                vectors_local_AMP.append(w_sparsified_transmission) # This is the y (y =Ax).
                self.number_non_zero_entries.append(AMP_all_entries)
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_sparsified_transmission)))


            #print(f"*****len(vectors_local_AMP[0]) = {len(vectors_local_AMP[0])}")
            size_of_upload_bits = vectors_local_AMP.size * vectors_local_AMP.itemsize * 8
            print(f"size_of_upload_bits = {size_of_upload_bits}")
            Num_sat_per_orbit = int(len(self.client_list) / self.number_orbits)
            for idx_orb in range(self.number_orbits):
               self.vectors_local_AMP_each_orbit[idx_orb] = vectors_local_AMP[idx_orb*Num_sat_per_orbit: idx_orb*Num_sat_per_orbit + Num_sat_per_orbit] # Each entry of
               x = self.count_non_zero_elements(self, self.vectors_local_AMP_each_orbit[idx_orb])
               self.sum_transform_vectorss[idx_orb] = self.sum_transform_AMP_(self.vectors_local_AMP_each_orbit[idx_orb]) # Gets average from \sum_yi
               self.sum_transform_vectors_each_orbit[idx_orb], self.error_each_orbit[idx_orb] = AMP_Implement_PS(self.sum_transform_vectorss[idx_orb], N_AMP, n_AMP) # Gets \avg_xi from \avg_yi (AMP)


            print(f"self.sum_transform_vectors_each_orbit = {self.sum_transform_vectors_each_orbit}")
            sum_transform_vectors = sum(self.sum_transform_vectors_each_orbit)
            avg_transform_vectors = sum_transform_vectors / self.args.client_num_per_round
            w_global = self._aggregate_gradients(w_locals, w_global_temp, avg_transform_vectors)
            self.model_trainer.set_model_params(w_global)

            if round_idx % self.args.frequency_of_the_test == 0:
                    self._local_test_on_all_clients(round_idx)


    def count_non_zero_elements(self, AMP_all_entries):

        #print(f"AMP_all_entries = {AMP_all_entries}")
        #print(f"AMP_all_entries = {len(AMP_all_entries)}")
        x0 = copy.deepcopy(AMP_all_entries)
        a = torch.where(x0[0] != 0.0000000000)[0]
        print(f"len(x0[0]) = {len(x0[0])}")
        print(f"len(a) = {len(a)}")
        file1 = open("Original.txt", "w")
        file1.write("%s = %s\n" %("AMP_all_entries", AMP_all_entries))
        file1.close()
        for i in range(1,len(x0)):
           b = torch.where(x0[i] != 0.0000000000)[0]
           print(f"len(b) = {len(b)}")
           a = torch.cat((b,a))

        unique, counts = torch.unique(a, return_counts=True)    # len(unique) gives the number of k(the number of non-zero) in the whole vector of all clients
        return len(unique)

    def _Strom_Threshold_sparsification(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)



            w_sum_train_residual = copy.deepcopy(w_residual_user)
            #print(f"###first = {w_sum_train_residual}")
            for key_idx in range(len(w_residual_user)):
              #print(f"$$w_residual_user[0] = {w_residual_user.values()[0][0]}, idx = {idx}")
              values_key_idx1 = list(w_residual_user.values())[key_idx]
              values_key_idx2 = list(w_sparsification1.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            #print(f"$$w_sum_train_residual = {w_sum_train_residual}")
            #print(mean_positive, mean_negative)
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            for g in range(len(w_sum_train_residual_main)):
                     jj = list(w_sum_train_residual_main.values())[g]
                     kk = torch.zeros(jj.shape)
                     jj_abs = torch.abs(jj)
                     #Threshold_med = torch.median(jj_abs)
                     flatten_matrix = torch.flatten(jj_abs)
                     #print(f"jj_abs'= {flatten_matrix}")
                     number_tensor = torch.numel(jj)
                     sort_flatten_matrix = torch.sort(flatten_matrix)
                     #print(f"sort_flatten_matrix = {sort_flatten_matrix}")
                     number_entrries = int(number_tensor * p)
                     #print(f"number_entrries = {number_entrries}")
                     if number_entrries == 0:
                         threshold = 100000
                     else:
                         threshold = sort_flatten_matrix[0][number_tensor - number_entrries]
                     #print(f"threshold = {threshold}")
                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(jj)>=threshold, jj, kk)


            #print(f"***w_sparsification2 = {w_sparsification2}")
            w_residual_new = copy.deepcopy(w_residual_user)

            for key_idx in range(len(w_sparsification1)):
              values_key_idx1 = list(w_sum_train_residual_main.values())[key_idx]
              values_key_idx2 = list(w_sparsification2.values())[key_idx]
              values_key_idx3 = values_key_idx1 - values_key_idx2
              w_residual_new[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            #if idx == 0:
            #   print(f"threshold = {threshold}")
            #   print(f"w_sparsification = {w_sparsification}")
            #   print(f"w_sparsification2 = {w_sparsification2}")
            #   print(f"w_residual_new = {w_residual_new}")

            return w_sparsification2, w_residual_new

    def _Strom_Threshold_sparsification_all(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)



            w_sum_train_residual = copy.deepcopy(w_residual_user)
            #print(f"###first = {w_sum_train_residual}")
            for key_idx in range(len(w_residual_user)):
              #print(f"$$w_residual_user[0] = {w_residual_user.values()[0][0]}, idx = {idx}")
              values_key_idx1 = list(w_residual_user.values())[key_idx]
              values_key_idx2 = list(w_sparsification1.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3


            all_entries_sep_values = []
            for g in range(len(w_sum_train_residual)):

               values_tensor = list(w_sum_train_residual.values())[g].float()
               flatten_values_tensor = torch.flatten(values_tensor)
               all_entries_sep_values.append(flatten_values_tensor)

            #print(f"all_entries_sep_values = {all_entries_sep_values}")
            all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            #print(f"all_entries = {all_entries}")
            #all_entries = torch.flatten(all_entries)


            #print(f"$$w_sum_train_residual = {w_sum_train_residual}")
            #print(mean_positive, mean_negative)
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            for g in range(len(w_sum_train_residual_main)):
                     jj = list(w_sum_train_residual_main.values())[g]
                     kk = torch.zeros(jj.shape)
                     jj_abs = torch.abs(all_entries)
                     #Threshold_med = torch.median(jj_abs)
                     flatten_matrix = torch.flatten(jj_abs)
                     #print(f"jj_abs'= {flatten_matrix}")
                     number_tensor = torch.numel(all_entries)
                     sort_flatten_matrix = torch.sort(flatten_matrix)
                     #print(f"sort_flatten_matrix = {sort_flatten_matrix}")
                     number_entrries = int(number_tensor * p)
                     #print(f"number_entrries = {number_entrries}")
                     if number_entrries == 0:
                         threshold = 100000
                     else:
                         threshold = sort_flatten_matrix[0][number_tensor - number_entrries]
                     #print(f"threshold = {threshold}")
                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(jj)>=threshold, jj, kk)


            #print(f"***w_sparsification2 = {w_sparsification2}")
            w_residual_new = copy.deepcopy(w_residual_user)

            for key_idx in range(len(w_sparsification1)):
              values_key_idx1 = list(w_sum_train_residual_main.values())[key_idx]
              values_key_idx2 = list(w_sparsification2.values())[key_idx]
              values_key_idx3 = values_key_idx1 - values_key_idx2
              w_residual_new[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            #if idx == 0:
            #   print(f"threshold = {threshold}")
            #   print(f"w_sparsification = {w_sparsification}")
            #   print(f"w_sparsification2 = {w_sparsification2}")
            #   print(f"w_residual_new = {w_residual_new}")

            return w_sparsification2, w_residual_new

    def write_vector_in_file(self, file_name, vector):
                   with open(file_name, "w") as f:
                        f.write(str(vector))

    def change_weight_vector(self, w_sum_train_residual):
            all_entries_sep_values = []
            for g in range(len(w_sum_train_residual)):

               values_tensor = list(w_sum_train_residual.values())[g].float()
               flatten_values_tensor = torch.flatten(values_tensor)
               all_entries_sep_values.append(flatten_values_tensor)

            #print(f"all_entries_sep_values = {all_entries_sep_values}")
            all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            return all_entries

    def _AMP_Threshold_sparsification_all(self, w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)
            w_sum_train_residual = copy.deepcopy(w_residual_user)
            #print(f"###first = {w_sum_train_residual}")
            for key_idx in range(len(w_residual_user)):
              #print(f"$$w_residual_user[0] = {w_residual_user.values()[0][0]}, idx = {idx}")
              values_key_idx1 = list(w_residual_user.values())[key_idx]
              values_key_idx2 = list(w_sparsification1.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            all_entries = self.change_weight_vector(w_sum_train_residual)

            #print(mean_positive, mean_negative)
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            for g in range(len(w_sum_train_residual_main)): # for loop to create the sparse vector
                     jj = list(w_sum_train_residual_main.values())[g]
                     kk = torch.zeros(jj.shape)
                     jj_abs = torch.abs(all_entries)
                     flatten_matrix = torch.flatten(jj_abs)
                     number_tensor = torch.numel(all_entries)
                     sort_flatten_matrix = torch.sort(flatten_matrix)
                     number_entrries = int(number_tensor * p)
                     if number_entrries == 0:
                         threshold = 100000
                     else:
                         threshold = sort_flatten_matrix[0][number_tensor - number_entrries]
                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(jj)>=threshold, jj.float(), kk)

            #print(f"***w_sparsification2 = {w_sparsification2}")
            k = number_entrries
            print(f"k = {k}")
            N = number_tensor
            n = int(N/2)
            all_entries_w_sparsification2 = []
            w_sparsification3 = copy.deepcopy(w_sparsification2)
            #print(f"w_sparsification3 = {w_sparsification3['linear.weight'][0]}")
            #print(f"@@@vbias_w_sparsification3 = {w_sparsification3['linear.bias']}")
            AMP_all_entries = self.change_weight_vector(w_sparsification3)

            self.write_vector_in_file("values1.dat", AMP_all_entries) # To write the vector in a file for test
            w_sparsified_transmission = AMP_Implement_Sat(AMP_all_entries, N, n)    # Each satellite derives y, (y=Ax)

            w_residual_new = copy.deepcopy(w_residual_user)

            for key_idx in range(len(w_sparsification1)):   # This for msked the new residual.
              values_key_idx1 = list(w_sum_train_residual_main.values())[key_idx]
              values_key_idx2 = list(w_sparsification2.values())[key_idx]
              values_key_idx3 = values_key_idx1 - values_key_idx2
              w_residual_new[list(w_residual_user.keys())[key_idx]] = values_key_idx3


            return w_sparsification2, w_residual_new, w_sparsified_transmission, N, n, k, AMP_all_entries

    def _digital_sparsification(self, w_sparsification, w_residual_user, idx):

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)
            #print(f"$$w_sparsification1 = {w_sparsification1}")
            #print(f"$$w_residual_user[] = {w_residual_user}")
            mean_negative = torch.zeros(len(w_sparsification1.keys()))
            mean_positive = torch.zeros(len(w_sparsification1.keys()))



            w_sum_train_residual = copy.deepcopy(w_residual_user)
            #print(f"###first = {w_sum_train_residual}")
            for key_idx in range(len(w_residual_user)):
              #print(f"$$w_residual_user[0] = {w_residual_user.values()[0][0]}, idx = {idx}")
              values_key_idx1 = list(w_residual_user.values())[key_idx]
              values_key_idx2 = list(w_sparsification1.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            #print(f"$$w_sum_train_residual = {w_sum_train_residual}")
            #print(mean_positive, mean_negative)
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            for g in range(len(w_sum_train_residual_main)):
                     jj = list(w_sum_train_residual_main.values())[g]
                     kk = torch.zeros(jj.shape)
                     xpos = torch.where(jj>0, jj, kk)
                     mean_positive[g] = torch.sum(xpos)
                     xneg = torch.where(jj<0, jj, kk)
                     mean_negative[g] = torch.sum(xneg)
                     #if g == 0 and idx ==0:
                     #     print(f"$$$ jj = {list(jj)}")
                     #     print(f"$$$ w_sum_train_residual_main = {w_sum_train_residual_main}")

                     #print(mean_positive[g], mean_negative[g])
                     if abs(mean_negative[g]) < mean_positive[g]:
                         w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(jj>0, jj, kk)
                     elif abs(mean_negative[g]) > mean_positive[g]:
                         w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(jj<0, jj, kk)

            #print(f"***w_sparsification2 = {w_sparsification2}")
            w_residual_new = copy.deepcopy(w_residual_user)
            for key_idx in range(len(w_sparsification1)):
              values_key_idx1 = list(w_sum_train_residual_main.values())[key_idx]
              values_key_idx2 = list(w_sparsification2.values())[key_idx]
              values_key_idx3 = values_key_idx1 - values_key_idx2
              w_residual_new[list(w_residual_user.keys())[key_idx]] = values_key_idx3


            return w_sparsification2, w_residual_new

    def sum_transform_AMP_(self, vectors_local_AMP):

        sum_output_transform_AMP = sum(vectors_local_AMP)
        #avg_output_transform_AMP = sum_output_transform_AMP / len(vectors_local_AMP)
        return sum_output_transform_AMP

    def _aggregate_gradients(self, w_locals, w_global, sum_transform_vectors):

        Grad = copy.deepcopy(w_global)

        #print(f"!!!!! len(sum_transform_vectors) = {sum_transform_vectors}")
        k1 = 0
        for i in range(len(list(w_global.keys()))): # This for changes the full local vector to matrix in the same shape of weights

            x1 = list(w_global.keys())[i]

            if len(w_global[x1].size()) != 1:
              for j in range(len(w_global[x1])):

                z = len(w_global[x1][j])
                Grad[x1][j] = tensor(sum_transform_vectors[k1:k1+z])
                k1 = k1+z

            elif len(Grad[x1].size()) == 1: # This if is for the bias part

                  z = len(Grad[x1])
                  Grad[x1] = tensor(sum_transform_vectors[k1:k1+z])
                  k1 = k1+z

        #print(f"1111111### w_local = {w_local}")
        w_global_new = copy.deepcopy(w_global)
        for key_idx in range(len(w_global)):    # This for derives the new global parameters by using the gradients and old global weights
              values_key_idx1 = list(w_global.values())[key_idx]
              values_key_idx2 = list(Grad.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_global_new[list(w_global_new.keys())[key_idx]] = values_key_idx3


        return w_global_new

    def _local_test_on_all_clients(self, round_idx):

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
        logging.info(stats)
