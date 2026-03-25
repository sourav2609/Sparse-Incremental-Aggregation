# FedAvg with weight implementation
import copy
import logging
import random

import numpy as np
import torch
import wandb

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

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
            print('train_data_local_num_dict[client_idx] = ' + str(self.train_data_local_num_dict))
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()

        residual = copy.deepcopy(w_global)
        for key_idx in range(len(residual)):
            values_key_idx = list(residual.values())[key_idx]
            zeros_key_idx = torch.zeros(values_key_idx.shape)
            residual[list(residual.keys())[key_idx]] = zeros_key_idx

        w_residual_all_users = [residual for i in range(self.args.client_num_per_round)]

        round_idx = 0
        self._local_test_on_all_clients(round_idx)
        for round_idx in range(1, self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            client_indexes = range(self.args.client_num_per_round)
            w_global_temp = copy.deepcopy(w_global)
            w_global_temp_to_train = copy.deepcopy(w_global)
            for idx, client in enumerate(self.client_list):

                print(f"w_global_temp_to_train = {w_global_temp_to_train}")
                w = client.train(w_global_temp_to_train, w_global_temp_to_train)
                w_sparsification = copy.deepcopy(w)
                w_global = copy.deepcopy(w_global_temp)
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_sparsification)))


            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)
            if round_idx % self.args.frequency_of_the_test == 0:
                    self._local_test_on_all_clients(round_idx)



    def _Strom_Threshold_sparsification(self, w_sparsification, w_residual_user, idx):

            # p specifies how many percent of entries?
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
                     Threshold_med = torch.median(jj_abs)
                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(jj)>Threshold_med, jj, kk)


            #print(f"***w_sparsification2 = {w_sparsification2}")
            w_residual_new = copy.deepcopy(w_residual_user)
            for key_idx in range(len(w_sparsification1)):
              values_key_idx1 = list(w_sum_train_residual_main.values())[key_idx]
              values_key_idx2 = list(w_sparsification2.values())[key_idx]
              values_key_idx3 = values_key_idx1 - values_key_idx2
              w_residual_new[list(w_residual_user.keys())[key_idx]] = values_key_idx3

            return w_sparsification2, w_residual_new


    '''
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
    '''



    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset




    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        #print('averaged_params: '+str(averaged_params))
        #print('averaged_params.keys(): '+str(averaged_params.keys()))
        for k in averaged_params.keys():
            #print('k: '+str(k))
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        print(f"cccccccccccccccccccc = {averaged_params}")
        return averaged_params




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
