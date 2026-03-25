import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
#from torch import torchsummary

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.LeNet.LeNet import LeNet
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cnn.cnn import Net

var = input("Enter algorithm (Baseline, SIA, RE_SIA, CL_SIA, CL_SIA_linear_growth_K, TC_SIA, RE_TC_SIA, CL_TC_SIA, Aggregate_transfer, CL_SIA_Heuristic,CL_SIA_Heuristic_new, MajorityVoting, NaiveStrategy, signVoting, CL_SIA_Bho, CL_SIA_Sourav, CL_SIA_equal_part, CL_SIA_adapt_part) = ")

if var == "Baseline":
    from fedml_api.standalone.fedavg.fedavg_api_baseline import FedAvgAPI
elif var == "SIA":
    from fedml_api.standalone.fedavg.fedavg_api_SIA import FedAvgAPI
elif var == "RE_SIA":
    from fedml_api.standalone.fedavg.fedavg_api_RE_SIA import FedAvgAPI
elif var == "CL_SIA":    
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA import FedAvgAPI
elif var == "TC_SIA":    
    from fedml_api.standalone.fedavg.fedavg_api_TC_SIA import FedAvgAPI
elif var == "RE_TC_SIA":    
    from fedml_api.standalone.fedavg.fedavg_api_RE_TC_SIA import FedAvgAPI
elif var == "CL_TC_SIA":    
    from fedml_api.standalone.fedavg.fedavg_api_CL_TC_SIA import FedAvgAPI
elif var == "CL_SIA_linear_growth_K":    
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_linear_growth_K import FedAvgAPI    
elif var == "Aggregate_transfer":    
    from fedml_api.standalone.fedavg.fedavg_api_aggregate_transfer import FedAvgAPI 
elif var == "CL_SIA_Heuristic":    
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_Heuristic import FedAvgAPI 
elif var == "CL_SIA_Heuristic_new":    
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_Heuristic_new import FedAvgAPI 
elif var == "MajorityVoting":
    from fedml_api.standalone.fedavg.fedavg_api_MajorityVoting import FedAvgAPI 
elif var == "NaiveStrategy":
    from fedml_api.standalone.fedavg.fedavg_api_NaiveStrategy import FedAvgAPI 
elif var == "signVoting":
    from fedml_api.standalone.fedavg.fedavg_api_signVoting import FedAvgAPI  
elif var == "CL_SIA_Bho":
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_Bho import FedAvgAPI 
elif var == "CL_SIA_Sourav":
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_Sourav import FedAvgAPI       
elif var == "CL_SIA_new":
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_equal_part import FedAvgAPI       
elif var == "CL_SIA_adapt_part":
    from fedml_api.standalone.fedavg.fedavg_api_CL_SIA_adapt_part import FedAvgAPI 
else:
    print(f"Give proper algo name !!!!!!!!!!!!!!!!!!!!!!")   


from fedml_api.standalone.fedavgPlannedAsync.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')
    
    parser.add_argument('--spar_ratio', type=float, default=1, help='sparsification ratio')    

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    #parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
    #                    help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; sgd')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.5)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=1,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    

#    parser.add_argument('--set_random_number_for_shuffle', type=int, default=100,
#                       help='set_random_number_for_shuffle')
    return parser


#def load-data returns dataset which is defined as follows:
# dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
#           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        print(train_data_local_dict[0][0][1])
        """
        For shallow NN or linear models,
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num



    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    #full_x = torch.from_numpy(np.asarray([])).long()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]

# def create_model returns the model specifications
def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
        #print(model,input_size=(768,),depth=1,batch_dim=1, dtypes=[‘torch.IntTensor’])

    elif model_name == "lr" and args.dataset == "shakespeare":
        logging.info("lr + shakespeare")
        model = RNN_OriginalFedAvg()

    elif model_name == "LeNet" and args.dataset == "mnist":
        logging.info("LeNet + MNIST")
        model = LeNet()



    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("cnn + mnist")
        model = Net()


    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model

# Specify for each dataset which model should be used
def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")
    logger.info(device)


    wandb.init( project="Test_17.12", name = "numNodes = " + str(args.client_num_in_total) + "_commRounds = " + str(args.comm_round) + "_algo = "+ var + "_q = " + str(args.spar_ratio),config=args)
    #wandb.init( project="Trash", name = "numNodes = " + str(args.client_num_in_total) + "_commRounds = " + str(args.comm_round) + "_algo = "+ var + "_q = " + str(args.spar_ratio),config=args)
    #wandb.init(project="Results_LEOsats_schemes", name="fedavgPlannedAsync_withoutaug" + ",-lr = " + str(args.lr)
    #+ ", Data = " + str(args.dataset)+ " ,Num_satellites = " +str(args.client_num_per_round),config=args)
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    # We have dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
    #train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num], then dataset[7] shows the class_num.
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print(f"%%% pytorch_total_params = {pytorch_total_params}")
    logging.info(model)
    #print(f"p in args = {args.spar_ratio}")
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()
