

import wandb

K = int(input("Enter number of clients: "))

globaL_iter_num = int(input("Number of global intr: "))

data_trans_norm = 0

wandb.init( project="Sparsification2", name="FedAVG" + ",-lr = " + "_numNodes = " + str(K) + "BatchSize = 4" + "_comRounds = " + str(globaL_iter_num) + "_algo = w/o IA w/o Spar" + "_q = ")
    #wandb.init(project="Results_LEOsats_schemes", name="fedavgPlannedAsync_withoutaug" + ",-lr = " + str(args.lr)


for global_iter in range(1, globaL_iter_num):
    for client in range(1, K):
        data_trans_norm = data_trans_norm + (K*(K+1))/2

    wandb.log({"Total transmitted data [normalized]": data_trans_norm, "round": global_iter})


