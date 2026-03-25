#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

LR=$8

ROUND=$9



python3 ./main_fedavg_satellite.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--lr $LR \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--batch_size $BATCH_SIZE

##
#To run the program
                                     #GPU     Client_num  worker_num  batch-size   dataset                   datapath                                    model       dist         round          epoch         lr        opt   CI
#sh run_fedavg_standalone_pytorch.sh   0       8           4           4           mnist              ./../../../data/mnist          lr      hetero           100            10         0.03       sgd    1
#sh run_fedavg_standalone_pytorch.sh 1 20 20 30 mnist ./../../../data/mnist lr hetero 50 10 0.03 sgd 1
#sh run_fedavg_standalone_pytorch.sh 0 2 2 4 cifar10 ./../../../data/cifar10 resnet56 hetero 1 1 0.03 sgd 1
#sh run_fedavg_standalone_pytorch.sh 1 2 2 4 cifar10 ./../../../data/cifar10 resnet56 hetero 20 3 0.03 sgd 1

#sh run_fedavg_satellite_standalone_pytorch.sh 1 20 20 30 shakespeare ./../../../data/shakespeare rnn hetero 50 5 0.03 sgd 1
#sh run_fedavg_standalone_pytorch.sh 1 20 20 30 mnist ./../../../data/mnist lr hetero 50 10 0.03 sgd 1
