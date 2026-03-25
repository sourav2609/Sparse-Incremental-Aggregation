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


python3 ./main_fedavgPlannedAsync_withoutaug.py \
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
#                                                 #GPU     Client_num  worker_num  batch-size   dataset                   datapath              model       dist         round          epoch         lr        opt   CI
##sh run_fedavgPlannedAsync_standalone_pytorch.sh   0       4           4           4           mnist              ./../../../data/mnist          lr      hetero           10            1         0.03       sgd    1
##sh run_fedasync_standalone_pytorch.sh 0 2 2 4 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 1
##sh run_fedavgPlannedAsync_standalone_pytorch.sh 1 20 20 128 shakespeare ./../../../data/shakespeare rnn hetero 50 5 0.03 sgd 1
