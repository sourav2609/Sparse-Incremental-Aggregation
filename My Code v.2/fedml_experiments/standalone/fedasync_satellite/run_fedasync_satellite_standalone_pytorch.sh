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





python3 ./main_fedasync_satellite.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--lr $LR \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--batch_size $BATCH_SIZE



#GPU=$1 , CLIENT_NUM=$2 , CNPR=$3 , BATCH_SIZE=$4 , DATASET=$5  , DATA_PATH=$6            , MODEL=$7  , DISTRIBUTION=$8 , EPOCH=$9  , LR=$10  , OPT=$11 , CI=$12  , MHP=$13 , MD=$14  , TP=$15  , TRN=$16 , BS=$17  ,  $WD         ,   LRate
#0          10               2             4         mnist       ./../../../data/mnist          lr            hetero          1         0.03      sgd         1       0.4         8         5        100      50        0.0001          20

#sh run_fedasync_satellite_standalone_pytorch.sh 0          10               2             4         mnist       ./../../../data/mnist          lr            hetero          1         0.03      sgd         1       0.4         8         5        100      50        0.0001          20
#sh run_fedasync_satellite_standalone_pytorch.sh 0 10  2  4   cifar10  ./../../../data/cifar10  20   hetero  1   0.03   sgd   1  0.4  8   5  100 50 0.0001
#sh run_fedasync_satellite_standalone_pytorch.sh 0 2 2 4 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 1
#sh run_fedasync_satellite_standalone_pytorch.sh 1 20 20 128 shakespeare ./../../../data/shakespeare rnn hetero 20 5 0.03 sgd 1
#sh run_fedasync_satellite_standalone_pytorch.sh 1 10 10 128 shakespeare ./../../../data/shakespeare rnn hetero 1 1 0.03 sgd 1
