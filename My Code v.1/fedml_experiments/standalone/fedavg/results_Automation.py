## Script for automation of result regeneration

import os



client_num = "4"

batch_size = "5"
data = "mnist"
dataPath = "./../../../data/mnist"
model = "lr"
commRound = "50"

q_list = list(1, 0.1, 0.01)
for q in q_list:
    
