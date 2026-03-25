import json
import numpy as np
import math
from random import shuffle
from operator import itemgetter
import random
import copy



# Opening JSON file (load Jason file as python dictionary)

with open('all_data_0_niid_0_keep_10_test_9.json') as json_file:
    data_all = json.load(json_file)


random.seed(10)
np.random.seed(10)
data_keys = list(data_all.keys())
values = list(data_all.values())
number_of_devices = 12
number_digits_each_client = 2
number_of_samples = data_all[data_keys[0]]


#Combine all data of x and all data of y in order to first shuffle and then divide between clients

data_x_y=data_all[data_keys[2]] #Take the x and y
#print(f"data_x_y = {data_x_y}")
xx=[]
yy=[]

for i in (data_all[data_keys[1]]):
    data=data_x_y[i] #take the data of each client

    keyss=list(data_x_y[i].keys())

    yy+=data[keyss[0]]
    xx+=data[keyss[1]]


#Shuffle the combined data of x and y --> at first were non-iid
index_shuf = list(range(len(yy)))
shuffle(index_shuf)
listxx_shuf=[]
listyy_shuf=[]
for i in index_shuf:
     listxx_shuf.append(xx[i])
     listyy_shuf.append(yy[i])



#print('listyy_shuf'+str(listyy_shuf))
indices, listyy_shuff_sorted = zip(*sorted(enumerate(listyy_shuf), key=itemgetter(1)))
indec=list(indices)
listyy_shuff_sorted=list(listyy_shuff_sorted)

#print(f"len(listyy_shuff_sorted): {listyy_shuff_sorted}")

listxx_shuff_sorted=[]
for i in indec:
    listxx_shuff_sorted.append(listxx_shuf[i])
listxx_shuff_sorted=list(listxx_shuff_sorted)

###################################
digits=[[0,1],[2,3],[4,5],[6,7],[8],[9]]
list_group_x = []
list_group_y = []
a=0
for k in range(len(digits)):
           list_group_y.append([x for x in listyy_shuff_sorted if x in digits[k]])
           if k==0:
               list_group_x.append(listxx_shuff_sorted[0:len(list_group_y[k])])
           else:
               a = a + len(list_group_y[k-1])
               #print(len(list_group_y[k-1]))
               list_group_x.append(listxx_shuff_sorted[a:a+len(list_group_y[k])])
           #print(len(list_group_x[k]), len(list_group_y[k]))
#print(len(list_group_x))
#print(len(list_group_y[0]))
#print(len(list_group_x[1]))

listxx_shuf = [[] for k in range(len(list_group_x))]
listyy_shuf = [[] for k in range(len(list_group_y))]

for k in range(len(list_group_y)):
   index_shuf = list(range(len(list_group_y[k])))
   shuffle(index_shuf)
   #print(f"$$$ = {len(index_shuf)}")

   for i in index_shuf:
     listxx_shuf[k].append(list_group_x[k][i])
     #print(list_group_y[k][i])
     listyy_shuf[k].append(list_group_y[k][i])

len_find = []
for k in range(len(list_group_y)):
    len_find.append(len(listyy_shuf[k]))

len_min = min(len_find)
listxx = []
listyy = []
for k in range(len(list_group_y)):
       listxx.append(listxx_shuf[k][0:len_min])
       listyy.append(listyy_shuf[k][0:len_min])

#print(listyy_shuf)



if number_of_devices == 4:
     matrix_x = [[] for i in range(4)]
     matrix_y = [[] for i in range(4)]
     for num in range(number_of_devices):
                  if num == 0:
                      matrix_x[num].append(listxx[num])
                      matrix_x[num].append(listxx[4][0:int(len(listxx[4])/2)])
                      matrix_y[num].append(listyy[num])
                      matrix_y[num].append(listyy[4][0:int(len(listyy[4])/2)])

                  elif num == 1:
                       matrix_x[num].append(listxx[num])
                       matrix_x[num].append(listxx[4][int(len(listxx[4])/2):int(len(listxx[4]))])
                       matrix_y[num].append(listyy[num])
                       matrix_y[num].append(listyy[4][int(len(listyy[4])/2):int(len(listyy[4]))])

                  elif num == 2:
                      matrix_x[num].append(listxx[num])
                      matrix_x[num].append(listxx[5][0:int(len(listxx[5])/2)])
                      matrix_y[num].append(listyy[num])
                      matrix_y[num].append(listyy[5][0:int(len(listyy[5])/2)])

                  elif num == 3:
                       matrix_x[num].append(listxx[num])
                       matrix_x[num].append(listxx[5][int(len(listxx[5])/2):int(len(listxx[5]))])
                       matrix_y[num].append(listyy[num])
                       matrix_y[num].append(listyy[5][int(len(listyy[5])/2):int(len(listyy[5]))])

elif number_of_devices == 8:
     matrix_x = [[] for i in range(8)]
     matrix_y = [[] for i in range(8)]
     for num in range(number_of_devices):
                  if num == 0:
                      matrix_x[num].append(listxx[0][0:int(len(listxx[0])/2)])
                      matrix_x[num].append(listxx[4][0:int(len(listxx[4])/4)])
                      matrix_y[num].append(listyy[0][0:int(len(listyy[0])/2)])
                      matrix_y[num].append(listyy[4][0:int(len(listyy[4])/4)])

                  if num == 1:
                      matrix_x[num].append(listxx[0][int(len(listxx[0])/2):int(len(listxx[0]))])
                      matrix_x[num].append(listxx[4][int(len(listxx[4])/4):int(len(listxx[4])/2)])
                      matrix_y[num].append(listyy[0][int(len(listyy[0])/2):int(len(listxx[0]))])
                      matrix_y[num].append(listyy[4][int(len(listxx[4])/4):int(len(listyy[4])/2)])

                  elif num == 2:
                      matrix_x[num].append(listxx[1][0:int(len(listxx[1])/2)])
                      matrix_x[num].append(listxx[4][int(2*len(listxx[4])/4):int(3*len(listxx[4])/4)])
                      matrix_y[num].append(listyy[1][0:int(len(listyy[1])/2)])
                      matrix_y[num].append(listyy[4][int(2*len(listxx[4])/4):int(3*len(listxx[4])/4)])

                  elif num == 3:
                      matrix_x[num].append(listxx[1][0:int(len(listxx[1])/2)])
                      matrix_x[num].append(listxx[4][int(3*len(listxx[4])/4):int(4*len(listxx[4])/4)])
                      matrix_y[num].append(listyy[1][0:int(len(listyy[1])/2)])
                      matrix_y[num].append(listyy[4][int(3*len(listxx[4])/4):int(4*len(listxx[4])/4)])

                  elif num == 4:
                      matrix_x[num].append(listxx[2][0:int(len(listxx[2])/2)])
                      matrix_x[num].append(listxx[5][0:int(len(listxx[5])/4)])
                      matrix_y[num].append(listyy[2][0:int(len(listyy[2])/2)])
                      matrix_y[num].append(listyy[5][0:int(len(listyy[5])/4)])

                  elif num == 5:
                      matrix_x[num].append(listxx[2][int(len(listxx[0])/2):int(len(listxx[0]))])
                      matrix_x[num].append(listxx[5][int(len(listxx[4])/4):int(len(listxx[4])/2)])
                      matrix_y[num].append(listyy[2][int(len(listyy[0])/2):int(len(listxx[0]))])
                      matrix_y[num].append(listyy[5][int(len(listxx[4])/4):int(len(listyy[4])/2)])

                  elif num == 6:
                      matrix_x[num].append(listxx[3][0:int(len(listxx[1])/2)])
                      matrix_x[num].append(listxx[5][int(2*len(listxx[4])/4):int(3*len(listxx[4])/4)])
                      matrix_y[num].append(listyy[3][0:int(len(listyy[1])/2)])
                      matrix_y[num].append(listyy[5][int(2*len(listxx[4])/4):int(3*len(listxx[4])/4)])

                  elif num == 7:
                      matrix_x[num].append(listxx[3][0:int(len(listxx[3])/2)])
                      matrix_x[num].append(listxx[5][int(3*len(listxx[5])/4):int(4*len(listxx[5])/4)])
                      matrix_y[num].append(listyy[3][0:int(len(listyy[3])/2)])
                      matrix_y[num].append(listyy[5][int(3*len(listxx[5])/4):int(4*len(listxx[5])/4)])

elif number_of_devices == 12:
     matrix_x = [[] for i in range(12)]
     matrix_y = [[] for i in range(12)]
     for num in range(number_of_devices):
                  if num == 0:
                      matrix_x[num].append(listxx[0][0:int(len(listxx[0])/3)])
                      matrix_x[num].append(listxx[4][0:int(len(listxx[4])/6)])
                      matrix_y[num].append(listyy[0][0:int(len(listyy[0])/3)])
                      matrix_y[num].append(listyy[4][0:int(len(listyy[4])/6)])

                  if num == 1:
                      matrix_x[num].append(listxx[0][int(len(listxx[0])/3):int(2*len(listxx[0])/3)])
                      matrix_x[num].append(listxx[4][int(1*len(listxx[4])/6):int(2*len(listxx[4])/6)])
                      matrix_y[num].append(listyy[0][int(len(listxx[0])/3):int(2*len(listxx[0])/3)])
                      matrix_y[num].append(listyy[4][int(1*len(listxx[4])/6):int(2*len(listyy[4])/6)])

                  elif num == 2:
                      matrix_x[num].append(listxx[0][int(2*len(listxx[0])/3):int(3*len(listxx[0])/3)])
                      matrix_x[num].append(listxx[4][int(2*len(listxx[4])/6):int(3*len(listxx[4])/6)])
                      matrix_y[num].append(listyy[0][int(2*len(listxx[0])/3):int(3*len(listxx[0])/3)])
                      matrix_y[num].append(listyy[4][int(2*len(listxx[4])/6):int(3*len(listxx[4])/6)])

                  elif num == 3:
                      matrix_x[num].append(listxx[1][0:int(len(listxx[0])/3)])
                      matrix_x[num].append(listxx[4][int(3*len(listxx[4])/6):int(4*len(listxx[4])/6)])
                      matrix_y[num].append(listyy[1][0:int(len(listyy[0])/3)])
                      matrix_y[num].append(listyy[4][int(3*len(listxx[4])/6):int(4*len(listxx[4])/6)])

                  if num == 4:
                      matrix_x[num].append(listxx[1][int(len(listxx[1])/3):int(2*len(listxx[1])/3)])
                      matrix_x[num].append(listxx[4][int(4*len(listxx[4])/6):int(5*len(listxx[4])/6)])
                      matrix_y[num].append(listyy[1][int(len(listxx[1])/3):int(2*len(listxx[1])/3)])
                      matrix_y[num].append(listyy[4][int(4*len(listxx[4])/6):int(5*len(listyy[4])/6)])

                  elif num == 5:
                      matrix_x[num].append(listxx[1][int(2*len(listxx[1])/3):int(3*len(listxx[1])/3)])
                      matrix_x[num].append(listxx[4][int(5*len(listxx[4])/6):int(6*len(listxx[4])/6)])
                      matrix_y[num].append(listyy[1][int(2*len(listxx[1])/3):int(3*len(listxx[1])/3)])
                      matrix_y[num].append(listyy[4][int(5*len(listxx[4])/6):int(6*len(listxx[4])/6)])


                  elif num == 6:
                      matrix_x[num].append(listxx[2][0:int(len(listxx[2])/3)])
                      matrix_x[num].append(listxx[5][0:int(len(listxx[5])/6)])
                      matrix_y[num].append(listyy[2][0:int(len(listyy[2])/3)])
                      matrix_y[num].append(listyy[5][0:int(len(listyy[5])/6)])

                  elif num == 7:
                      matrix_x[num].append(listxx[2][int(len(listxx[2])/3):int(2*len(listxx[2])/3)])
                      matrix_x[num].append(listxx[5][int(1*len(listxx[5])/6):int(2*len(listxx[5])/6)])
                      matrix_y[num].append(listyy[2][int(len(listxx[2])/3):int(2*len(listxx[2])/3)])
                      matrix_y[num].append(listyy[5][int(1*len(listxx[5])/6):int(2*len(listyy[5])/6)])

                  elif num == 8:
                      matrix_x[num].append(listxx[2][int(2*len(listxx[2])/3):int(3*len(listxx[2])/3)])
                      matrix_x[num].append(listxx[5][int(2*len(listxx[5])/6):int(3*len(listxx[5])/6)])
                      matrix_y[num].append(listyy[2][int(2*len(listxx[2])/3):int(3*len(listxx[2])/3)])
                      matrix_y[num].append(listyy[5][int(2*len(listxx[5])/6):int(3*len(listxx[5])/6)])

                  elif num == 9:
                      matrix_x[num].append(listxx[3][0:int(len(listxx[3])/3)])
                      matrix_x[num].append(listxx[5][int(3*len(listxx[5])/6):int(4*len(listxx[5])/6)])
                      matrix_y[num].append(listyy[3][0:int(len(listyy[3])/3)])
                      matrix_y[num].append(listyy[5][int(3*len(listxx[5])/6):int(4*len(listxx[5])/6)])

                  if num == 10:
                      matrix_x[num].append(listxx[3][int(len(listxx[3])/3):int(2*len(listxx[3])/3)])
                      matrix_x[num].append(listxx[5][int(4*len(listxx[5])/6):int(5*len(listxx[5])/6)])
                      matrix_y[num].append(listyy[3][int(len(listxx[3])/3):int(2*len(listxx[3])/3)])
                      matrix_y[num].append(listyy[5][int(4*len(listxx[5])/6):int(5*len(listyy[5])/6)])

                  elif num == 11:
                      matrix_x[num].append(listxx[3][int(2*len(listxx[3])/3):int(3*len(listxx[3])/3)])
                      matrix_x[num].append(listxx[5][int(5*len(listxx[5])/6):int(6*len(listxx[5])/6)])
                      matrix_y[num].append(listyy[3][int(2*len(listxx[3])/3):int(3*len(listxx[3])/3)])
                      matrix_y[num].append(listyy[5][int(5*len(listxx[5])/6):int(6*len(listxx[5])/6)])


def shufff(vector_x, vector_y):

    vector1_x = []
    vector2_y = []
    for k in range(0, len(vector_y)):
        vector1_x = vector1_x + vector_x[k]
        vector2_y = vector2_y + vector_y[k]
    #print(vector2_y)
    index_shuf = list(range(len(vector1_x)))
    shuffle(index_shuf)
    listxx_shuf = []
    listyy_shuf = []
    for i in index_shuf:
             listxx_shuf.append(vector1_x[i])
             listyy_shuf.append(vector2_y[i])
    return listxx_shuf, listyy_shuf

matrix_x_new2222 = []
for k in range(len(matrix_x)):
     matrix_x_new2222.append(matrix_x[k])

no_dupes = [x for n, x in enumerate(matrix_x_new2222) if x not in matrix_x_new2222[:n]]
print(no_dupes)




matrix_x_new = []
matrix_y_new = []

for k in range(len(matrix_x)):
      x,y = shufff(matrix_x[k], matrix_y[k])
      matrix_x_new.append(x)
      matrix_y_new.append(y)



print(matrix_x_new)

matrix_y = copy.deepcopy(matrix_y_new)
matrix_x = copy.deepcopy(matrix_x_new)
## Update the values of dictionary
#dict_new = copy.deepcopy(data_all)
x_keys={}
y_keys={}
for i in range(number_of_devices):
    x_keys[str(i)]={'y':[],'x':[]}


users_dict=[]
for i in range(number_of_devices):
    users_dict.append(str(i))


dict_new = {'num_samples': [] , 'users': users_dict , 'user_data': x_keys }
print(dict_new)

data_keys = list(dict_new.keys())
values = list(dict_new.values())
len_data=[]
val = list(values[2].keys())


I_index = []
for k in range(len(matrix_x)):
    I_index.append(k)

random.shuffle(I_index)
I_index1 = np.array(I_index)
print(f" I_index1 = {I_index1}")
for w in range(len(I_index1)):
    dict_new[data_keys[2]][val[w]]['x'] = matrix_x[I_index1[w]]
    dict_new[data_keys[2]][val[w]]['y'] = matrix_y[I_index1[w]]
    len_data.append(len(matrix_x[I_index1[w]]))

dict_new[data_keys[0]]=len_data


with open('IID_strict_train_' + str(number_of_devices)+'.json', 'w') as jsonFile:
    json.dump(dict_new, jsonFile)
