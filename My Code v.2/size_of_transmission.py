
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#x = [0,1,2,3,6]
def avg_get(x):
    data = x[1:]
    avg_value = sum(data)/(len(data))
    avg_value = avg_value/(64 + np.log2(7850))
    return avg_value
#print(avg_get(x))

data = pd.read_csv("trans_length_10_14.csv")
data = data[data.columns.drop(list(data.filter(regex='MIN')))]
data = data[data.columns.drop(list(data.filter(regex='MAX')))]
data = data[data.columns.drop(list(data.filter(regex='_step')))]
print(data.columns)
data = data.rename(columns={data.columns[1]: '14q0.01', data.columns[2]: '14q0.1',data.columns[3]: '14q0.5',
                            data.columns[4]: '12q0.01', data.columns[5]: '12q0.1', data.columns[6]: '12q0.5',
                            data.columns[7]: '10q0.01', data.columns[8]: '10q0.1', data.columns[9]: '10q0.5'})





data1 = pd.read_csv("trans_size_4_8.csv")
data1 = data1[data1.columns.drop(list(data1.filter(regex='MIN')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='MAX')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='_step')))]
print(data1.columns)
data1 = data1.rename(columns={data1.columns[1]: '8q0.01', data1.columns[2]: '8q0.1',data1.columns[3]: '8q0.5',
                            data1.columns[4]: '6q0.5', data1.columns[5]: '6q0.1', data1.columns[6]: '6q0.01',
                            data1.columns[7]: '4q0.1', data1.columns[8]: '4q0.01', data1.columns[9]: '4q0.5'})


x1=[4,6, 8, 10,12,14]
#data11 = [avg_get(data1['4q0.5']),avg_get(data1['6q0.5']),avg_get(data1['8q0.5']), avg_get(data['10q0.5']), avg_get(data['12q0.5']), avg_get(data['14q0.5'])]
#data11 = [avg_get(data1['4q0.1']),avg_get(data1['6q0.1']),avg_get(data1['8q0.1']), avg_get(data['10q0.1']), avg_get(data['12q0.1']), avg_get(data['14q0.1'])]
data11 = [avg_get(data1['4q0.01']),avg_get(data1['6q0.01']),avg_get(data1['8q0.01']), avg_get(data['10q0.01']), avg_get(data['12q0.01']), avg_get(data['14q0.01'])]
print(data11)
plt.plot(x1,data11)
'''
avg_get(data['4q0.5']), avg_get(data['6q0.5']), avg_get(data['8q0.5']),

x1=[4,6,8,10,12,14]
data1 = [avg_get(data['4q0.1']), avg_get(data['6q0.1']), avg_get(data['8q0.1']), avg_get(data['10q0.1']), avg_get(data['12q0.1']), avg_get(data['14q0.1'])]
plt.plot(x1,data1)

x1=[4,6,8,10,12,14]
data1 = [avg_get(data['4q0.01']), avg_get(data['6q0.01']), avg_get(data['8q0.01']), avg_get(data['10q0.01']), avg_get(data['12q0.01']), avg_get(data['14q0.01'])]
plt.plot(x1,data1)


num_trans1 = []
Num_satellites = [4, 6, 8, 10, 12, 14]
for k in range(len(Num_satellites)):
 big_hops = int(Num_satellites[k]/2)
 #print(big_hops)
 num_trans = 0
 if Num_satellites[k] % 2 !=0:
    for i in range(big_hops+1):
       #print(i)
       num_trans = num_trans + (i*2)
 elif Num_satellites[k] % 2 ==0:
    for i in range(big_hops+1):
        if i == big_hops:
             num_trans = num_trans + (i)
        else:
             num_trans = num_trans + (i*2)
 num_trans1.append(num_trans)

#print(num_trans1)
num_all_transmission = np.array(num_trans1) + np.array(Num_satellites)
#print(num_all_transmission)
data_size_without_inc_agg = 7850 * 64 * num_all_transmission

data_size_with_inc_agg = 7850 * 64 * Num_satellites
#print(data_size)
'''
Num_satellites = [4,6,8,10,12,14]
q=0.01
data_size_Theory = []
for i in range(len(Num_satellites)):
    print(7850*((1-q)**Num_satellites[i]))
    data_size_Theory.append(( 7850 - (7850*((1-q)**Num_satellites[i])))*Num_satellites[i])

plt.plot(Num_satellites, data_size_Theory)

data_length_upper = []
for k in range(len(Num_satellites)):
    data_length_upper.append(7850*Num_satellites[k])
plt.plot(Num_satellites, data_length_upper)

plt.legend(['Exp_sim','Exp_Theory','upper','lower'])
plt.xlabel('Number of satellites')
plt.ylabel('Bits upload')



plt.show()
