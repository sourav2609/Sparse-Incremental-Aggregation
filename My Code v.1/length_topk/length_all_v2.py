
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#x = [0,1,2,3,6]
def avg_get(x):
    data = x[1:]
    avg_value = sum(data)/(len(data))
    avg_value = avg_value
    return avg_value
#print(avg_get(x))


data = pd.read_csv(".\length_topk\iid_length_3.csv")
data = data[data.columns.drop(list(data.filter(regex='MIN')))]
data = data[data.columns.drop(list(data.filter(regex='MAX')))]
data = data[data.columns.drop(list(data.filter(regex='_step')))]
data = data[data.columns.drop(list(data.filter(regex='len_14')))]
#print(data.columns)
data = data.rename(columns={data.columns[1]: '12iidq0.01', data.columns[2]: '12iidq0.1', data.columns[3]: '12iidq0.5',
                            data.columns[4]: '8iidq0.01',data.columns[5]: '8iidq0.1', data.columns[6]: '8iidq0.5',
                            data.columns[7]: '4iidq0.01',data.columns[8]: '4iidq0.1', data.columns[9]: '4iidq0.5'})



data1 = pd.read_csv("niid_length.csv")
data1 = data1[data1.columns.drop(list(data1.filter(regex='MIN')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='MAX')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='_step')))]
#print(data1.columns)
data1 = data1.rename(columns={data1.columns[1]: '12niidq0.01',data1.columns[2]: '12niidq0.1', data1.columns[3]: '12niidq0.5',
                              data1.columns[4]: '8niidq0.01',data1.columns[5]:  '8niidq0.1',  data1.columns[6]: '8niidq0.5',
                              data1.columns[7]: '4niidq0.01',data1.columns[8]:  '4niidq0.1',  data1.columns[9]: '4niidq0.5'})



data2 = pd.read_csv(".\length_topk\strict_length_2.csv")
data2 = data2[data2.columns.drop(list(data2.filter(regex='MIN')))]
data2 = data2[data2.columns.drop(list(data2.filter(regex='MAX')))]
data2 = data2[data2.columns.drop(list(data2.filter(regex='_step')))]
data2 = data2.rename(columns={data2.columns[1]: '12strictq0.01', data2.columns[2]: '12strictq0.1',  data2.columns[3]: '12strictq0.5',
                              data2.columns[4]: '8strictq0.01',  data2.columns[5]: '8strictq0.1',   data2.columns[6]: '8strictq0.5',
                              data2.columns[7]: '4strictq0.01',  data2.columns[8]: '4strictq0.1',   data2.columns[9]: '4strictq0.5'})






All_data = {}
for i in range(1,10):
    All_data[data.columns[i]] = avg_get(data[data.columns[i]])
#print(All_data)

for i in range(1,10):
    All_data[data1.columns[i]] = avg_get(data1[data1.columns[i]])
#print(All_data)

for i in range(1,10):
    All_data[data2.columns[i]] = avg_get(data2[data2.columns[i]])
print(All_data)
#print(data1)
Num_satellites=[4, 8, 12]
with open('length_topk/length_parameters_topk.csv', 'w') as f:
                                 f.write(f'data_length = {All_data}')
                                 f.write(f'Num_satellites = {Num_satellites}')







def Theory(q, Num_satellites):
   length_Theory = []
   print(Num_satellites)
   for i in range(len(Num_satellites)):
             length_Theory.append(( 7850 - (7850*((1-q)**Num_satellites[i]))))
   return length_Theory


def up_Theory(q, Num_satellites):
   length_upper = []
   for k in range(len(Num_satellites)):
       length_upper.append(min(7850,(Num_satellites[k]*7850*q)) )
   return length_upper

def low_Theory(q, Num_satellites):
   length_lower = []
   for k in range(len(Num_satellites)):
         length_lower.append(7850*q)
   return length_lower


q= [0.01, 0.1, 0.5]
for i in range(3):
    Num_satellites=[4, 8, 12]
    '''
    if q[i] == 0.01:
        dataiid001 = [All_data['4iidq0.01'], All_data['8iidq0.01'], All_data['12iidq0.01']]
        plt.plot(Num_satellites,dataiid001, label = 'IID_q0.01')
        dataNoniid001 = [All_data['4niidq0.01'], All_data['8niidq0.01'], All_data['12niidq0.01']]
        plt.plot(Num_satellites,dataNoniid001, label = 'NonIID_q0.01')
        datastrict001 = [All_data['4strictq0.01'], All_data['8strictq0.01'], All_data['12strictq0.01']]
        plt.plot(Num_satellites, datastrict001, label = 'Strict_q0.01')
        length_Theory1 = Theory(q[i], Num_satellites)
        plt.plot(Num_satellites,length_Theory1, label = 'Theory_0.01')
        length_upper1 = up_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_upper1, label = 'Upper_0.01')
        length_lower1 = low_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_lower1, label = 'Lower_0.01')
        plt.legend()
        plt.xlabel('Number of satellites, q=' + str(q[i]))
        plt.ylabel('length of vector')

    if q[i] == 0.1:
        dataiid01 = [All_data['4iidq0.1'], All_data['8iidq0.1'], All_data['12iidq0.1']]
        plt.plot(Num_satellites, dataiid01, label = 'IID_q0.1')
        dataNoniid01 = [All_data['4niidq0.1'], All_data['8niidq0.1'], All_data['12niidq0.1']]
        plt.plot(Num_satellites, dataNoniid01, label = 'NonIID_q0.1')
        datastrict01 = [All_data['4strictq0.1'], All_data['8strictq0.1'], All_data['12strictq0.1']]
        plt.plot(Num_satellites, datastrict01, label = 'Strict_q0.1')
        length_Theory2 = Theory(q[i], Num_satellites)
        plt.plot(Num_satellites,length_Theory2, label = 'Theory_0.1')
        length_upper2 = up_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_upper2, label = 'Upper_0.1')
        length_lower2 = low_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_lower2, label = 'Lower_0.1')
        plt.legend()
        plt.xlabel('Number of satellites, q=' + str(q[i]))
        plt.ylabel('length of vector')
    '''
    if q[i] == 0.5:
        dataiid05 = [All_data['4iidq0.5'], All_data['8iidq0.5'], All_data['12iidq0.5']]
        plt.plot(Num_satellites, dataiid05, label = 'IID_q0.5')
        dataNoniid05 = [All_data['4niidq0.5'], All_data['8niidq0.5'], All_data['12niidq0.5']]
        plt.plot(Num_satellites, dataNoniid05, label = 'NonIID_q0.5')
        datastrict05 = [All_data['4strictq0.5'], All_data['8strictq0.5'], All_data['12strictq0.5']]
        plt.plot(Num_satellites, datastrict05, label = 'Strict_q0.5')
        length_Theory3 = Theory(q[i], Num_satellites)
        plt.plot(Num_satellites,length_Theory3, label = 'Theory_0.5')
        length_upper3 = up_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_upper3, label = 'Upper_0.5')
        length_lower3 = low_Theory(q[i], Num_satellites)
        plt.plot(Num_satellites, length_lower3, label = 'Lower_0.5')
        plt.legend()
        plt.xlabel('Number of satellites, q=' + str(q[i]))
        plt.ylabel('length of vector')

#length_up = up_Theory(q, Num_satellites)
#length_low = low_Theory(q, Num_satellites)







plt.show()
