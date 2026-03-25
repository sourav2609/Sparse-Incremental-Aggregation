
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
data = data[data.columns.drop(list(data.filter(regex='round')))]
#print(data.columns)
data = data.rename(columns={data.columns[0]: '12iidq001', data.columns[1]: '12iidq01', data.columns[2]: '12iidq05',
                            data.columns[3]: '8iidq001',data.columns[4]: '8iidq01', data.columns[5]: '8iidq05',
                            data.columns[6]: '4iidq001',data.columns[7]: '4iidq01', data.columns[8]: '4iidq05'})



data1 = pd.read_csv("niid_length.csv")
data1 = data1[data1.columns.drop(list(data1.filter(regex='MIN')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='MAX')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='_step')))]
data1 = data1[data1.columns.drop(list(data1.filter(regex='round')))]
#print(data1.columns)
data1 = data1.rename(columns={data1.columns[0]: '12niidq001',data1.columns[1]: '12niidq01', data1.columns[2]: '12niidq05',
                              data1.columns[3]: '8niidq001',data1.columns[4]:  '8niidq01',  data1.columns[5]: '8niidq05',
                              data1.columns[6]: '4niidq001',data1.columns[7]:  '4niidq01',  data1.columns[8]: '4niidq05'})



data2 = pd.read_csv(".\length_topk\strict_length_2.csv")
data2 = data2[data2.columns.drop(list(data2.filter(regex='MIN')))]
data2 = data2[data2.columns.drop(list(data2.filter(regex='MAX')))]
data2 = data2[data2.columns.drop(list(data2.filter(regex='_step')))]
data2 = data2[data2.columns.drop(list(data2.filter(regex='round')))]
print(data2.columns)
print(len(data2.columns))
data2 = data2.rename(columns={data2.columns[0]: '12strictq001', data2.columns[1]: '12strictq01',  data2.columns[2]: '12strictq05',
                              data2.columns[3]: '8strictq001',  data2.columns[4]: '8strictq01',   data2.columns[5]: '8strictq05',
                              data2.columns[6]: '4strictq001',  data2.columns[7]: '4strictq01',   data2.columns[8]: '4strictq05'})


print(data2)



All_data = {}
for i in range(len(data)):
    All_data[data.columns[i]] = avg_get(data[data.columns[i]])
#print(All_data)

for i in range(len(data1)):
    All_data[data1.columns[i]] = avg_get(data1[data1.columns[i]])
#print(All_data)

for i in range(len(data2)):
    All_data[data2.columns[i]] = avg_get(data2[data2.columns[i]])
print(All_data)
#print(data1)
Num_satellites=[4, 8, 12]

#dataiid001 = [All_data['4iidq0.01'], All_data['8iidq0.01'], All_data['12iidq0.01']]
#dataNoniid001 = [All_data['4niidq0.01'], All_data['8niidq0.01'], All_data['12niidq0.01']]
#datastrict001 = [All_data['4strictq0.01'], All_data['8strictq0.01'], All_data['12strictq0.01']]



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

Num_satellites=[4, 8, 12]
All_values = []
All_values = {'Num_sat': Num_satellites}
q= [0.01, 0.5]
for i in range(len(q)):
    Num_satellites=[4, 8, 12]

    if q[i] == 0.01:
        dataiid001 = [All_data['4iidq0.01'], All_data['8iidq0.01'], All_data['12iidq0.01']]
        All_values['dataiid001'] = dataiid001
        #plt.plot(Num_satellites,dataiid001, label = 'IID_q0.01')
        dataNoniid001 = [All_data['4niidq0.01'], All_data['8niidq0.01'], All_data['12niidq0.01']]
        All_values['dataNoniid001'] = dataNoniid001
        #plt.plot(Num_satellites,dataNoniid001, label = 'NonIID_q0.01')
        datastrict001 = [All_data['4strictq0.01'], All_data['8strictq0.01'], All_data['12strictq0.01']]
        All_values['datastrict001'] = datastrict001
        #plt.plot(Num_satellites, datastrict001, label = 'Strict_q0.01')
        length_Theory001 = Theory(q[i], Num_satellites)
        All_values['length_Theory001'] = length_Theory001
        #plt.plot(Num_satellites,length_Theory001, label = 'Theory_0.01')
        length_upper001 = up_Theory(q[i], Num_satellites)
        All_values['length_upper001'] = length_upper001
        #plt.plot(Num_satellites, length_upper001, label = 'Upper_0.01')
        length_lower001 = low_Theory(q[i], Num_satellites)

        #plt.plot(Num_satellites, length_lower001, label = 'Lower_0.01')
        #plt.legend()
        #plt.xlabel('Number of satellites, q=' + str(q[i]))
        #plt.ylabel('length of vector')
    '''
    if q[i] == 0.1:
        dataiid01 = [All_data['4iidq0.1'], All_data['8iidq0.1'], All_data['12iidq0.1']]
        #plt.plot(Num_satellites, dataiid01, label = 'IID_q0.1')
        dataNoniid01 = [All_data['4niidq0.1'], All_data['8niidq0.1'], All_data['12niidq0.1']]
        #plt.plot(Num_satellites, dataNoniid01, label = 'NonIID_q0.1')
        datastrict01 = [All_data['4strictq0.1'], All_data['8strictq0.1'], All_data['12strictq0.1']]
        #plt.plot(Num_satellites, datastrict01, label = 'Strict_q0.1')
        length_Theory01 = Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites,length_Theory2, label = 'Theory_0.1')
        length_upper01 = up_Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites, length_upper2, label = 'Upper_0.1')
        length_lower01 = low_Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites, length_lower2, label = 'Lower_0.1')
        #plt.legend()
        #plt.xlabel('Number of satellites, q=' + str(q[i]))
        #plt.ylabel('length of vector')
    '''

    if q[i] == 0.5:
        dataiid05 = [All_data['4iidq0.5'], All_data['8iidq0.5'], All_data['12iidq0.5']]
        All_values['dataiid05'] = dataiid05
        #plt.plot(Num_satellites, dataiid05, label = 'IID_q0.5')
        dataNoniid05 = [All_data['4niidq0.5'], All_data['8niidq0.5'], All_data['12niidq0.5']]
        All_values['dataNoniid05'] = dataNoniid05
        #plt.plot(Num_satellites, dataNoniid05, label = 'NonIID_q0.5')
        datastrict05 = [All_data['4strictq0.5'], All_data['8strictq0.5'], All_data['12strictq0.5']]
        All_values['datastrict05'] = datastrict05
        #plt.plot(Num_satellites, datastrict05, label = 'Strict_q0.5')
        length_Theory05 = Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites,length_Theory3, label = 'Theory_0.5')
        length_upper05 = up_Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites, length_upper3, label = 'Upper_0.5')
        length_lower05 = low_Theory(q[i], Num_satellites)
        #plt.plot(Num_satellites, length_lower3, label = 'Lower_0.5')
        #plt.legend()
        #plt.xlabel('Number of satellites, q=' + str(q[i]))
        #plt.ylabel('length of vector')



df = pd.DataFrame(All_data, index=[0])
print(df)

df.to_csv(r'length_topk/length_parameters_topk.csv', index=False)







plt.show()
