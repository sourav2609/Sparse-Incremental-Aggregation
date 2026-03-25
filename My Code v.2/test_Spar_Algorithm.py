import numpy as np
import torch
import copy
from nltk import flatten
from collections import OrderedDict
from torch import tensor
import operator
import math

def th_delete(tensor, indices): # To delete the indices from tensor
      mask = torch.ones(tensor.numel(), dtype=torch.bool)
      mask[indices] = False
      return tensor[mask]



def _size_data_TopK_WO_IA2(w_sparsification): # This function derives the size of AMP algorithm for one orbit
        '''
        p=0.1
        if p != 1:
                        x1 = copy.deepcopy(w_sparsification)
                        #print(f"$$$x1 = {x1}")
                        values = []
                        indices = []
                        for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                            b = x1[k] != 0
                            values.append(x1[k][b].tolist())
                            print(f"111 x1[k][b] = {x1[k][b]}")
                            #print(f"*@!&&&&&values == {values}")
                            check = torch.nonzero(b, as_tuple=True)
                            #indices1.append(b.nonzero().tolist())
                            #indices1.append(b.nonzero().tolist())
                            indices.append(check[0].tolist())                           
                            #print(f"222 b.nonzero().tolist() = {b.nonzero()}")                            
                            #print(f"check = {check[0]} ")
                            #print(f"*@!&&&&&indices == {indices1}")
        elif p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                            x1 = copy.deepcopy(w_sparsification)
                            #print(f"$$$x1 = {x1}")
                            values = []
                            indices = []
                            for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                                b = x1[k] != 1000000
                                values.append(x1[k][b])
                                #print(f"*@!&&&&&len(values) == {len(values[])}")
                                indices.append(b.nonzero().tolist())
                                #print(f"x1 = {x1}")
                            #print(f"###values = {w_sparsification.element_size()}")
        #print(f"%$$$indices = {indices}")
        print(f"## values = {values}")
        '''
        p = 0.1
        if p != 1:
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 0
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())


        elif p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 1000000
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())
                        #print(f"x1 = {x1}")
                    #print(f"###values = {w_sparsification.element_size()}")
        print(f"%$$$indices = {indices}")
            #################################################
        print(f"%$$$values = {values}")
                

                #print(f"%$$$indices = {indices}")    
        x0 = torch.tensor(copy.deepcopy(values))
                
        
        number_sat_each_orbit  = len(x0)
        number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
        group_index  = [0 for i in range(number_groups_each_orbit)]
        indexes = torch.arange(number_sat_each_orbit)
        print(x0)
        print(indexes)
        for idx in range(number_groups_each_orbit):

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0])


        
        group1 = []
        group2 = []
        group_all = []

        for idxx in range(len(group_index)):
            if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
               if group1 != []:

                   y1 = copy.deepcopy(group1)
                   
                   group1.append(torch.cat((y1[-1], x0[group_index[idxx][0]])))
                   #print(f"group1 = {group1}")
                   y2 = copy.deepcopy(group2)
                   group2.append(torch.cat((y2[-1],x0[group_index[idxx][1]])))
                   #print(f"group2 = {group2}")
               elif group1 == []:
                   group1.append(x0[group_index[idxx][0]])
                   group2.append(x0[group_index[idxx][1]])


            elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                y1 = copy.deepcopy(group1)
                group1.append(torch.cat((y1[-1],x0[group_index[idxx][0]])))

            elif  idxx == (len(group_index) - 1):
                group_all.append([group1,group2,x0])
        
        #print(f"####group_all = {group_all}")
        flattened_list = flatten(group_all)
        #print(flattened_list)
        sizee = 0
        for i in range(len(flattened_list)):
                   #flattened_list[i] = flattened_list[i].long()
                   #print(f"@@@ flattened_list[i] = {flattened_list[i]}")
                   flattened_list[i] = flattened_list[i].to(dtype=torch.float32)
                   #flattened_list[i] = flattened_list[i], dtype=torch.int32)
                   sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   sizee = sizee + np.log2(7850) * flattened_list[i].nelement()
                   #print(f"%%%%%%% = {flattened_list[i].nelement()}")
                   #print(f"$$$$$$$ = {flattened_list[i].element_size()}")
        #sizee = sizee + log2(7850)*
        #print(f" sizee = {sizee}")
        return sizee

def _size_data_TopK_WO_IA4(w_sparsification): # This function derives the size of AMP algorithm for one orbit
        
        
        if p != 1:
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 0
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())


        elif p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 1000000
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())
                        #print(f"x1 = {x1}")
                    #print(f"###values = {w_sparsification.element_size()}")
        print(f"%$$$indices = {indices}")
            #################################################
        print(f"%$$$values = {values}")
                

                #print(f"%$$$indices = {indices}")    
        x0 = copy.deepcopy(values)
                
        
        number_sat_each_orbit  = len(x0)
        number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
        group_index  = [0 for i in range(number_groups_each_orbit)]
        indexes = torch.arange(number_sat_each_orbit)
        print(x0)
        print(indexes)
        for idx in range(number_groups_each_orbit):

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0])


        
        group1 = []
        group2 = []
        group_all = []

        for idxx in range(len(group_index)):
            if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
               if group1 != []:

                   y1 = copy.deepcopy(group1)
                   
                   group1.append(torch.cat((y1[-1], x0[group_index[idxx][0]])))
                   #print(f"group1 = {group1}")
                   y2 = copy.deepcopy(group2)
                   group2.append(torch.cat((y2[-1],x0[group_index[idxx][1]])))
                   #print(f"group2 = {group2}")
               elif group1 == []:
                   group1.append(x0[group_index[idxx][0]])
                   group2.append(x0[group_index[idxx][1]])


            elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                y1 = copy.deepcopy(group1)
                group1.append(torch.cat((y1[-1],x0[group_index[idxx][0]])))

            elif  idxx == (len(group_index) - 1):
                group_all.append([group1,group2,x0])
        
        #print(f"####group_all = {group_all}")
        flattened_list = flatten(group_all)
        #print(flattened_list)
        sizee = 0
        for i in range(len(flattened_list)):
                   #flattened_list[i] = flattened_list[i].long()
                   #print(f"@@@ flattened_list[i] = {flattened_list[i]}")
                   flattened_list[i] = flattened_list[i].to(dtype=torch.float32)
                   #flattened_list[i] = flattened_list[i], dtype=torch.int32)
                   sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   sizee = sizee + np.log2(7850) * flattened_list[i].nelement()
                   #print(f"%%%%%%% = {flattened_list[i].nelement()}")
                   #print(f"$$$$$$$ = {flattened_list[i].element_size()}")
        #sizee = sizee + log2(7850)*
        #print(f" sizee = {sizee}")
        return sizee


def _size_data_TopK_WO_IA4(w_sparsification):
            p=0.1
            if p != 1:
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 0
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())


            elif p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                    x1 = copy.deepcopy(w_sparsification)
                    #print(f"$$$x1 = {x1}")
                    values = []
                    indices = []
                    for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                        b = x1[k] != 1000000
                        values.append(x1[k][b])
                        #print(f"*@!&&&&&len(values) == {len(values[])}")
                        indices.append(b.nonzero().tolist())
                        #print(f"x1 = {x1}")
                    #print(f"###values = {w_sparsification.element_size()}")
            #print(f"%$$$indices = {indices}")
            #################################################
            #print(f"%$$$values = {values}")
            x0 = copy.deepcopy(values)
            number_sat_each_orbit  = len(x0)
            number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
            group_index  = [0 for i in range(number_groups_each_orbit)]
            indexes = torch.arange(number_sat_each_orbit)



            for idx in range(number_groups_each_orbit): # For example, [0,1] which 0 is from one group and 1 from the other group

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0])

            

            print(f"#####group_index = {group_index}")
            group1 = []
            group2 = []
            group_all = []
            group1_ind = []
            group2_ind = []
            group_all_ind = []
            len_size_group = 0

            for idxx in range(len(group_index)):
                if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
                   #print(group_index[idxx])
                   if group1 != []:

                       y1 = copy.deepcopy(group1)
                       y1_ind = copy.deepcopy(group1_ind)


                       group1.append([y1[-1], x0[group_index[idxx][0]]])
                       same1_ind1 = [x for x in y1_ind[-1] if x in indices[group_index[idxx][0]]]
                       same1_ind2 = [x for x in indices[group_index[idxx][0]] if x in y1_ind[-1]]
                       same11_ind = same1_ind1 + same1_ind2
                       same1_ind = []
                       [same1_ind.append(item) for item in same11_ind if item not in same1_ind]

                       not_same2_ind1 = [x for x in y1_ind[-1] if x not in indices[group_index[idxx][0]]]
                       not_same2_ind2 = [x for x in indices[group_index[idxx][0]] if x not in y1_ind[-1]]
                       not_same22_ind = not_same2_ind1 + not_same2_ind2
                       not_same2_ind = []
                       [not_same2_ind.append(item) for item in not_same22_ind if item not in not_same2_ind]
                       all_index = same1_ind + not_same2_ind
                       group1_ind.append(all_index)
                       for k in range(len(group1_ind)):
                           for k1 in range(len(group1_ind[k])):
                             #print(f"111&&& group1_ind = {group1_ind}")
                             if group1_ind[k][k1] == []:
                                    del(group1_ind[k])



                       y2 = copy.deepcopy(group2)
                       y2_ind = copy.deepcopy(group2_ind)
                       group2.append([y2[-1], x0[group_index[idxx][1]]])
                       same1_ind1 = [x for x in y2_ind[-1] if x in indices[group_index[idxx][1]]]
                       same1_ind2 = [x for x in indices[group_index[idxx][1]] if x in y2_ind[-1]]
                       same11_ind = same1_ind1 + same1_ind2
                       same1_ind = []
                       [same1_ind.append(item) for item in same11_ind if item not in same1_ind]
                       not_same2_ind1 = [x for x in y2_ind[-1] if x not in indices[group_index[idxx][1]]]
                       not_same2_ind2 = [x for x in indices[group_index[idxx][1]] if x not in y2_ind[-1]]
                       not_same22_ind = not_same2_ind1 + not_same2_ind2
                       not_same2_ind = []
                       [not_same2_ind.append(item) for item in not_same22_ind if item not in not_same2_ind]
                       all_ind = same1_ind + not_same2_ind
                       group2_ind.append(all_ind)


                   elif group1 == []:

                       group1.append(x0[group_index[idxx][0]])
                       group1_ind.append(indices[group_index[idxx][0]])
                       #print(f"%%%%group1_ind = {x0[group_index[idxx][0]]}")
                       group2.append(x0[group_index[idxx][1]])
                       group2_ind.append(indices[group_index[idxx][1]])
                       #print(f"%%%%group2_ind = {x0[group_index[idxx][1]]}")


                elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                    #print(f"AAA = {group1_ind}")
                    y1 = copy.deepcopy(group1)
                    y1_ind = copy.deepcopy(group1_ind)
                    group1.append([y1[-1], x0[group_index[idxx][0]]])
                    #print(group1)

                    index_same1 = [x for x in y1_ind[-1] if x in indices[group_index[idxx][0]]]
                    index_same2 = [x for x in indices[group_index[idxx][0]] if x in y1_ind[-1]]
                    index11_same = index_same1 + index_same2
                    index1_same = []
                    [index1_same.append(item) for item in index11_same if item not in index1_same]
                    #print(index1_same)
                    index_not_same1 = [x for x in y1_ind[-1] if x not in indices[group_index[idxx][0]]]
                    index_not_same2 = [x for x in indices[group_index[idxx][0]] if x not in y1_ind[-1]]
                    index11_not_same = index_not_same1 + index_not_same2
                    index1_not_same = []
                    [index1_not_same.append(item) for item in index11_not_same if item not in index1_not_same]

                    jj4 = index1_same + index1_not_same


                    jj = max(jj4)
                    group1_ind.append(jj4)
                    #print(f"CCCC = {group1_ind}")

                elif  idxx == (len(group_index) - 1):
                    group_all.append([group1, group2, x0])
                    flatten_group1_ind =flatten(group1_ind)
                    #print(f"((( ))) = {flatten_group1_ind}")

                    flatten_group1_ind = list(set(flatten_group1_ind))
                    #print(f"CCC flatten_group1_ind= {flatten_group1_ind} ")
                    flatten_group2_ind =flatten(group2_ind)
                    flatten_group2_ind = list(set(flatten_group2_ind))
                    #print(f"HHH flatten_group2_ind= {flatten_group2_ind} ")
                    #print(f"###((( ))) = {flatten_group2_ind}")
                    index_not_same1 = [x for x in flatten_group2_ind if x not in flatten_group1_ind]
                    flatten_group1_ind = flatten_group1_ind + index_not_same1



                    indices = flatten(indices)
                    indices = list(set(indices)) # remove the duplicated items
                    jj2_ind = [x for x in indices if x not in flatten_group1_ind]
                    flatten_group1_ind = flatten_group1_ind + jj2_ind

                    #group_all.append([group1, group2, x0])
                    group_all_ind.append([group1_ind, group2_ind,flatten_group1_ind])
                    #group_all_ind.append([group1_ind, group2_ind, indices])

            #print(f"# %%%%%%%%%%%%% = {group_all_ind}")
            flattened_list = flatten(group_all)
            #print(flattened_list)

            flattened_list_same_index = []

            for i in range(len(flattened_list)):
                flattened_list_same_index.extend(flattened_list[i].tolist())
                #print(f"**** flattened_list[i] = {len(flattened_list[i])}")

            flattened_list_ind = flatten(group_all_ind)
            #print(f"flattened_list_ind = {flattened_list_ind}")
            sizee_value = 0
            sizee_ind = 0
            flattened_list_ind_tensor = torch.tensor(flattened_list_ind)
            flattened_list_ind_tensor1 = torch.tensor(flattened_list_ind_tensor)
            flattened_list_ind_tensor1 = flattened_list_ind_tensor1.to(dtype=torch.float32)
            print(f" 111 flattened_list_ind_tensor1.element_size() = {flattened_list_ind_tensor1.element_size()}")
            print(f" 222 flattened_list_ind_tensor1.nelement() = {flattened_list_ind_tensor1.nelement()}")
            sizee_value = flattened_list_ind_tensor1.element_size() * flattened_list_ind_tensor1.nelement() * 8
            #print(flattened_list_ind_tensor1.element_size())
            #print(f"TopK flattened_list[i].element_size() = {flattened_list_ind_tensor1.nelement()}")
            sizee_position = np.log2(7850) * flattened_list_ind_tensor1.nelement() # for each Position entry , we need log2(7850)
            #print(sizee_position)
            #print(f"TopK flattened_list[i].element_size() = {flattened_list_ind_tensor1.nelement()}")
            #print(f"$$$$flattened_list_ind_tensor1.nelement() = {flattened_list_ind_tensor1.nelement()}")
            #print(f"###sizee_value = {sizee_value}")
            size_transmission = sizee_value + sizee_position
            print(f"333 size_transmission = {size_transmission}")
            return size_transmission


w_sparsification = torch.tensor([[0,0,13], [16, 22,23], [31,32,33], [41,42,44], [51,52,53], [61,62,63], [71, 72, 73], [81,82,83]])
#w_sparsification = torch.tensor()
print(_size_data_TopK_WO_IA4(w_sparsification))
#print(_size_data_TopK_WO_IA(w_sparsification))


'''

def _size_data_TopK_WO_IA1(w_sparsification): # This function derives the size of AMP algorithm for one orbit

        x0 = copy.deepcopy(w_sparsification)
        number_sat_each_orbit  = len(x0)
        number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
        group_index  = [0 for i in range(number_groups_each_orbit)]
        indexes = torch.arange(number_sat_each_orbit)
        print(x0)
        print(indexes)
        for idx in range(number_groups_each_orbit):

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0])


        
        group1 = []
        group2 = []
        group_all = []

        for idxx in range(len(group_index)):
            if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
               if group1 != []:

                   y1 = copy.deepcopy(group1)
                   
                   group1.append(torch.cat((y1[-1], x0[group_index[idxx][0]])))
                   #print(f"group1 = {group1}")
                   y2 = copy.deepcopy(group2)
                   group2.append(torch.cat((y2[-1],x0[group_index[idxx][1]])))
                   #print(f"group2 = {group2}")
               elif group1 == []:
                   group1.append(x0[group_index[idxx][0]])
                   group2.append(x0[group_index[idxx][1]])


            elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                y1 = copy.deepcopy(group1)
                group1.append(torch.cat((y1[-1],x0[group_index[idxx][0]])))

            elif  idxx == (len(group_index) - 1):
                group_all.append([group1,group2,x0])
        
        #print(f"####group_all = {group_all}")
        flattened_list = flatten(group_all)
        #print(flattened_list)
        sizee = 0
        for i in range(len(flattened_list)):
                   #flattened_list[i] = flattened_list[i].long()
                   #print(f"@@@ flattened_list[i] = {flattened_list[i]}")
                   flattened_list[i] = flattened_list[i].to(dtype=torch.float32)
                   #flattened_list[i] = flattened_list[i], dtype=torch.int32)
                   sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   sizee = sizee + np.log2(7850) * flattened_list[i].nelement()
                   #print(f"%%%%%%% = {flattened_list[i].nelement()}")
                   #print(f"$$$$$$$ = {flattened_list[i].element_size()}")
        #sizee = sizee + log2(7850)*
        #print(f" sizee = {sizee}")
        return sizee


def _size_data_TopK_WO_IA1(w_sparsification): # This function derives the size of AMP algorithm for one orbit

        x0 = copy.deepcopy(w_sparsification)
        number_sat_each_orbit  = len(x0)
        number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
        group_index  = [0 for i in range(number_groups_each_orbit)]
        indexes = torch.arange(number_sat_each_orbit)
        print(x0)
        print(indexes)
        for idx in range(number_groups_each_orbit):

                if len(indexes) > 2:
                    z = [indexes[0],indexes[1]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0,1])
                elif len(indexes) <= 2:
                    z = [indexes[0]]
                    group_index[idx] = z
                    indexes = th_delete(indexes, [0])


        
        group1 = []
        group2 = []
        group_all = []

        for idxx in range(len(group_index)):
            if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
               if group1 != []:

                   y1 = copy.deepcopy(group1)
                   
                   group1.append(torch.cat((y1[-1], x0[group_index[idxx][0]])))
                   #print(f"group1 = {group1}")
                   y2 = copy.deepcopy(group2)
                   group2.append(torch.cat((y2[-1],x0[group_index[idxx][1]])))
                   #print(f"group2 = {group2}")
               elif group1 == []:
                   group1.append(x0[group_index[idxx][0]])
                   group2.append(x0[group_index[idxx][1]])


            elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                y1 = copy.deepcopy(group1)
                group1.append(torch.cat((y1[-1],x0[group_index[idxx][0]])))

            elif  idxx == (len(group_index) - 1):
                group_all.append([group1,group2,x0])
        
        #print(f"####group_all = {group_all}")
        flattened_list = flatten(group_all)
        #print(flattened_list)
        sizee = 0
        for i in range(len(flattened_list)):
                   #flattened_list[i] = flattened_list[i].long()
                   #print(f"@@@ flattened_list[i] = {flattened_list[i]}")
                   flattened_list[i] = flattened_list[i].to(dtype=torch.float32)
                   #flattened_list[i] = flattened_list[i], dtype=torch.int32)
                   sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                   sizee = sizee + np.log2(7850) * flattened_list[i].nelement()
                   #print(f"%%%%%%% = {flattened_list[i].nelement()}")
                   #print(f"$$$$$$$ = {flattened_list[i].element_size()}")
        #sizee = sizee + log2(7850)*
        #print(f" sizee = {sizee}")
        return sizee


def _size_data_TopK_WO_IA(w_sparsification): # This function derives the size of AMP algorithm for one orbit
                p=0.1
                if p != 1:
                        x1 = copy.deepcopy(w_sparsification)
                        #print(f"$$$x1 = {x1}")
                        values = []
                        indices = []
                        for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                            b = x1[k] != 0
                            values.append(x1[k][b].tolist())
                            print(f"111 x1[k][b] = {x1[k][b]}")
                            #print(f"*@!&&&&&values == {values}")
                            check = torch.nonzero(b, as_tuple=True)
                            #indices1.append(b.nonzero().tolist())
                            #indices1.append(b.nonzero().tolist())
                            indices.append(check[0].tolist())                           
                            #print(f"222 b.nonzero().tolist() = {b.nonzero()}")                            
                            #print(f"check = {check[0]} ")
                            #print(f"*@!&&&&&indices == {indices1}")
                elif p == 1:   # The gradient vector has some zero entries by itself, this if consider that for large p
                            x1 = copy.deepcopy(w_sparsification)
                            #print(f"$$$x1 = {x1}")
                            values = []
                            indices = []
                            for k in range(len(x1)):    # This for loop removes the zero elements from each nested element.
                                b = x1[k] != 1000000
                                values.append(x1[k][b])
                                #print(f"*@!&&&&&len(values) == {len(values[])}")
                                indices.append(b.nonzero().tolist())
                                #print(f"x1 = {x1}")
                            #print(f"###values = {w_sparsification.element_size()}")
                #print(f"%$$$indices = {indices}")
                print(f"## values = {values}")

                

                #print(f"%$$$indices = {indices}")    
                x0 = torch.tensor(copy.deepcopy(values))
                print(f"11x0 = {x0}")
                number_sat_each_orbit  = len(x0)
                number_groups_each_orbit = math.floor(number_sat_each_orbit/2)+1    # grouping the satellite in each orbit based on their position to transmit the weights
                group_index  = [0 for i in range(number_groups_each_orbit)]
                indexes = torch.arange(number_sat_each_orbit)
                print(indexes)

                for idx in range(number_groups_each_orbit):

                        if len(indexes) > 2:
                            z = [indexes[0],indexes[1]]
                            group_index[idx] = z
                            indexes = th_delete(indexes, [0,1])
                        elif len(indexes) <= 2:
                            z = [indexes[0]]
                            group_index[idx] = z
                            indexes = th_delete(indexes, [0])



                group1 = []
                group2 = []
                group_all = []

                for idxx in range(len(group_index)):
                    if len(group_index[idxx]) == 2 and idxx != len(group_index)-1:
                      if group1 != []:

                        y1 = copy.deepcopy(group1)
                        
                        group1.append(torch.cat((y1[-1], x0[group_index[idxx][0]])))
                        print(f"group1 = {group1}")
                        y2 = copy.deepcopy(group2)
                        group2.append(torch.cat((y2[-1],x0[group_index[idxx][1]])))
                        print(f"group2 = {group2}")
                    elif group1 == []:
                        group1.append(x0[group_index[idxx][0]])
                        group2.append(x0[group_index[idxx][1]])


                    elif len(group_index[idxx]) == 1 and idxx != len(group_index)-1:
                        y1 = copy.deepcopy(group1)
                        group1.append(torch.cat((y1[-1],x0[group_index[idxx][0]])))

                    elif  idxx == (len(group_index) - 1):
                        group_all.append([group1,group2,x0])
                
                #print(f"####group_all = {group_all}")
                flattened_list = flatten(group_all)
                print(flattened_list)
                sizee = 0
                for i in range(len(flattened_list)):
                        #flattened_list[i] = flattened_list[i].long()
                        #print(f"@@@ flattened_list[i] = {flattened_list[i]}")
                        flattened_list[i] = flattened_list[i].to(dtype=torch.float32)
                        #flattened_list[i] = flattened_list[i], dtype=torch.int32)
                        sizee = sizee + (flattened_list[i].element_size() * flattened_list[i].nelement() * 8)
                        sizee = sizee + np.log2(7850) * flattened_list[i].nelement()
                        #print(f"%%%%%%% = {flattened_list[i].nelement()}")
                        #print(f"$$$$$$$ = {flattened_list[i].element_size()}")
                #sizee = sizee + log2(7850)*
                print(f" sizee = {sizee}")
                return sizee

h1 = 2000*(10**3)
h2 = 2000*(10**3)
rE = 6371*(10**3)
sintheta1 = (np.sqrt(h1*(2*rE+h1)))/(rE+h1)
sintheta2 = (np.sqrt(h2*(2*rE+h2)))/(rE+h2)
costheta1 = (rE)/(rE+h1)
costheta2 = (np.sqrt(h2*(2*rE+h2)))/(rE+h2)
theta1 = np.arcsin(sintheta1)
theta2 = np.arcsin(sintheta2)
theta1 = np.arccos(costheta1)
theta  = theta1*2
print(theta)
print(theta*180/np.pi)

costheta1 = (rE)/(rE+h1)
print(costheta1)
theta1 = 2*np.arccos(costheta1)
print(theta1)
print(theta1*180/np.pi)

print(0.18 + 166.06 - 13 - 13)


p = 0.8
w_residual_user = OrderedDict([('linear', tensor([1.0, 2, 8])), ('weight', tensor([3.0, 4]))])
w_sparsification1 = OrderedDict([('linear', tensor([0.0, 3, 5])), ('weight', tensor([9.0, 10]))])
idx = 0

def _Strom_Threshold_sparsification_all(w_sparsification, w_residual_user, idx, p): # Based on the value of maximum

            w_sparsification1 = copy.deepcopy(w_sparsification)
            w_sparsification2 = copy.deepcopy(w_sparsification)

            operator1 = operator.add
            w_sum_train_residual = sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator1) # % sum resifual and new receive global model
            all_entries = change_weight_vector(w_sum_train_residual)   # % Change the form of w_sum_train_residual (in the form of global model) to one tensor vector:
                                                                            # % For Examle: w_sum_train_residual = OrderedDict([('linear', tensor([ 1,  5, 13])), ('weight', tensor([12, 14]))])
                                                                            # % all_entries = tensor([ 1.,  5., 13., 12., 14.])

            w_sparsification2 = Spar_Threshold(w_sum_train_residual, all_entries, p)   # Makes the received weights global and in the form of global matrix


            operator2 = operator.sub
            w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
            w_residual_new = sum_subtraktion_residual_weight(w_sum_train_residual_main, w_sparsification2, operator2) # % sum resifual and new receive global model

            return w_sparsification2, w_residual_new

def sum_subtraktion_residual_weight(w_sparsification1, w_residual_user, operator):

        w_sum_train_residual = copy.deepcopy(w_residual_user)
        for key_idx in range(len(w_residual_user)): # w_all = w_global + w_residual
              values_key_idx1 = list(w_sparsification1.values())[key_idx]
              values_key_idx2 = list(w_residual_user.values())[key_idx]
              values_key_idx3 = operator(values_key_idx1 , values_key_idx2)
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3
        return w_sum_train_residual

def Spar_Threshold(w_sum_train_residual, all_entries, p): # % Makes the vector sparse

        w_sparsification2 = copy.deepcopy(w_sum_train_residual)
        w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
        for g in range(len(w_sum_train_residual_main)):
                             w_each_dic = list(w_sum_train_residual_main.values())[g]   # % Values of each dict_key
                             w_each_dic_zero = torch.zeros(w_each_dic.shape)

                             all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
                             number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
                             sort_flatten_matrix = torch.sort(all_entries_abs)
                             number_tensor_accepted = int(number_tensor * p)
                             print(f"number_tensor_accepted = {number_tensor_accepted}")
                             if number_tensor_accepted == 0:
                                 threshold = 100000
                             else:
                                 threshold = sort_flatten_matrix[0][number_tensor - number_tensor_accepted]

                             w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(w_each_dic)>=threshold, w_each_dic, w_each_dic_zero)
        return w_sparsification2

def change_weight_vector(w_sum_train_residual):   # Combine each component of w_sum_train_residual
            all_entries_sep_values = []
            for g in range(len(w_sum_train_residual)):

               values_tensor = list(w_sum_train_residual.values())[g].float()
               flatten_values_tensor = torch.flatten(values_tensor)
               all_entries_sep_values.append(flatten_values_tensor)

            #print(f"all_entries_sep_values = {all_entries_sep_values}")
            all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
            all_entries = torch.flatten(all_entries_sep_values_cat)
            return all_entries


w_sparsification2, w_residual_new = _Strom_Threshold_sparsification_all(w_sparsification1, w_residual_user, idx, p)
print(f"w_sparsification2 = {w_sparsification2}")
print(f"w_residual_new = {w_residual_new}")


w_sum_train_residual = copy.deepcopy(w_sparsification1)
w_sparsification2 = copy.deepcopy(w_sparsification1)

for key_idx in range(len(w_residual_user)): # w_all = w_global + w_residual
              values_key_idx1 = list(w_residual_user.values())[key_idx]
              values_key_idx2 = list(w_sparsification1.values())[key_idx]
              values_key_idx3 = values_key_idx1 + values_key_idx2
              w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3

print(f"w_sum_train_residual = {w_sum_train_residual}")


all_entries_sep_values = []
for g in range(len(w_sum_train_residual)):

               values_tensor = list(w_sum_train_residual.values())[g].float()
               flatten_values_tensor = torch.flatten(values_tensor)
               all_entries_sep_values.append(flatten_values_tensor)

all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
all_entries = torch.flatten(all_entries_sep_values_cat)
print(f"all_entries = {all_entries}")


w_sum_train_residual_main = copy.deepcopy(w_sum_train_residual)
for g in range(len(w_sum_train_residual_main)):
                     w_each_dic = list(w_sum_train_residual_main.values())[g]   # % Values of each dict_key
                     w_each_dic_zero = torch.zeros(w_each_dic.shape)

                     all_entries_abs = torch.abs(all_entries)   # % sparsification based on absolute value
                     number_tensor = torch.numel(all_entries)   # % The number of entries in all_entries
                     sort_flatten_matrix = torch.sort(all_entries_abs)
                     number_tensor_accepted = int(number_tensor * p)
                     if number_tensor_accepted == 0:
                         threshold = 100000
                     else:
                         threshold = sort_flatten_matrix[0][number_tensor - number_tensor_accepted]

                     w_sparsification2[list(w_sparsification2.keys())[g]] = torch.where(abs(w_each_dic)>=threshold, w_each_dic, w_each_dic_zero)

print(f" w_sparsification2 = {w_sparsification2}")
operator = operator.sub
#w_sum_train_residual = copy.deepcopy(w_residual_user)
for key_idx in range(len(w_sum_train_residual)): # w_all = w_global + w_residual
      values_key_idx1 = list(w_sum_train_residual.values())[key_idx]
      values_key_idx2 = list(w_sparsification2.values())[key_idx]
      values_key_idx3 = operator(values_key_idx1 , values_key_idx2)
      w_sum_train_residual[list(w_residual_user.keys())[key_idx]] = values_key_idx3

print(w_sum_train_residual)

'''
