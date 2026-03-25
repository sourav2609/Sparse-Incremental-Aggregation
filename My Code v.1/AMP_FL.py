## AMP implementation written by Nasrin Razmi from Matlab implementation on https://sites.google.com/site/ampandcamp/demo-1
## This is for noiseless setting
import numpy as np
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import torch
np.random.seed(1)
#torch.set_printoptions(precision=25)
#torch.set_printoptions(profile="full")
#np.set_printoptions(precision=25)
#from decimal import *
#getcontext().prec = 25

# Inputs:
#   y :          observations
#   A :          a function handle that represents matrix A, A(x,1) means
#                A*x, A(x,2) means A'*x
#   Eta :        a function handle which is a generic denoiser,
#                xhat=Eta(temp,sigma)
#   Etader :     a function handle which is the derivative function of the
#                denoise function Eta, if you can't provide this derivative
#                function, please input "Null"
#   niter :      the maximum number of iterations
#   par:         a cell with two elements, the first denotes whether we need
#                all the estimates in whole process or just the final estimate
#                we obtain, "1" means need, "0" means not; the second
#                denotes how many iteration you want AMP to run, if you
#                input a positive integer t, then AMP runs t times
#                iterations for you, if you input the string 'Auto', then
#                AMP will try to runs 100 times iterations and stop when
#                the ratio of l2 norm of (x_(t+1)-x_(t)) and l2 norm of
#                x_(t) less then 0.01

'''
def flatten_weights(weights):

                all_entries_sep_values = []
                for g in range(len(weights)):

                   values_tensor = list(weights.values())[g].float()
                   flatten_values_tensor = torch.flatten(values_tensor)
                   all_entries_sep_values.append(flatten_values_tensor)

                #print(f"all_entries_sep_values = {all_entries_sep_values}")
                all_entries_sep_values_cat = torch.cat((all_entries_sep_values[0], all_entries_sep_values[1]))
                all_entries = torch.flatten(all_entries_sep_values_cat)
                return all_entries
'''


def AMP_Implement_Sat(v, N, n):

      np.random.seed(1)
      vector_input = v
      par = 1
      A = np.random.normal(loc=0, scale=1/np.sqrt(n), size=(n, N))
      #print(f" 111 A[0] = {A[0]}")
      #print(f" 222 vector_input = {vector_input[7820:7850]}")
      #check_point = np.multiply(A[0], vector_input)
      #sum_check = sum(check_point)
      #print(f" $$$ sum_check  = {sum_check}")
      #A = torch.normal(0, 1/np.sqrt(n), size=(n, N))
      #print(f"A[0] = {A[0]}, np.size(A) = {np.shape(A)}")
      if  par == 1:

          #print(np.shape(A), np.shape(vector_input))
          #vector_input.numpy()
          #print(vector_input.dtype)
          #A = A.astype('float32')
          #vector_input = vector_input.astype('float32')
          #print(A.dtype)

          #print(type(A), type(vector_input))
          #vector_input = vector_input.astype('float32')

          newv = np.matmul(A,vector_input)
      else:
          newv = np.matmul(A.transpose(), vector_input)
      return newv






def AMP_Implement_PS(sparsified_weights, N, n):
    np.random.seed(1)
    A = np.random.normal(loc=0, scale=1/np.sqrt(n), size=(n, N))
    #print(f" 222 A[0] = {A[0]}")
    #print(f"A[0] = {A[0]}")
    #N = 5000
    #N = len(sparsified_vector)
    #n = 1500
    #k = 100 # sparsity


    #sparsified_vector = flatten_weights(sparsified_weights)
    sparsified_vector = copy.deepcopy(sparsified_weights.numpy())
    #print(f" sparsified_vector = {sparsified_vector}")
    #print(f"sparsified_vector ={sparsified_vector}")
    def Ax_represent(A,x,i):
        if i == 1:
           Aout = np.matmul(A, x)
        elif i==2:
           Aout = np.matmul(A.transpose(), x)
        return Aout

    def Eta(temp, sigma):   # Y = sign(X)*((|X| - T)+)
        Y_out = np.sign(temp) * (np.abs(temp) - sigma) * (np.abs(temp) > sigma)
        return Y_out

    def Eta_der_Estimate(v,sigma,Eta):

       p = len(v)
       sigma_zz = 1e-3
       zz = np.random.randn(p,1)
       ed = sum( zz*( Eta(v+zz*sigma_zz, np.sqrt(sigma**2+sigma_zz**2))-Eta(v,sigma) ))/sigma_zz/p
       return ed

    def normalize(DtmNorms, n, N):

        if (sum(DtmNorms>1.1) + sum(DtmNorms<0.9))>0:

              tempA = np.zeros((n,N))
              tempA_ra = np.zeros((n,N))
              normalize_time_total = 0
              for j in range(N):
                  tempA[:,j] =  Ax_represent(A,I[:,j],1)
              for j in range(N):
                  tempA_ra[:,j] = tempA[:,j] - np.mean(tempA[:,j])

              colnormA = np.transpose(np.sqrt(sum(tempA_ra**2,0)))
              #ind = colnormA.find(0)
              #colnormA[ind] = np.transpose(np.sqrt(sum((abs(tempA[:,ind]))**2,1)))
        else:
              colnormA = np.ones((N,1))
        return colnormA

    def SoftThresholdDer(temp, sigma):
        Y_out = np.abs(temp) > sigma
        return Y_out


    y1 = copy.deepcopy(sparsified_vector)
    y2 = np.array([y1])
    y = np.transpose(y2)
    #print(f"y = {y}")
    #y = SNM(x0, 1, A, N, n)
    lengthN = Ax_represent(A, np.zeros((n,1)), 2)
    N = len(lengthN)
    pick = np.random.permutation(N)


    DtmIndex = pick[0:5]
    DtmNorms= np.zeros((5,1))
    I = np.eye(N)

    for i in range(5):
        DtmNorms[i] = LA.norm(np.matmul(A, I[:, DtmIndex[i]]))**2


    colnormA1 = normalize(DtmNorms, n, N)
    if np.shape(colnormA1) == (N,):
        colnormA2 = np.array([colnormA1])
        colnormA = np.transpose(colnormA2)
    else:
        colnormA = copy.deepcopy(colnormA1)

    # Denote normalized A matrix as AA, then, when we calculate AA*v, we need
    # to do A*(v./colnormA); when we calculate AA'*v, we need to do (A'*v)./colnormA.

    ###############################%%%%%
    par = [True, 4000]
    par1 = par[0]
    par2 = par[1]

    niter = par2

    empiricaliterwatch_sigma = np.zeros((niter, 1))
    xall = np.zeros((N, niter))
    #print(f"@@@xall = {xall[:,0]}")
    mx = np.zeros((N,1))
    mz = y - Ax_represent(A, mx/colnormA, 1)
    #print(f"mz = {mz}")
    Etader = 'Null'
    for iter in range(niter):

           temp_z = Ax_represent(A,mz,2) / colnormA + mx
           sigma_hat= LA.norm(mz) / np.sqrt(n)
           mx = Eta(temp_z, sigma_hat)
           
           #print(f"### mz = {mz}")
           if Etader.lower()=='Null'.lower():
              mz = y - Ax_represent(A, mx/colnormA, 1) + mz * Eta_der_Estimate(temp_z, sigma_hat, Eta) * N/n
           else:
              mz = y - Ax_represent(A, mx/colnormA, 1) + mz * sum(SoftThresholdDer(temp_z, sigma_hat)) / n

           
           empiricaliterwatch_sigma[iter]=sigma_hat
           y1 = mx/colnormA
           #if iter <= 10:
           #    print(f"y1 = {y1}")            
           y2 = np.transpose(y1)[0]
           xall[:, iter] = y2
           #if iter <= 10:
           #    print(f"y2 = {y2}")

           if niter==7200 and abs(empiricaliterwatch_sigma[iter]-LA.norm(mz)/np.sqrt(n))<0.00000000000001:
           #if niter==100 and abs(empiricaliterwatch_sigma[iter])<0.001:
             break

           empiricaliterwatch_sigma[iter] = LA.norm(mz)/np.sqrt(n)
           #print(f"empiricaliterwatch_sigma = {empiricaliterwatch_sigma[iter]}")
           #print(f"xhat = {xall[:,iter]}")
           '''
           if niter == 100:
               #empiricaliterwatch_sigma[iter] = empiricaliterwatch_sigma[0:iter]
               xall = xall[:, 0:iter]

           if par1 == True:
              xhat = xall
           else:
              xhat = xall[:,-1]

                         #if iter == 10:
              print(f"111 y = {y}")
              print(f"222  Ax_represent(A, mx/colnormA, 1) = {Ax_represent(A, mx/colnormA, 1)}")
              print(f"333  mz * sum(SoftThresholdDer(temp_z, sigma_hat)) = {mz * sum(SoftThresholdDer(temp_z, sigma_hat))}")
           '''
    #print(f"iter = {iter}")
    #print(f"xhat = {xall[:,iter]}")
    #print(f"~~xhat = {len(xall[:,iter])}")
    #print(f"empiricaliterwatch_sigma = {empiricaliterwatch_sigma[iter]}")
    #z = len(empiricaliterwatch_sigma)
    #x = np.arange(z)
    #plt.plot(x, empiricaliterwatch_sigma)
    #plt.show()
    return xall[:,iter], empiricaliterwatch_sigma[iter]
