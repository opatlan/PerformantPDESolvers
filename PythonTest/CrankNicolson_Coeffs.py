#Libraries
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.linalg as linalg
import statistics as st
import matplotlib.pyplot as plt 
import pandas
import math

#Initial parameters
Vol=0.2
int_Rate=0.05
Expiration=1.0
Strike=100
N_S=20
dS=(2*Strike)/N_S
dt=0.9/Vol**2/N_S**2
N_t=math.ceil(Expiration/dt)+1
dt=Expiration/N_t

#Initialize Stock Price Grid 
S=[10.0,] #Minimum stock price
for i in range(1,N_S):
    S.append(S[i-1]+dS)
print(S)

#Payoff boundary condition
V=[]
for i in range (N_S):
    V.append(max(S[i]-Strike,0.0))
print(V)

#Coefficient Calculation for Crank Nicolson Method
#Crank Nicolson 2
nu1=dt/(dS**2)
nu2=dt/dS
                                                                                            
A=[]                                                                            #Lower Diagonal Left Hand Side (k+1)
B=[]                                                                            #Main Diagonal Left Hand Side (k+1)
C=[]                                                                            #Upper Diagonal Left Hand Side (k+1)

nA=[]                                                                           #Lower Diagonal Right Hand Side (k)
nB=[]                                                                           #Main Diagonal Right Hand Side (k)
nC=[]                                                                           #Upper Diagonal Right Hand Side (k)

for i in range (1,N_S-1):                           #Middle part of stock price array without boundaries V_{0} and V_{N_S}      
    
    ak=-0.5*(Vol**2)*(S[i]**2)                                                  # a(s,t)
    bk=-int_Rate*S[i]                                                           # b(s,t)
    ck=int_Rate                                                                 # c(s,t)
    
    Ak=0.5*nu1*ak - 0.25*nu2*bk
    Bk=-nu1*ak + 0.5*dt*ck
    Ck=0.5*nu1*ak + 0.25*nu2*bk
    
    if(i==N_S-2):                                                       #Final row in grid apply boundary conditions 
        A.append(-Ak + Ck)
        B.append(1 - Bk - 2*Ck)
        C.append(-Ck)
        
        nA.append(Ak - Ck)
        nB.append(1 + Bk + 2*Ck)
        nC.append(Ck)
    else:                                                               #Compute coefficients for other rows
        A.append(-Ak)
        B.append(1 - Bk)
        C.append(-Ck)
        nA.append(Ak)
        nB.append(1 + Bk)
        nC.append(Ck)

A.pop(0)                                            #Remove first element of lower diagonal Left Hand Side (k+1)
C.pop()                                             #Remove last element of upper diagonal Left Hand Side (k+1)
nA.pop(0)                                           #Remove first element of lower diagonal Right Hand Side (k)
nC.pop()                                            #Remove last element of upper diagonal Right Hand Side (k)

print(A)
print(len(A))

print(B)
print(len(B))

print(C)
print(len(C))
