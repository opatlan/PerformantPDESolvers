import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.linalg as linalg
import statistics as st
import matplotlib.pyplot as plt 
import pandas
import math

Vol=0.2
int_Rate=0.05
Expiration=1.0
Strike=100
N_S=20
N_t= 20
dS=(2*Strike)/N_S
dt=Expiration/N_t

S = [x*dS for x in range (N_S)]
print(S)

V=[]
for i in range (N_S):
    V.append(max(S[i]-Strike,0.0))
print(V)

#Crank Nicolson Boundary Conditions 1st and last row
def CrankNicolson(Vol,S,int_Rate):
    nu1=dt/(dS**2)
    nu2=dt/dS
    A=[]
    B=[]
    C=[]
    nA=[]
    nB=[]
    nC=[]
    for i in range (1,N_S-1):
        ak=0.5*(Vol**2)*(S[i]**2)
        bk=int_Rate*S[i]
        ck=-int_Rate

        Ak=0.5*nu1*ak - 0.25*nu2*bk
        Bk=-nu1*ak + 0.5*dt*ck
        Ck=0.5*nu1*ak + 0.25*nu2*bk

        if (i==1):
            A.append(-Ak)
            B.append(1 - Bk -2*Ak)
            C.append(-Ck + Ak)

            nA.append(Ak)
            nB.append(1 + Bk + 2*Ak)
            nC.append(Ck - Ak)
        elif(i==N_S-2):     
            A.append(-Ak + Ck)
            B.append(1 - Bk - 2*Ck)
            C.append(-Ck)

            nA.append(Ak - Ck)
            nB.append(1 + Bk + 2*Ck)
            nC.append(Ck)
        else:
            A.append(-Ak)
            B.append(1 - Bk)
            C.append(-Ck)
            nA.append(Ak)
            nB.append(1 + Bk)
            nC.append(Ck)
    A.pop(0)
    bndC=C.pop()
    nA.pop(0)
    nC.pop()
    return A,B,C,nA,nB,nC

#Crank Nicolson Boundary Conditions last row
def CrankNicolson2(Vol,S,int_Rate):
    nu1=dt/(dS**2)
    nu2=dt/dS
    A=[]
    B=[]
    C=[]
    nA=[]
    nB=[]
    nC=[]
    for i in range (1,N_S-1):
        ak=0.5*(Vol**2)*(S[i]**2)
        bk=int_Rate*S[i]
        ck=-int_Rate

        Ak=0.5*nu1*ak - 0.25*nu2*bk
        Bk=-nu1*ak + 0.5*dt*ck
        Ck=0.5*nu1*ak + 0.25*nu2*bk

        if(i==N_S-2):     
            A.append(-Ak + Ck)
            B.append(1 - Bk - 2*Ck)
            C.append(-Ck)

            nA.append(Ak - Ck)
            nB.append(1 + Bk + 2*Ck)
            nC.append(Ck)
        else:
            A.append(-Ak)
            B.append(1 - Bk)
            C.append(-Ck)
            nA.append(Ak)
            nB.append(1 + Bk)
            nC.append(Ck)
    A.pop(0)
    bndC=C.pop()
    nA.pop(0)
    nC.pop()        
    return A,B,C,nA,nB,nC

#Implicit Boundary Conditions last row
def Implicit(Vol,S,int_Rate):
    nu1=dt/(dS**2)
    nu2=dt/dS
    A=[]
    B=[]
    C=[]

    for i in range (1,N_S-1):
        ak= 0.5*(Vol**2)*(S[i]**2)
        bk= int_Rate*S[i]
        ck= -int_Rate
        
        Ak= -nu1*ak + 0.5*nu2*bk
        Bk= 2*nu1*ak - dt*ck
        Ck= -nu1*ak - 0.5*nu2*bk
        
        if(i==N_S-2):     
            A.append(Ak - Ck)
            B.append(1 + Bk + 2*Ck)
            C.append(Ck)
            
        else:
            A.append(Ak)
            B.append(1 + Bk)
            C.append(Ck)
    A.pop(0)
    bndC=C.pop()
    return A,B,C

A,B,C,nA,nB,nC=CrankNicolson(Vol,S,int_Rate)

#Build LHS and RHS Matrices
def Build2Mat(A,B,C,nA,nB,nC):
    diag_lhs = [A,B,C]
    diag_rhs = [nA,nB,nC]
    lhs=sparse.diags(diag_lhs, [-1,0,1]).toarray()
    rhs=sparse.diags(diag_rhs, [-1,0,1]).toarray()
    return lhs,rhs

#Build LHS Matrix
def Build1Mat(A,B,C):
    diag_lhs = [A,B,C]
    lhs=sparse.diags(diag_lhs, [-1,0,1]).toarray()
    return lhs

lhs,rhs=Build2Mat(A,B,C,nA,nB,nC)

Vtemp=V[1:N_S-1]
Vtemp=np.array(Vtemp)
print(Vtemp)

#Solve Iteratively Crank Nicolson
def SolveCN(lhs,rhs,Vtemp,N_S):
    for i in range (N_S):
        Vmul=rhs@Vtemp
        Vres=linalg.solve(lhs,Vmul)
        Vtemp=Vres
        print(Vres)

#Solve Iteratively Implicit
def SolveImp(lhs,Vtemp,N_S):
    for i in range (N_S):
        Vres=linalg.solve(lhs,Vtemp)
        Vtemp=Vres
        print(Vres)

SolveCN(lhs,rhs,Vtemp,N_S)
