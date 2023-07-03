import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

#stevilo sprejetih potez(T)


st_kroglic = 17
postavitev = []
energija = []




#zacetna energija
alpha = 2


T = [0.1,0.2,0.3,0.7,1,2,3,5,10]


N = 100000
spremembe_en = []
stevec0 = 0
stevec1 = 0
st=[[],[]]

for j in range(len(T)):
    postavitev = [0, -3, -16, -9, 0, -2, -10, -1, -2, -11, -12, 0, -13, -13, -9, -1, 0]
    E = alpha*sum(postavitev)
    for i in range(st_kroglic-1):
        E = E + 1/2*(postavitev[i+1]-postavitev[i])**2
        energija.append(E)
    for i in range(N):
        a = random.randint(1,15)
        sprememba = 1 if random.random() < 0.5 else -1
        postavitev[a] = postavitev[a] + sprememba
        if postavitev[a] < -18:
            postavitev[a] = -18
        if postavitev[a] > 0:
            postavitev[a] = 0    
        deltaE = sprememba**2 - sprememba*(postavitev[a+1]-2*postavitev[a]+postavitev[a-1]-alpha)
        spremembe_en.append(deltaE)
        if deltaE < 0:
            E = E + deltaE
            stevec0 = stevec0 + 1
        else:
            if random.random() <= math.exp(-deltaE/T[j]):
                E = E + deltaE
                stevec1 = stevec1 + 1
            else:
                postavitev[a] = postavitev[a] - sprememba    
        energija.append(E)
    st[0].append(stevec0/N)
    st[1].append(stevec1/N)    
    stevec0 = 0
    stevec1 = 0 
        

"""
stevilo = np.linspace(0,N,N+1)
plt.plot(stevilo,energija)
plt.xscale("log")
plt.show()
print(f"koncna postavitev = {postavitev}")


kroglice = np.linspace(0,st_kroglic,st_kroglic)
#plt.plot(kroglice,zacetna,"-ro",label="zacetna postavitev")
plt.plot(kroglice, postavitev,"-bo", label="koncna postavitev")
plt.legend()
plt.show()
"""


plt.plot(T,st[0],"o-", label="zmanjša en")
plt.plot(T,st[1],"o-", label="poveča en")
plt.legend()
plt.xlabel("T")
plt.ylabel("P")
plt.show()
