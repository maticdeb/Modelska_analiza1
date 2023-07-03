import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

polje_zac=[] 

sprememba = 0
N = 50
for i in range(N): 
    row=[] 
    for j in range(N): 
        row.append(1) 
    polje_zac.append(row) 

polje = polje_zac



E1 = 0
H = 0

for k in range(1,N-1):
    for l in range(1,N-1):
        E1 = E1 - polje[k][l]* (polje[k-1][l] + polje[k+1][l] + polje[k][l+1] + polje[k][l-1]) - H*polje[k][l]


for i in range(1,N-1):
    E1 = E1 - polje[0][i]*(polje[1][i] + polje[0][i-1] + polje[0][i+1]) - H*polje[0][i]
    E1 = E1 - polje[N-1][i]*(polje[N-2][i] + polje[N-1][i-1] + polje[N-1][i+1]) - H*polje[N-1][i]
    E1 = E1 - polje[i][0]*(polje[i][1] + polje[i-1][0] + polje[i+1][0]) - H*polje[i][0]
    E1 = E1 - polje[i][N-1]*(polje[i][N-2] + polje[i-1][N-1] + polje[i+1][N-1]) -H*polje[i][N-1]
E1 = E1 - polje[0][0]*(polje[1][0] + polje[0][1]) - H*polje[0][0]
E1 = E1 - polje[0][N-1]*(polje[1][N-1] + polje[0][N-2]) - H*polje[0][N-1]
E1 = E1 - polje[N-1][0]*(polje[N-2][0] + polje[N-1][1]) - H*polje[N-1][0]
E1 = E1 - polje[N-1][N-1]*(polje[N-2][N-1] + polje[N-1][N-2]) - H*polje[N-1][N-1]

E = E1

magnetizacija = []
T = [1,1.5,2,2.3,2.4,2.5,2.7,3,3.5,4,5]

M = 100*N*N
for j in range(len(T)):
    energija = []
    energija.append(E)
    for i in range(M):
        a = random.randint(0,N-2)
        b = random.randint(0,N-2)
        if polje[a][b] == 1:
            polje[a][b] = -1
        else:
            polje[a][b] = 1    
        deltaE =  -2*polje[a][b]*(polje[a-1][b] + polje[a+1][b] + polje[a][b+1] + polje[a][b-1]) + 2*H*polje[a][b] 
        if deltaE < 0:
            E = E + deltaE
            sprememba = sprememba + 1
        else:
            c = random.random()    
            if c <= math.exp(-deltaE/T[j]):
                sprememba = sprememba + 1
                E = E + deltaE
            else:
                if polje[a][b] == 1:
                    polje[a][b] = -1
                else:
                    polje[a][b] = 1
        energija.append(E)
    mag = 0
    for i in range(N):
        for j in range(N):
            mag = mag + polje[i][j]
    mag = mag/N/N
    magnetizacija.append(mag) 
    polje = polje_zac
    E = E1

plt.plot(T,magnetizacija,"-bo")    
plt.show()


st_korakov=np.linspace(0,M,M+1)
plt.plot(st_korakov,energija,".")     
plt.show()


print(f"M = {mag}")       

"""
print(sprememba)
plt.figure(figsize=(7,7))
for i in range(N-1):
    for j in range(N-1):
        if polje[i][j] == 1:
            plt.scatter(i,j,s=1,c="k",marker="s")

    

plt.show()

"""
"""
l=1
plt.figure(figsize=(7,7))

plt.scatter(1,1,s=l,c="k",marker="s")
plt.scatter(1,2,s=l,c="k",marker="s")
plt.scatter(1,3,s=l,c="k",marker="s")
plt.scatter(2,1,s=l,c="k",marker="s")
plt.scatter(2,2,s=l,c="k",marker="s")
plt.scatter(1,4,s=l,c="k",marker="s")
plt.scatter(1,5,s=l,c="k",marker="s")
plt.scatter(1,7,s=l,c="k",marker="s")
plt.scatter(1,9,s=l,c="k",marker="s")
plt.scatter(1,11,s=l,c="k",marker="s")

plt.scatter(200,200,s=l,c="k",marker="s")
plt.show()
"""