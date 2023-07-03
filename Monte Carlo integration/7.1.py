import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random

#KOSTANTNA GOSTOTA
"""
stevec = 0
n = [10,1000,100000,10000000]
for j in range(len(n)):
    for i in range(n[j]):
        x=random.uniform(-1,1)
        y=random.uniform(-1,1)
        z=random.uniform(-1,1)
        a = math.sqrt(abs(x)) + math.sqrt(abs(y)) + math.sqrt(abs(z))
        if a <= 1:
            stevec = stevec + 1
    print(f"n = {n[j]}")
    print(f"masa = {stevec/n[j]}")

"""    

n =  1000000
razmerje = []
masa = []
vztrajnostni_moment = []
stevec  = 0
masa1 = 0
vztrajnostni_moment1 = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1,n):
    x=random.uniform(-1,1)
    y=random.uniform(-1,1)
    z=random.uniform(-1,1)
    r = math.sqrt(x**2 + y**2 + z**2)
    a = math.sqrt(abs(x)) + math.sqrt(abs(y)) + math.sqrt(abs(z))
    if a <= 1:
        stevec = stevec + 1
        masa1 = masa1 + 8
        vztrajnostni_moment1 = vztrajnostni_moment1 + 8*r**2
        #ax.scatter(x,y,z,c="red")
    masa.append(masa1/i)
    razmerje.append(stevec/i)
    vztrajnostni_moment.append(vztrajnostni_moment1/i)    
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")        
plt.show()

#RAZMERJE 
st_tock = np.linspace(1,n,n-1)
plt.plot(st_tock,razmerje)
plt.xlabel("število točk")
plt.ylabel("razmerje")
plt.xscale("log")
plt.show()

print(f"razmerje = {razmerje[n-2]}")
"""
#MASA
st_tock = np.linspace(1,n,n-1)
plt.plot(st_tock,masa)
plt.xlabel("število točk")
plt.ylabel("masa")
plt.xscale("log")
plt.show()

print(f"masa = {8*razmerje[n-2]}")
"""

#VZTRAJNOSTNI MOMENT
st_tock = np.linspace(1,n,n-1)
plt.plot(st_tock,vztrajnostni_moment)
plt.xlabel("število točk")
plt.ylabel("vztrajnstni moment")
plt.xscale("log")
plt.show()

print(f"masa = {vztrajnostni_moment[n-2]}")
