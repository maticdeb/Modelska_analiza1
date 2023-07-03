import math
import scipy.optimize
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

print("koliko kroglic")
st = int(input())

def energija(n):
    
    A1 = 0
    B1 = 0
    r = 1
    global x
    x = []
    global y
    y = []
    global z
    z = []

    x.append(r*math.sin(A1)*math.cos(B1))
    y.append(r*math.sin(A1)*math.sin(B1))
    z.append(r*math.cos(A1))

    for i in range(0, len(n), 2):
        x.append(r*math.sin(n[i])*math.cos(n[i+1]))
        y.append(r*math.sin(n[i])*math.sin(n[i+1]))
        z.append(r*math.cos(n[i]))
    E = 0 
    for i in range(len(n)//2 + 1):
        for j in range(i):
            E += 1 / math.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2 + (z[j] - z[i])**2)


    return E  

x0 = []
for i in range(2*(st - 1)):
    x0.append(random.uniform(0.1, 5.0))

#scipy.optimize.fmin_powell(energija, x0)
scipy.optimize.minimize(energija,x0, method="Nelder-Mead")

"""
print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
"""
seznam = [[] for i in range(st)]

for i in range(st):
    seznam[i].append(x[i])
    seznam[i].append(y[i])
    seznam[i].append(z[i])

vsotax = 0
vsotay = 0
vsotaz = 0

for i in range(st):
    vsotax += seznam[i][0]
    vsotay += seznam[i][1]
    vsotaz += seznam[i][2]

print(seznam)
print(f"vsota x = {vsotax}")
print(f"vsota y = {vsotay}")
print(f"vsota z = {vsotaz}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
a = 1 * np.outer(np.cos(u), np.sin(v))
b = 1 * np.outer(np.sin(u), np.sin(v))
c = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(a, b, c, color='b', alpha = 0.1)

ax.scatter(x,y,z, s = 50, c = "red")

plt.show()