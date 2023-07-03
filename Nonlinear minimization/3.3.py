import math
import scipy.optimize
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def semafor(v):
    
    t0 = 10
    dt = t0/len(v)
    global t
    t = np.arange(0, 1, 1/len(v)) 
    l = 20
    zacetna = 3
    F = ((zacetna - v[0]) /dt)**2
    kappa1 = 2
    kappa2 = 2
    razdalja = 1/2 * zacetna * dt 
    for i in range(1,len(v)):
        F +=  ( (v[i] - v[i-1]) / dt )**2
        razdalja  +=  v[i] * dt
    razdalja += -1/2 * v[len(v)-1] * dt

    F1 = 1 + math.e **(kappa1* (razdalja-l))
    F2 = 1 + math.e **(kappa2* (l-razdalja))
    global hitrost
    hitrost = v/zacetna

    return F + F1 + F2     

v0 = []
for i in range(150):
    v0.append(random.uniform(1, 5.0))
#scipy.optimize.fmin_powell(semafor, v0)
scipy.optimize.fmin_powell(semafor, v0,maxiter=1000)



plt.title('Hitrost v odvisnosti od časa pri t0 = 10s, l = 20m in v0 = 3')
plt.xlabel('t/t0')
plt.ylabel('v/v0')
plt.plot(t, hitrost)
plt.show()