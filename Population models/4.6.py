import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

#ZAJCI IN LISICE 
def model(zacetni,t):
    Z, L = zacetni
    alpha = 2/3/1000
    beta = 4/3/1000000
    gamma = 1/1000
    delta = 1/1000000
    dZdt = alpha*Z -beta*Z*L
    dLdt = - gamma*L + delta*Z*L
    d = [dZdt, dLdt]
    return d

zacetni = [1000,1000]  
n = 10000
t = np.linspace(0,10000,n)



# store solution
Z = np.empty_like(t)
L = np.empty_like(t)



# record initial conditions
Z[0] = zacetni[0]
L[0] = zacetni[1]


# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = integrate.odeint(model,zacetni,tspan)
    # store solution for plotting
    Z[i] = z[1][0]
    L[i] = z[1][1]

    # next initial condition
    zacetni = z[1]

plt.plot(t,Z,"g",label="zajci")
plt.plot(t,L,"b",label="lisice")
plt.ylabel('N')
plt.xlabel('dni')
#plt.ylim([0,20])
plt.legend(loc='best')
plt.show()

plt.plot(Z,L)
plt.ylabel('lisice')
plt.xlabel('zajci')
plt.show()
