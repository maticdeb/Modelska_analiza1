import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

#VEC STADIJEV BOLEZNI + MRTVI 
def model(zacetni,t,eta0):
    D,B1,B2,B3, B4,I,M = zacetni
    alpha = 0.3*10**(-6) #monžnosti za prenos okužbe
    beta = 0.3  #koliko časa potrebuje bolezen da se razvije
    eta = beta*eta0 #koliko procentov bolnih ljudi preventivno pošljemo v karanteno
    gamma = 0.15    #koliko časa traja kužnost
    delta = 0.1     #po koliko časa ljude ozdravijo 
    kappa = delta*0.01  #koliko procentov ljudi umre
    if B4 > 10000:
        kappa = delta*(0.03)
    dDdt = -alpha*D*B2 
    dB1dt = alpha*D*B2 - (beta-eta)*B1 - eta*B1
    dB2dt = (beta-eta)*B1 - gamma*B2
    dB3dt = eta*B1 - gamma*B3
    dB4dt = gamma*B2 + gamma*B3 - delta*B4 - kappa*B4
    dIdt = delta*B4
    dMdt = kappa*B4
    d = [dDdt, dB1dt, dB2dt, dB3dt, dB4dt, dIdt, dMdt]
    return d

zacetni = [2000000, 10, 10, 0, 0,0 ,0]  
n = 1000
t = np.linspace(0,600,n)

# store solution
D = np.empty_like(t)
B1 = np.empty_like(t)
B2 = np.empty_like(t)
B3 = np.empty_like(t)
B4 = np.empty_like(t)
I = np.empty_like(t)
M = np.empty_like(t)


# record initial conditions
D[0] = zacetni[0]
B1[0] = zacetni[1]
B2[0] = zacetni[2]
B3[0] = zacetni[3]
B4[0] = zacetni[4]
I[0] = zacetni[5]
M[0] = zacetni[6]

koeficienti = ( 0, 0.2, 0.5, 0.6, 0.7)
# solve ODE
for j in range(len(koeficienti)):
    for i in range(1,n):
    # span for next time step
        tspan = [t[i-1],t[i]]
        # solve for next step
        z = integrate.odeint(model,zacetni,tspan,args=(koeficienti[j], ))
        # store solution for plotting
        D[i] = z[1][0]
        B1[i] = z[1][1]
        B2[i] = z[1][2]
        B3[i] = z[1][3]
        B4[i] = z[1][4]
        I[i] = z[1][5]
        M[i] = z[1][6]

        # next initial condition
        zacetni = z[1]
        if i == n-1:
            plt.plot(t,M,f"C{j}",label=f"eta = {koeficienti[j]}")
            D = np.empty_like(t)
            B1 = np.empty_like(t)
            B2 = np.empty_like(t)
            B3 = np.empty_like(t)
            B4 = np.empty_like(t)
            I = np.empty_like(t)
            M = np.empty_like(t)

            zacetni = [2000000, 10, 10, 0, 0,0 ,0]  

            D[0] = zacetni[0]
            B1[0] = zacetni[1]
            B2[0] = zacetni[2]
            B3[0] = zacetni[3]
            B4[0] = zacetni[4]
            I[0] = zacetni[5]
            M[0] = zacetni[6]

"""
plt.plot(t,D,"C1",label="zdravi")
plt.plot(t,B1,"C2",label="okuženi")
plt.plot(t,B2,"C3",label="kužni")
plt.plot(t,B3,"C4",label=" predčasna izolacija")
plt.plot(t,B4,"C5",label="zaden stadij")
plt.plot(t,M,"C6",label="mrtvi")
"""
plt.ylabel('število umrilh')
plt.xlabel('dni')
plt.legend(loc='best')
plt.show()
