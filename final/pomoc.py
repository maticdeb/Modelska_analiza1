import numpy as np
from scipy.optimize import linprog, curve_fit
import matplotlib.pyplot as plt
import time
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import random
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize,listSolvers ,GUROBI,PULP_CBC_CMD
from scipy.stats.stats import pearsonr


def center(N):
    nodes = np.arange(N,N**2-N)
    center = []
    for i in range(len(nodes)):
        if not (nodes[i] % N == 0 or nodes[i] % N == N-1):
            center.append(nodes[i])    
    return center

def povezave(N,upload):
    #upload True pomeni smer od odjemalcev do strežnikov
    #False pa od strežnikov do odjemlcev

    X = np.zeros((N**2,N**2))

    #koti
    X[0][N] = 1
    X[N-1,2*N-1] = 1 
    X[N*N-N,N*N-2*N] = 1
    X[N*N-1, N*N-1-N] = 1

    #zgornji rob
    for i in range(1,N-1):
        X[i,i+N] = 1

    #spodnji rob
    for i in range(N*N-N+1,N*N-1):
        X[i,i-N] = 1

    #lev rob
    for i in range(N,N*N-N,N):
        X[i,i-N] = 1
        X[i,i+N] = 1
        X[i,i+1] = 1

    #desni rob
    for i in range(2*N-1, N*N-N,N):
        X[i,i-N] = 1
        X[i,i+N] = 1
        X[i,i-1] = 1            

    #center
    centralni = center(N)
    for i in range(len(centralni)):
        a = centralni[i]
        X[a, a-1] = 1 
        X[a, a+1] = 1
        X[a, a-N] = 1 
        X[a, a+N] = 1 

    
    if upload == True:
        for i in range(N):
            X[i+N][i] = 0
            X[N**2-N+i][N**2-2*N+i] = 0               
    else:
        for i in range(N):
            X[N**2-2*N+i][N**2-N+i] = 0    
            X[i][i+N] = 0   


    """
    plt.matshow(X)
    if upload:
        plt.title("upload")
    else:
        plt.title("download")  
    plt.colorbar()    
    plt.show()
    """
    return X

def read(povezava):
    _, i,j = povezava.split("_")
    i = int(i)
    j = int(j)
    return i,j


def vizualizacija(n,hitrosti,upload,hitrost):
    G = nx.DiGraph()
    nodes = np.arange(0, n**2).tolist()
    G.add_nodes_from(nodes)
    barva_nodes = []

    matrika = povezave(n,True)
    edges = []
    for i in range(n**2):
        if i < n:
            barva_nodes.append("blue")
        elif i >= n **2 - n :
            barva_nodes.append("red")    
        else:
            barva_nodes.append("green")    
   
        for j in range(n**2):
            if matrika[i][j] != 0:
                edges.append((i,j))



    G.add_edges_from(edges)

    x = 0 
    y = 100
    korak = 100 / (n-1)
    pozicije = {}
    for i in range(n**2):
        pozicije[i] = (x,y)
        x += korak
        if (i + 1) % n == 0:
            x = 0
            y -= korak

    #print(pozicije)

    nx.draw_networkx(G,pozicije,node_color = barva_nodes)
    
    #plt.arrow(40,40,0,20,head_width=1, head_length=1)
    
    for var in hitrosti():
        if var.value() != None  and var.value() != 0:
            print(f"{var.name}: {var.value()}")
            zacetni,konci = read(var.name)
            x1,y1 = pozicije[zacetni]
            x2,y2 = pozicije[konci]

            rotation = 0

            if x1 == x2:
                if y1 > y2:
                    rotation = 270
                else:
                    rotation = 90    
            else:
                if x1 > x2:
                    rotation = 180
                #else ne rabimo saj je rotacija v tem primeru že pravilna    


            plt.text((x1+x2)/2, (y1 + y2)/2, round(var.value(),3),
                    ha="center", va="center", rotation=rotation, size=50/n,
                    bbox=dict(boxstyle="rarrow,pad=0.3",
                            fc="white", ec="black", lw=1.5))

    
    if upload:
        plt.title(f"upload: {hitrost}")
    else:
        plt.title(f"download: {hitrost}")    
    plt.show()

    return 0

def fit_kvadrat(x,a,b,c):
    return a*x**2 + b*x + c

def fit_tri(x,a,b,c,d):    
    return a*x**3 + b*x**2 + c*x + d

def fit_stiri(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fit_exp(x,a,b,c):
    return a *np.exp(-b*x) + c


def problem(n, upload):
    #časovno optimiziran algoritem, cost funciton maksimizira skupno hitrost računalnikov in strežnikov
    mreza = povezave(n, upload)
    for i in range(len(mreza)):
        for j in range(len(mreza[0])):
            if mreza[i][j] != 0:
                mreza[i][j] *= random.random()

    #plt.matshow(mreza)
    #plt.colorbar()
    #plt.show() 
    model = LpProblem(name= "omrezje", sense=LpMaximize)

    x = {f"{i}_{j}": LpVariable(f"x_{i}_{j}", lowBound=0, upBound=mreza[i][j])for i in range(n**2) for j in range(n**2) if mreza[i][j] != 0} 
    #print(x)
    #omejitev: kirchofov zakon
    for i in range(n,n**2-n):
        model += (lpSum(x[f"{i}_{j}"]  for j in range(n**2) if mreza[i][j] != 0) 
        - lpSum(x[f"{j}_{i}"]  for j in range(n**2) if mreza[j][i] != 0)) == 0

    #cost function: upload + dowload uporabnikov in strežnikov        
    if upload == True:
        #upload
        model += (lpSum(x[f"{i}_{i+n}"]for i in range(n) )    #upload uporabnikov
                + lpSum(x[f"{i-n}_{i}"]for i in range(n**2-n,n**2) )) # dowload streznikov
    else:
        #download
        model +=  (lpSum(x[f"{i+n}_{i}"]for i in range(n) )  #download uporabnikov
        + lpSum(x[f"{i}_{i-n}"]for i in range(n**2-n,n**2) )) #upload streznikov     

    #print(model)

    status = model.solve(PULP_CBC_CMD(msg=0))
    """
    print(f"status: {model.status}, {LpStatus[model.status]}")
    for var in x.values():
        if var.value() != None  and var.value() != 0:
            print(f"{var.name}: {var.value()}")
    if upload == True:
        print("upload")
    else:
        print("download") 
    """    
    return status, model.objective.value()/2/n, x.values , mreza


def problem2(n, upload):
    #časovno optimiziran algoritem, cost funciton maksimizirna minimalno hitrost
    mreza = povezave(n, upload)
    for i in range(len(mreza)):
        for j in range(len(mreza[0])):
            if mreza[i][j] != 0:
                mreza[i][j] *= random.random()

    #plt.matshow(mreza)
    #plt.colorbar()
    #plt.show() 
    model = LpProblem(name= "omrezje", sense=LpMaximize)

    x = {f"{i}_{j}": LpVariable(f"x_{i}_{j}", lowBound=0, upBound=mreza[i][j])for i in range(n**2) for j in range(n**2) if mreza[i][j] != 0} 
    #print(x)
    #omejitev: kirchofov zakon
    for i in range(n,n**2-n):
        model += (lpSum(x[f"{i}_{j}"]  for j in range(n**2) if mreza[i][j] != 0) 
        - lpSum(x[f"{j}_{i}"]  for j in range(n**2) if mreza[j][i] != 0)) == 0


    for i in range(n):
            model += lpSum(x[f"{i}_{j}"] for j in range(n**2) if mreza[i][j] != 0) >= min_speed


    #cost function: upload + dowload uporabnikov in strežnikov        
    if upload == True:
        #upload
        model += (min(x[f"{i}_{i+n}"]for i in range(n) ))   #upload uporabnikov
                #+ min(x[f"{i-n}_{i}"]for i in range(n**2-n,n**2) )) # dowload streznikov
    else:
        #download
        model +=  (min(x[f"{i+n}_{i}"]for i in range(n) ))  #download uporabnikov
        #+ lpSum(x[f"{i}_{i-n}"]for i in range(n**2-n,n**2) )) #upload streznikov     

    #print(model)

    status = model.solve(PULP_CBC_CMD)
    """
    print(f"status: {model.status}, {LpStatus[model.status]}")
    for var in x.values():
        if var.value() != None  and var.value() != 0:
            print(f"{var.name}: {var.value()}")
    if upload == True:
        print("upload")
    else:
        print("download") 
    """    
    return status, model.objective.value(), x.values , mreza

def problem_gpt(n, upload):
    #časovno optimiziran algoritem, cost funciton minimizira najpočasnejši računalnik/odjemalca
    mreza = povezave(n, upload)
    for i in range(len(mreza)):
        for j in range(len(mreza[0])):
            if mreza[i][j] != 0:
                mreza[i][j] *= random.random()

    model = LpProblem(name= "omrezje", sense=LpMaximize)

    x = {f"{i}_{j}": LpVariable(f"x_{i}_{j}", lowBound=0, upBound=mreza[i][j])for i in range(n**2) for j in range(n**2) if mreza[i][j] != 0}

    #omejitev: kirchofov zakon
    for i in range(n,n**2-n):
        model += (lpSum(x[f"{i}_{j}"]  for j in range(n**2) if mreza[i][j] != 0) 
        - lpSum(x[f"{j}_{i}"]  for j in range(n**2) if mreza[j][i] != 0)) == 0

    # definirajte novo spremenljivko za najpočasnejšega odjemalca
    slowest_client = LpVariable("slowest_client", lowBound=0)

    # dodajte pogoj za maksimiziranje hitrosti najpočasnejšega odjemalca
    model += slowest_client <= lpSum(x[f"{i}_{j}"] for i in range(n) for j in range(n**2-n,n**2) if mreza[i][j] != 0 )

    # spremenite ciljno funkcijo, da maksimizirate hitrost najpočasnejšega odjemalca
    model += slowest_client


    status = model.solve(PULP_CBC_CMD(msg=0))

    return status, model.objective.value(), x.values, mreza



def problem1(n, upload):
    # prvi algoritem, ki še  ni časovno optimiziran, cost funciton maksimizira skupno hitrost računalnikov in strežnikov
    mreza = povezave(n, upload)
    for i in range(len(mreza)):
        for j in range(len(mreza[0])):
            if mreza[i][j] != 0:
                mreza[i][j] *= random.random()

    #plt.matshow(mreza)
    #plt.colorbar()
    #plt.show() 
    model = LpProblem(name= "omrezje", sense=LpMaximize)

    x = {f"{i}_{j}": LpVariable(f"x_{i}_{j}", lowBound=0, upBound=mreza[i][j])for i in range(n**2) for j in range(n**2)} 
    #print(x)
    """
    #omejitev: maksimalna hitrost v povezavi x_ij <= 1
    for i in range(n**2):
        for j in range(n**2):
            #if mreza[i][j] != 0:
            #tuki bo mogoče treba dat hitrost za ii na 0 (trenutno je na 1)
            model += (x[f"{i}_{j}"] <= mreza[i][j], f"x_{i}_{j}")
    """
    #omejitev: kirchofov zakon
    for i in range(n,n**2-n):
        model += (lpSum(x[f"{i}_{j}"]  for j in range(n**2) ) 
        - lpSum(x[f"{j}_{i}"]  for j in range(n**2) )) == 0

    #cost function: upload + dowload uporabnikov in strežnikov        
    if upload == True:
        #upload
        model += (lpSum(x[f"{i}_{i+n}"]for i in range(n))    #upload uporabnikov
                + lpSum(x[f"{i-n}_{i}"]for i in range(n**2-n,n**2))) # dowload streznikov
    else:
        #download
        model +=  (lpSum(x[f"{i+n}_{i}"]for i in range(n))  #download uporabnikov
        + lpSum(x[f"{i}_{i-n}"]for i in range(n**2-n,n**2))) #upload streznikov     

    #print(model)

    status = model.solve(PULP_CBC_CMD(msg=0))

    """
    print(f"status: {model.status}, {LpStatus[model.status]}")
    for var in x.values():
        if var.value() != None  and var.value() != 0:
            print(f"{var.name}: {var.value()}")
    
    if upload == True:
        print("upload")
    else:
        print("download") 
    """                
    return status, model.objective.value()/2/n, x.values

#tukaj so shranjeni povprečne maksimalne hitrosti v odvisnosti od n
#meritve = np.load("D:/modelska_analiza_1/zaklucna/shrani.npy")



ponivitve = 1000
n= np.array([4])


upload = False
maximalne_hitrost= [0] * ponivitve
solverji = [ GUROBI,PULP_CBC_CMD]

status, hitrost,hitrosti,mreza = problem_gpt(n[0],upload)

print(hitrost)
for var in hitrosti():
    print(f"{var.name}: {var.value()}")
"""
pon = [[0]*ponivitve for i in range(4)]
rezultati = [0] * ponivitve
for k in range(ponivitve):
    vodoravne = 0
    navpicne = 0
    vodoravne_res = 0
    navpicne_res = 0
    st1, st2 = 0,0
    status, hitrost,hitrosti,mreza = problem(n[0],upload)

    rezultati[k] = hitrost
    if status != 1:
        print("napaka")

    for i in range(len(mreza)):
        for j in range(len(mreza[0])):
            if abs(i-j) == 1:
                vodoravne += mreza[i][j]
            elif abs(i-j) == n[0]:
                navpicne += mreza[i][j]

          

    for var in hitrosti():
        #print(f"{var.name}: {var.value()}")

        od,do = read(var.name)
        #print(od,do, var.value())
        if abs(od-do) == 1:
            vodoravne_res += var.value()
            st1 += 1
        else:
            navpicne_res += var.value()    
            st2 += 1

    pon[0][k] = navpicne / st2
    pon[1][k] = vodoravne  / st1
    pon[2][k] = navpicne_res / st2
    pon[3][k] = vodoravne_res / st2


print(pearsonr(pon[0],rezultati))
print(pearsonr(pon[1],rezultati))
print(pearsonr(pon[2],rezultati))
print(pearsonr(pon[3],rezultati))

plt.plot(pon[0],rezultati,"o")
plt.ylabel("hitrost interneta")
plt.xlabel("povprečje dovoljenih navpičnih povezav")
plt.text(0.35,0.5,f"r={round(pearsonr(pon[0], rezultati)[0],4 )}")
plt.text(0.35,0.45,f"p={round(pearsonr(pon[0], rezultati)[1],4 )}")
plt.show()

plt.plot(pon[1],rezultati,"o")
plt.ylabel("hitrost interneta")
plt.xlabel("povprečje dovoljenih vodoravnih povezav")
plt.text(0.3,0.55,f"r={round(pearsonr(pon[1], rezultati)[0],4 )}")
plt.text(0.3,0.50,f"p={round(pearsonr(pon[1], rezultati)[1],4 )}")
plt.show()

plt.plot(pon[2],rezultati,"o")
plt.ylabel("hitrost interneta")
plt.xlabel("povprečje uporabljenih navpičnih povezav")
plt.text(0.35,0.5,f"r={round(pearsonr(pon[2], rezultati)[0],4 )}")
plt.text(0.35,0.45,f"p={round(pearsonr(pon[2], rezultati)[1],4 )}")
plt.show()

plt.plot(pon[3],rezultati,"o")
plt.ylabel("hitrost interneta")
plt.xlabel("povprečje uporabljenih vodoravnih povezav")
plt.text(0.0,0.6,f"r={round(pearsonr(pon[3], rezultati)[0],4 )}")
plt.text(0.0,0.55,f"p={round(pearsonr(pon[3], rezultati)[1],4 )}")
plt.show()
"""


"""
for j in range(len(n)):

    racunalniki = np.zeros(n[j])

    for i in range(ponivitve):
        status, hitrost,hitrosti = problem(n[j],upload)
        if status != 1:
            print("napaka")

        for var in hitrosti():
            #print(f"{var.name}: {var.value()}")

            _,odjmalec = read(var.name)
            if odjmalec < n[j]:
                racunalniki[odjmalec] += var.value()

    racunalniki /= ponivitve
    print(racunalniki)

    x = np.linspace(0,1,n[j],True)

    plt.plot(x,racunalniki,"o--",label=f"{n[j]}")
plt.ylabel("povprečna maskimalna hitrost posameznega odjemalca")
plt.legend()
plt.show()

"""

"""
koncna = []


for j in range(len(n)):
    print(n[j])
    maximalne_hitrost= np.zeros(ponivitve)
    avg_hitrost = np.zeros(ponivitve)
    for i in range(ponivitve):
        #status pove ali je program najdel optimalno rešitev, 
        #hitrost: optimalna rešitev
        #hitorsti: povezave med posameznimi vozlišči
        status, hitrost,hitrosti = problem(n[j],upload)
        #print(f"{i}:{status}, hitrost: {hitrost}")

        maximalne_hitrost[i] = hitrost
        avg_hitrost[i] = (avg_hitrost[i-1] * i + maximalne_hitrost[i]) / (i+1)
        
        if i > 500 and  i % 200 == 0:
            povprecje = np.average(avg_hitrost[i-100:i])
            a = abs(avg_hitrost[i-100:i] - povprecje)
            #print(a)
            if max(a) < 10**(-4):
                print(i)
                break
    avg_hitrost = avg_hitrost[avg_hitrost!=0]     
    #avg_hitrost = np.ma.masked_equal(avg_hitrost,0)

    koncna.append(avg_hitrost[-1])
    plt.plot(avg_hitrost,label=f"n={n[j]}")
"""









"""
originalni = [0] * len(n) 
for j in range(len(n)):
    zacetek = time.time()

    for i in range(ponivitve):
        #status pove ali je program najdel optimalno rešitev, 
        #hitrost: optimalna rešitev
        #hitorsti: povezave med posameznimi vozlišči
        status, hitrost,hitrosti = problem1(n[j],upload)
        #print(f"{i}:{status}, hitrost: {hitrost}")

        maximalne_hitrost[i] = hitrost

    print(sum(maximalne_hitrost)/ponivitve)

    originalni[j] = time.time()-zacetek



popt, _ = curve_fit(fit_stiri, n, originalni)
d,e,f,g,h = popt
print(d,e,f,g,h)
fit4 =fit_stiri(n,d,e,f,g,h)
"""




"""
popt, napaka1 = curve_fit(fit_kvadrat, n, optimalni)
a,b,c = popt
print(a,b,c)
fit4 =fit_kvadrat(n,a,b,c)

popt,napak2 = curve_fit(fit_tri, n, optimalni)
i,j,k,l= popt
print(i,j,k,l)
fit3 =fit_tri(n,i,j,k,l)



plt.plot(n,optimalni,"bo",label="rezultati")
plt.plot(n,fit3,"r",label="tretja potenca")
plt.plot(n,fit4,"g",label="druga potenca")


plt.legend()
plt.xlabel("n")
plt.ylabel("t[s]")
plt.title("Čas potreben za 10 iteraciji problema")
plt.show()
"""




"""
for var in hitrosti():
    if var.value() != None  and var.value() != 0:
        print(f"{var.name}: {var.value()}")
print(f"reseno :{status}")
print(f"optimalan hitrost :{hitrost}")


vizualizacija(n,hitrosti,upload,hitrost)
"""






"""
def povezave1(n,upload):
    #poskus optimizacije, ki pa ni bil uspešen
    t1 = time.time()
    gor_dol = np.ones(n**2-n)
    desno = np.ones(n**2-1)
    for i in range(len(desno)):
        if i < n or i >= n**2 - n or i % n == n-1:
            desno[i] = 0
    X = np.diag(gor_dol,n)
    X += np.diag(gor_dol,-n)
    X += np.diag(desno,-1)
    X += np.diag(desno,1)
    if upload == True:
        for i in range(n):
            X[i+n][i] = 0
            X[n**2-n+i][n**2-2*n+i] = 0               
    else:
        for i in range(n):
            X[n**2-2*n+i][n**2-n+i] = 0    
            X[i][i+n] = 0    
    print(time.time()-t1)
    return X
"""


