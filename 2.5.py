from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#naloga 1 (minimizacija cena)
tabela1 = pd.read_csv('tabela.txt', sep='\t')
energija1 = tabela1["energija"]
mascobe1 = tabela1["mascobe"]
OH1 = tabela1["ogljikovi hidrati"]
proteini1 = tabela1["proteini"]
kalcij1 = tabela1["Ca"]
zelezo1 = tabela1["Fe"]
vitamin1 = tabela1["Vitamin C"]
kalij1 = tabela1["Kalij"]
natrij1 = tabela1["Natrij"]
cena1 = tabela1["Cena"]
zivilo1 = tabela1["zivilo"]




en= []
ma = []
OH = []
pr = []
Ca = []
Fe = []
C = []
K = []
Na = []
ce = []
teza = []
ziv = []
for i in range(49):
    teza.append(1)

for i in range(49):
    en.append(energija1[i])
    ma.append(mascobe1[i])
    OH.append(OH1[i])
    pr.append(proteini1[i])
    Ca.append(kalcij1[i])
    Fe.append(zelezo1[i])
    C.append(vitamin1[i])
    K.append(kalij1[i])
    Na.append(natrij1[i])
    ce.append(cena1[i])
    ziv.append(f"{zivilo1[i]}")


model = LpProblem(name= "hrana_min_cena", sense=LpMinimize)

x = {i: LpVariable(name = f"{ziv[i]}", lowBound=0)for i in range(49)}

omejitve_energija = 0
omejitve_ma =  0
omejitve_OH = 0
omejitve_pr = 0
omejitve_Ca = 0
omejitve_Fe = 0
omejitve_C = 0
omejitve_K = 0
omejitve_Na = 0
omejitve_ce = 0
omejitve_teza = 0


for i in range(49):
    omejitve_energija += x[i] * en[i]
    omejitve_ma += x[i] * ma[i]
    omejitve_OH += x[i] * OH[i] 
    omejitve_pr += x[i] * pr[i] 
    omejitve_Ca += x[i] * Ca[i] 
    omejitve_Fe += x[i] * Fe[i] 
    omejitve_C += x[i] * C[i] 
    omejitve_K += x[i] * K[i] 
    omejitve_Na += x[i] * Na[i] 
    omejitve_ce += x[i] * ce[i] 
    omejitve_teza += x[i] * teza[i] 


#omejtive
model += (omejitve_ma >= 70, "ma")
model += (omejitve_OH >= 310, "OH")
model += (omejitve_pr >= 50, "pr")
model += (omejitve_Ca >= 1000, "Ca")
model += (omejitve_Fe >= 18, "Fe")
model += (omejitve_C >= 60, "C")
model += (omejitve_K >= 3500, "K")
model += (omejitve_Na >= 500, "Nadol")
model += (omejitve_Na <= 2400, "Nagor")
model += (omejitve_teza <= 20, "teza")

#cost funciton
model += omejitve_ce

status = model.solve()

final_en = 0
final_ma = 0
final_OH = 0
final_pr = 0
final_Ca = 0
final_Fe = 0
final_C = 0
final_K = 0
final_Na = 0
final_teza = 0

hrana = []
hrana_en = []
hrana_OH = []
hrana_pr = []
hrana_ma = []
hrana_Ca = []
hrana_Fe = []
hrana_C = []
hrana_K = []
hrana_Na = []
hrana_teza = []

print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"CENA: {model.objective.value()}")


print("\n")

print("\\hline")
print("sestavina & masa[g] \\\\")
print("\\hline")

for index, kolicina in x.items():
    if kolicina.value() > 0:
        final_ma += kolicina.value()*ma[index]
        final_OH += kolicina.value()*OH[index]
        final_pr += kolicina.value()*pr[index]
        final_Ca += kolicina.value()*Ca[index]
        final_Fe += kolicina.value()*Fe[index]
        final_C += kolicina.value()*C[index]
        final_K += kolicina.value()*K[index]
        final_Na += kolicina.value()*Na[index]
        final_teza += kolicina.value()*teza[index]*100
        final_en += kolicina.value()*en[index]

        hrana.append(kolicina)
        hrana_en.append(kolicina.value()*en[index]/2000)
        hrana_OH.append(kolicina.value()*OH[index]/310)
        hrana_ma.append(kolicina.value()*ma[index]/70)
        hrana_pr.append(kolicina.value()*pr[index]/50)
        hrana_Ca.append(kolicina.value()*Ca[index]/1000)
        hrana_Fe.append(kolicina.value()*Fe[index]/18)
        hrana_C.append(kolicina.value()*C[index]/60)
        hrana_K.append(kolicina.value()*K[index]/3500)
        hrana_Na.append(kolicina.value()*Na[index]/1400)
        hrana_teza.append(kolicina.value()*teza[index]/2000)

        print(f"{kolicina.name} & {round(kolicina.value()*100)} \\\\")
        print("\\hline")


print("\n")


print("\\hline")
print(f"kalorije &  {round(final_en)} g \\\\")
print("\\hline")
print(f"ogljikovi hidrati &  {round(final_OH)} g \\\\")
print("\\hline")
print(f"proteini &  {round(final_pr)} g \\\\ ")
print("\\hline")
print(f"kalcij &  {round(final_Ca)} mg \\\\")
print("\\hline")
print(f"železo &  {round(final_Fe)} mg \\\\")
print("\\hline")
print(f"vitamin C &  {round(final_C)} mg \\\\")
print("\\hline")
print(f"kalij &  {round(final_K)} mg \\\\")
print("\\hline")
print(f"natrij & {round(final_Na)} mg \\\\")
print("\\hline")
print(f"masa &  {round(final_teza)} g \\\\")
print("\\hline")


#grafi z nekaj podatki

x = np.arange(len(hrana))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
barEN = ax.bar(x - 3*width/2, hrana_en, width, label='Kalorije')
barMA = ax.bar(x -width/2, hrana_ma, width, label= 'maščobe')
barOH = ax.bar(x + width/2, hrana_OH, width, label='ogljikovi hidrati')
barPR = ax.bar(x + 3*width/2, hrana_pr, width, label=' proteini')


ax.set_ylabel('delež pripročenega dnevnega vnosa hranil')
ax.set_title('Minimizacija cene')
ax.set_xticks(x)
ax.set_xticklabels(hrana)
ax.legend()



fig.tight_layout()

plt.show()

"""

#grafi z vsemi podatki
x = np.arange(len(hrana))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
barEN = ax.bar(x - 4*width, hrana_en, width, label='kalorije')
barOH = ax.bar(x - 3*width, hrana_OH, width, label='ogljikovi hidrati')
barPR = ax.bar(x - 2*width, hrana_pr, width, label=' proteini')
barMA = ax.bar(x - width, hrana_ma, width, label='maščobe')
barCA = ax.bar(x , hrana_Ca, width, label='kalcij')
barFE = ax.bar(x + width, hrana_Fe, width, label='železo')
barC = ax.bar(x + 2*width, hrana_C, width, label='vitamin C')
barK = ax.bar(x + 3*width, hrana_K, width, label='kalij')
barNa = ax.bar(x + 4*width, hrana_Na, width, label='natrij')


ax.set_ylabel('delež pripročenega dnevnega vnosa hranil')
ax.set_title('Minimizacija cene')
ax.set_xticks(x)
ax.set_xticklabels(hrana)
ax.legend()



fig.tight_layout()

plt.show()
"""

"""
for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")
"""

