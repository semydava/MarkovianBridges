from math import sqrt
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import random
import matplotlib.pyplot as plt
steps = 100
rho = 0.5
T = 1
def random_walk(n):
    x, y = 0, 0
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(n + 1)
    positions = [y]
    for i in range(1, n + 1):
        # Randomly select either UP or DOWN based on probability
        prob = random.uniform(0, 1)
        # Move the object up or down
        if prob <= 0.5:
            y += 1
        elif prob > 0.5:
            y -= 1
        # Keep track of the positions
        positions.append(y)
    print(len(positions))
    return positions

def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        e = 1
    elif prob > 0.5:
        e = -1
    return e
#e = np.random.randn(steps)
#e = random_walk(steps - 1)
y1 = [0]
y2 = [2]
fig, ax = plt.subplots(2)
plt.xlim(0, T)
#plt.ylim(0, 10)

# Possible to get rid of this loop?
for i in range(steps - 1):
    e = discrete_noise()
    v1 = rho*y1[i] + e
    y1.append(v1)
    e = discrete_noise()
    v2 = rho * y2[i] + e
    y2.append(v2)

t1 = np.linspace(0, T, steps)
t2 = t1.tolist()
t2.reverse()
y1_d = {}
for i,t in zip(y1, t1):
    y1_d[t] = i
y2_d = {}
for i,t in zip(y2, t2):
    y2_d[t] = i
z = {}
diff = []
for i in t1:
    d = abs(y2_d[i] - y1_d[i])
    diff.append(d)
min_d = min(diff)
if min_d <= 0.01:
    print('yes')

min_d_index = diff.index(min_d)
print(min_d)
print(min_d_index)
print(t1[min_d_index])
for i in t1:
    if i <= t1[min_d_index]:
        z[i] = y1_d[i]
    else:
        z[i] = y2_d[i]
#print(i)
ax[0].scatter(y1_d.keys(), y1_d.values(), color = 'orange')
ax[0].plot(t1[min_d_index], min_d, 'ro')
ax[0].scatter(y2_d.keys(), y2_d.values(), color = 'green')
ax[1].plot(z.keys(), z.values(), color = 'blue')
plt.show()

