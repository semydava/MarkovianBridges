from math import sqrt
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import random
import matplotlib.pyplot as plt
def random_walk(n):
    x, y = 0, 0
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(n + 1)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, n + 1):
        # Randomly select either UP or DOWN
        prob = random.uniform(0, 1)

        # Move the object up or down
        if prob <= 0.5:
            y += 1
        elif prob > 0.5:
            y -= 1
        # Keep track of the positions
        positions.append(y)
    return timepoints, positions

def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y

mu = 10.  # Mean.
#tau = .05  # Time constant.
phi = 0.5
c = 2
#dt = .01  # Time step.
T = 1  # Total time.
#n = int(T / dt)  # Number of time steps.
n = 100
print(n)
t1 = np.linspace(0, T, n)  # Vector of times.
t2 = t1.tolist()
t2.reverse()
#sqrtdt = np.sqrt(dt)
x1 = np.zeros(n)
x2 = np.zeros(n)
z = np.zeros(n)
x1[0] = 4
x2[0] = 7

for i in range(n - 1):
    n1 = discrete_noise()
    x1[i + 1] = int(c + x1[i] * phi + n1)
    n2 = discrete_noise()
    x2[i + 1] = int(c + x2[i] * phi + n2)
diff = []
for i in range(n - 1):
    d = abs(x2[n - 1 - i] - x1[i])
    diff.append(d)
    if x1[i] == x2[n - 1 - i]:
        stop_index = i
        break
print(t1[stop_index])
t1_half = []
t2_half = []
z1 = {}
z2 = {}

for i in range(n):
    if i <= stop_index:
        z[i] = x1[i]
    else:
        z[i] = x2[n - i]

for idx,t in enumerate(t1):
    if idx <= stop_index:
        t1_half.append(t)
        z1[t] = z[idx]
    else:
        t2_half.append(t)
        z2[t] = z[idx]


fig, ax = plt.subplots(2)
plt.xlim(0, 10)
plt.ylim(0, T)
plt.xlabel('Time steps')
plt.ylabel('States')
ax[0].plot(t1, x1, lw=2, color = 'red', label="process X^1")
ax[0].plot(t2, x2, lw=2, color = 'green', label="time-reversed process X^2")
#ax[1].plot(z1.keys(), z1.values(), lw=2,  color = 'red', label="Z-bridge approx.")
#ax[1].plot(z2.keys(), z2.values(), lw=2, color = 'green', label="Z-bridge approx.")
plt.legend()
plt.show()



