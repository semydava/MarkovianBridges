import numpy as np
import matplotlib.pyplot as plt

rate = 20.  # average number of events per second

dt = .001  # time step
n = int(1./dt)  # number of time steps
x = np.zeros(n)
x[np.random.rand(n) <= rate*dt] = 1
x1 = np.cumsum(x)
x2 = np.cumsum(x)
x2 = list(x2)
x2.reverse()
plt.plot(np.linspace(0., 1., n), x1)
plt.plot(np.linspace(0., 1., n), x2)
plt.xlabel("Time")
plt.ylabel("Counting process")
plt.show()