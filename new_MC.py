import quantecon as qe
import numpy as np
import random
import sympy
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from matplotlib.animation import FuncAnimation
from random import randint
from quantecon import MarkovChain
from math import sqrt
from scipy.stats import norm
def generate_transitions(n, T):
     states = [i for i in range(n)]
     dt = 0.1
     steps = int(T/dt)
     transitions = []
     for i in range(steps + 1):
          transitions.append(random.choice(states))
     return transitions


def transition_matrix(n, T):
     transitions = generate_transitions(n, T)
     states = [i for i in range(n)]
     df = pd.DataFrame(transitions)
     # create a new column with data shifted one space
     df['shift'] = df[0].shift(-1)
     # add a count column (for group by function)
     df['count'] = 1
     # groupby and then unstack, fill the zeros
     M = df.groupby([0, 'shift']).count().unstack().fillna(0)
     # normalise by occurences and save values to get transition matrix
     M = M.div(M.sum(axis=1), axis=0).values
     mc = qe.MarkovChain(M, states)
     if mc.is_irreducible == True and mc.is_aperiodic == True:
          return M
     # print M:
     #for row in M:
          #print(row)


#print(transition_matrix(5, 1))

P = [[0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.4, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1],
     [0.3, 0.2, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.075, 0.025],
     [0.4, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.04, 0.06],
     [0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05],
     [0.3, 0.2, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.2, 0.3, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05]
     ]


def MC_Sim(P, T, a, b):
     #P = transition_matrix(n, T)
     states = [i for i in range(11)]
     mc = qe.MarkovChain(P, states)
     #dt = 0.1
     #N = int(T/dt)
     #time = list(np.linspace(0, T, num=N))
     time = [i for i in range (0, T)]
     #reversed_time = list(np.linspace(0, T, num=N))
     #reversed_time.reverse()
     X_1 = {}
     X_2 = {}
     Z = {}
     X1 = mc.simulate(ts_length=len(time), init=a)
     X2 = mc.simulate(ts_length=len(time), init=b)
     X2 = list(X2)
     X2.reverse()
     inter = 0
     for i in range(T):
          if X1[i] == X2[i]:
               inter = 1
               break

     # plt.plot(time, X1)
     # plt.plot(time, X2)
     # plt.show()
     return inter

#print(MC_Sim(P, 1, 0, 3))
def Monte_Carlo_b1(sim_n, T):
     count = 0
     for i in range(sim_n):
          inter = MC_Sim(P, T, 0, 0)
          count += inter
     prob_inter = (count / sim_n)
     return prob_inter
#print(Monte_Carlo_b1(100, 1))

def Monte_Carlo_b2(sim_n, T):
     count = 0
     for i in range(sim_n):
          inter = MC_Sim(P, T, 0, 0)
          count += inter
     prob_inter = (count / sim_n)
     return prob_inter

def Monte_Carlo_b3(sim_n, T):
     count = 0
     for i in range(sim_n):
          inter = MC_Sim(P, T, 0, 5)
          count += inter
     prob_inter = (count / sim_n)
     return prob_inter

def prob_plot(T):
     Ts = [i for i in range(1,T)]
     prob_freq1 = []
     prob_freq2 = []
     prob_freq3 = []
     prob_time = []
     avrg = []
     iterator = 0
     for i in range(1, T):
          p1 = Monte_Carlo_b1(10, i)
          p2 = Monte_Carlo_b2(10, i)
          p3 = Monte_Carlo_b3(10, i)
          prob_freq1.append(p1)
          prob_freq2.append(p2)
          prob_freq3.append(p3)
          #if iterator < len(n) - 1:
               #avr = (p + Monte_Carlo(100, n[iterator + 1])) / 2
               #avrg.append(avr)
          #iterator += 1
     #for i in n:
          #p = Monte_Carlo_time_interval(100, i)
          #prob_time.append(p)

     plt.title("Probability of stopping time detection depending on time frequency" )
     plt.xlabel('Time')
     plt.ylabel('Probability')
     plt.plot(Ts, prob_freq1, color="orange", label="Probability of stopping time detection with b = 1")
     plt.plot(Ts, prob_freq2, color="lawngreen", label="Probability of stopping time detection with b = 5")
     plt.plot(Ts, prob_freq3, color="purple", label="Probability of stopping time detection with b = 10")
     # plt.plot(n1, y)

     # ARCTAN:
     #y = (np.arctan(np.sqrt(n1)) / 2) * 1.39
     #plt.plot(n1, y, lw=2, color = "red", label = "Arctan function")

     # SMOOTHED:
     #ysmoothed = gaussian_filter1d(avrg, sigma=2)
     #new_y =  np.insert(ysmoothed,0,0)
     #plt.plot(n, new_y, lw=2, color = "green", label = "Averaged probability")
     plt.legend()

     #ax[1].plot(n, prob_time)
     plt.show()
print(prob_plot(100))




