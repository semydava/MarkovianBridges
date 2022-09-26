import random
import numpy as np
import matplotlib.pyplot as plt
def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y

def AR(T, a, b):
    #steps = 10000
    rho = 1
    x1, y1 = 0, a
    positions1 = [y1]
    x2, y2 = 0, b
    positions2 = [y2]
    dt = 0.0001
    n = int(T/dt)
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = [i for i in range(n + 1)]
    timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints = [i for i in range(n + 1)]
    reversed_timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints.reverse()
    reversed_timepoints_steps.reverse()
    Y_1 = {}
    Y_2 = {}
    intersect = False
    for i in range(0, n):
        n1 = discrete_noise()
        #n2 = discrete_noise()
        el_1 = (-0.5)*positions1[i] + dt * (-0.5)* (-positions1[i]) + n1*np.sqrt(dt)
        el_2 = (-0.5)*positions2[i] + dt * (-0.5)* (-positions2[i]) + n1*np.sqrt(dt)
        positions1.append(el_1)
        positions2.append(el_2)
        #positions2.append(positions2[i] + dt * (-positions2[i]) + n2)
    for i, j in zip(timepoints, reversed_timepoints):
        Y_1[i] = positions1[i]
        Y_2[j] = positions2[i]

    positions2 = list(Y_2.values())
    positions2.reverse()
    for e1, e2 in zip(positions1, positions2):
        index1 = positions1.index(e1)
        #print(len(x1))
        if index1 < len(positions1) - 1:
            new_index1 = index1 + 1
            n_el1 = positions1[new_index1]
            n_el2 = positions2[new_index1]
            if (e1 >= e2 and n_el1 <= n_el2) or (e1 <= e2 and n_el1 >= n_el2):
                stop_index = index1
                intersect = True
                break
    #for i in timepoints:
        #print(Y_1[i], Y_2[i])
        #if Y_1[i] == Y_2[i]:
            #intersect = True

    #return intersect
    #plt.plot(timepoints, Y_1.values())
    #plt.plot(reversed_timepoints, Y_2.values())
    #plt.show()

    return intersect

#print(AR(1, 10, 0, 4))


def random_walk(T, a, b):
    x1, y1 = 0, a
    x2, y2 = 0, b
    dt = 0.001
    n = int(T/dt)
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = [i for i in range(n+1)]
    timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints = [i for i in range(n + 1)]
    reversed_timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints.reverse()
    reversed_timepoints_steps.reverse()
    positions_1 = [y1]
    positions_2 = [y2]
    Y_1 = {}
    Y_2 = {}
    intersect = False
    for i in range(0, n):
        n1 = discrete_noise()
        #n2 = discrete_noise()
        el_1 = positions_1[i] + n1
        el_2 = positions_2[i] + n1
        #print(el_1, el_2)
        # Randomly select either UP or DOWN
        #prob = random.uniform(0, 1)
        # Move the object up or down
        #if prob <= 0.5:
            #y1 = 1
            #y2 = 1
        #elif prob > 0.5:
            #y1 = -1
            #y2 = -1
        # Keep track of the positions
        positions_1.append(el_1)
        positions_2.append(el_2)
    for i, j in zip(timepoints, reversed_timepoints):
        Y_1[i] = positions_1[i]
        Y_2[j] = positions_2[i]
    for i in timepoints:
        if Y_1[i] == Y_2[i]:
            #print(Y_1[i],Y_2[i])
            intersect = True
    #plt.plot(timepoints_steps, Y_1.values())
    #plt.plot(reversed_timepoints_steps, Y_2.values())
    #plt.show()

    return intersect
def rw(T, a, b):
    x1, y1 = 0, a
    x2, y2 = 0, b
    dt = 0.001
    n = int(T / dt)
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = [i for i in range(n + 1)]
    timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints = [i for i in range(n + 1)]
    reversed_timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints.reverse()
    reversed_timepoints_steps.reverse()
    positions_1 = [y1]
    positions_2 = [y2]
    Y_1 = {}
    Y_2 = {}
    intersect = False
    for i in range(0, n):
        n1 = discrete_noise()
        n2 = discrete_noise()
        #el_1 = positions_1[i] + n1
        #el_2 = positions_2[i] + n1
        positions_1.append(n1)
        positions_2.append(n2)
    for i, j in zip(timepoints, reversed_timepoints):
        Y_1[i] = positions_1[i]
        Y_2[j] = positions_2[i]
    for i in timepoints:
        if Y_1[i] == Y_2[i]:
            # print(Y_1[i],Y_2[i])
            intersect = True
    # plt.plot(timepoints_steps, Y_1.values())
    # plt.plot(reversed_timepoints_steps, Y_2.values())
    # plt.show()

    return intersect

#print(random_walk(1, 0, 1))
def Monte_Carlo_b1(sim_n, T):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR(T, 0, 0)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter

def Monte_Carlo_b2(sim_n, T):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR(T, 0, 50)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter

def Monte_Carlo_b3(sim_n, T):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR(T, 0, 100)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter


def Monte_Carlo1(sim_n, b):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR(100, 0, b)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter
#print(Monte_Carlo1(10, 100))
def Monte_Carlo2(sim_n, b, steps):
    n = [i for i in range(1, steps+1)]
    b_s = [i for i in range(0, b)]
    dt = 1./steps
    time = [i * dt for i in range(1, steps+1)]
    probs = []
    for i in n:
        count = 0
        j = 0
        while j < sim_n:
            inter = random_walk(1, i, 0, b)
            if inter == True:
                count += 1
            else:
                count += 0
            j += 1
        prob_inter = (count / sim_n)
        #print("prob", prob_inter)
        probs.append(prob_inter)
    print(probs)
    x = [i for i in range(0, len(probs))]
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot(time, b_s, probs)
    #plt.plot(time, probs)
    #plt.show()

#print(Monte_Carlo2(10, 100, 1000))
def prob_plot(T, b):
    #Ts = [i for i in range(1,T)]
    Ts = [i*0.0001 for i in range(1, T*100)]
    dt = 0.001
    b_s = [i for i in range(0,b - 1)]
    #time = [i * dt for i in range(1,steps)]
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_b = []
    prob_combo = []
    for i in Ts:
        p1 = Monte_Carlo_b1(20, i)
        p2 = Monte_Carlo_b2(20, i)
        p3 = Monte_Carlo_b3(20, i)
        prob_freq1.append(p1)
        prob_freq2.append(p2)
        prob_freq3.append(p3)

    #for i in b_s:
        #p = Monte_Carlo1(5, i)
        #print(p)
        #prob_b.append(p)

    #for i in n:
        #for j in b_s:
            #print(i,j)
            #p = Monte_Carlo2(10, j, i)
            #print(p)
            #prob_combo.append(p)

    combo_x = [i for i in range(len(prob_combo))]
    fig, ax = plt.subplots(3)
    ax[0].set_title("Probability of stopping time detection")
    ax[0].set_xlabel('Different time Δ')
    ax[0].set_ylabel('Probability')
    #ax[1].set_title("Probability of stopping time detection depending on size of end point b")
    #ax[1].set_xlabel('Different end points b')
    #ax[1].set_ylabel('Probability')
    ax[0].plot(Ts, prob_freq1, color="orange", label="Probability of stopping time detection wuth b = 0")
    ax[1].plot(Ts, prob_freq2, color="orange", label="Probability of stopping time detection with b = 50")
    ax[2].plot(Ts, prob_freq3, color="orange", label="Probability of stopping time detection with b = 100")
    #ax[1].plot(b_s, prob_b, color="green", label="Probability of stopping time detection")
    ax[1].set_xlabel('Differences between a and b')
    ax[1].set_ylabel('Probability')
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[2].legend(loc='lower right')
    #plt.legend()
    #ax[0].plot(time, prob_freq, color="orange", label="Probability of stopping time detection")
    #ax[1].plot(b_s, prob_b, color="green", label="Probability of stopping time detection")
    #ax[2].plot(combo_x, prob_combo, color="purple", label="Probability of stopping time detection")
    plt.show()

print(prob_plot(10, 10))