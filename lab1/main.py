import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x0 = 4
e = 3
a = 0.5
h = 0.01
c = 0.67
tau = c*h/a

def xi(x):
    return np.abs(x-x0)/e
def phi1(x):
    return np.heaviside(1-xi(x), 0)
def phi2(x):
    return phi1(x)*(1-xi(x)**2)
def phi3(x):
    return phi1(x)*np.exp(-xi(x)**2 / np.abs(1-xi(x)**2))
def phi4(x):
    return phi1(x)*np.cos(np.pi*xi(x)/2)**3

def phi(x):
    return phi1(x)
def mu(x):
    return np.zeros_like(x)

def analyt_sol(x, t):
    return phi(x-a*t)*np.heaviside(x-a*t, 1) + mu(t - x/a) * np.heaviside(a*t-x, 0)


x = np.arange(0, 20, h)
t = np.arange(0, 5, tau)
plt.plot(x, analyt_sol(x, 1))
plt.show()

'''
x = np.linspace(0, 10, 100)
plt.plot(x, phi1(x), label='1')
plt.plot(x, phi2(x), label='2')
plt.plot(x, phi3(x), label='3')
plt.plot(x, phi4(x), label='4')
plt.legend()
plt.show()
'''
