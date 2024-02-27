import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x0 = 1
e = 2
a = 3
h = 0.01
c = 0.67
tau = c*h/a
x_min = 0
x_max = 30
y_min = -0.1
y_max = 1.1

x = np.arange(x_min, x_max, h)
t = np.arange(0, 5, tau)

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

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
x_sol_axis, x_err_axis = axes[0], axes[1]
x_sol_axis.set_xlabel('x')
x_sol_axis.set_xlim(x_min, x_max)
x_sol_axis.set_ylim(y_min, y_max)
x_err_axis.set_xlabel('x')
x_err_axis.set_xlim(x_min, x_max)

analyt_line, = x_sol_axis.plot([], [], label='analytical')

def update(t):
    curr_analyt = analyt_sol(x, t)
    analyt_line.set_data(x, curr_analyt)
    return analyt_line

anim = FuncAnimation(fig, update, frames=t, init_func=None, interval=5)

x_sol_axis.legend()
plt.show()