import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

x0 = -3
e = 8
a = 30
h = 0.1
l = 16
c = 0.7
tau = c*h/a
x_min = -l
x_max = l
y_min = -0.1
y_max = 1.1

x = np.arange(x_min, x_max, h)
t = np.arange(2, 3.3, tau)

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
def mu_l(x):
    #return np.zeros_like(x)
    return np.sin(x*40)
def mu_r(x):
    #return np.zeros_like(x)
    return -np.cos(x*55)

def analyt_sol(x, t):
    x_temp = l * np.arcsinh(np.sinh(1)*np.exp(-a*t/l))
    temp = l * np.arcsinh(np.sinh(x[np.abs(x) <= x_temp]/l)*np.exp(a*t/l))
    temp_l = t + l / a * np.log(np.sinh(np.abs(x[x < -x_temp])/l)/np.sinh(1))
    temp_r = t + l / a * np.log(np.sinh(np.abs(x[x > x_temp])/l)/np.sinh(1))
    res = np.zeros_like(x)
    res[x < -x_temp] = mu_l(temp_l)
    res[np.abs(x) <= x_temp] = phi(temp)
    res[x > x_temp] = mu_r(temp_r)
    return res

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
x_sol_axis, x_err_axis = axes[0], axes[1]
x_sol_axis.set_xlabel('x')
x_sol_axis.set_xlim(x_min, x_max)
x_sol_axis.set_ylim(y_min, y_max)
x_err_axis.set_xlabel('x')
x_err_axis.set_xlim(x_min, x_max)

analyt_line, = x_sol_axis.plot([], [])

def update(t):
    curr_analyt = analyt_sol(x, t)
    analyt_line.set_data(x, curr_analyt)
    return analyt_line

anim = FuncAnimation(fig, update, frames=t, init_func=None, interval=3)

#x_sol_axis.legend()
#writer = PillowWriter(fps=15,metadata=dict(artist='Me'), bitrate=900)
#anim.save('analitical.gif', writer=writer)
plt.show()
#check