import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = np.sqrt(0.5)
c = 0.7
def tau(h):
    return c*h/a

y_min = 0.9
y_max = 1.8
x_min = 0
x_max = 1
t_end = 1

h = 0.001

x_h = np.arange(x_min, x_max+h, h)
animation_t = np.arange(0, t_end+tau(h), tau(h))

def f(x, t):
    return ((1+t)**2 -2*x**2)/8 /(1+x+x*t)**1.5

def phi(x):
    return np.sqrt(1+x)

def phi_xx(x):
    return -1/4 /(1+x)**1.5

def psi(x):
    return x/(2*np.sqrt(1+x))

def mu_l(t):
    return (7+t)/2

def mu_r(t):
    return 1/np.sqrt(2+t)

def analyt_sol(x, t):
    return np.sqrt(1+x+x*t)

def leftSecond(u, frame, hs):
    return (mu_l(frame) + (u[2]-4*u[1])/(2*hs)) /(3*(1-1/(2*hs)))

def rightSecond(u, frame, hs):
    return (mu_r(frame) + (u[-3] - 4 * u[-2])/hs)/(1 - 3/hs)


fig, ax = plt.subplots(figsize=(12, 8))
analyt_line, = ax.plot([], [], label = 'Аналитическое решение', color='green')
num_line, =  ax.plot([], [], label = f'Крест h={h}', color='orange')

u0 = phi(x_h)
u1 = phi(x_h) + tau(h)*psi(x_h) + tau(h)**2 / 2 *(a**2 * phi_xx(x_h) + f(x_h, 0))
t = -1


def init_animation():
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return num_line, analyt_line


def update(frame):

    global u0
    global u1
    global t
    t+=1 

    u_analyt = analyt_sol(x_h, frame)
    analyt_line.set_data(x_h, u_analyt)

    if (t == 0):
        num_line.set_data(x_h, u0)
    elif (t == 1):
        num_line.set_data(x_h, u1)
    else:
        u_new = np.copy(u1)
        u_new[1:-1] = (a**2 * (u1[2:] - 2*u1[1:-1]+u1[:-2])/h**2 + f(x_h[1:-1], frame - tau(h)))*tau(h)**2-u0[1:-1]+2*u1[1:-1]
        u_new[0] = leftSecond(u_new, frame, h)
        u_new[-1] = rightSecond(u_new, frame, h)
        u0 = np.copy(u1)
        u1 = np.copy(u_new)

        num_line.set_data(x_h, u_new)
        ax.set_title(f'time = {frame:.2f}')
        if (frame > t_end - tau(h)):
            print(np.max(np.abs(u_new - u_analyt)))

    return num_line, analyt_line
    
anim = FuncAnimation(fig=fig, func=update, frames=animation_t, init_func=init_animation, interval=h*3000, repeat=False, blit=True)

ax.grid(True)
ax.legend()

plt.show()