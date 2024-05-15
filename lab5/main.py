import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = np.sqrt(0.5)
c = 0.7

y_min = -0.2
y_max = 2.0
x_min = 0
x_max = 1
t_end = 1

k = 1
h = 0.01
sig = 0.5
tau = c*h/k #c* h**2 / (2*k)

alpha_l = 0
beta_l = -1
alpha_r = 1
beta_r = -3

x_h = np.arange(x_min, x_max+h, h)
animation_t = np.arange(0, t_end+tau, tau)

def f(x, t):
    return x-t**2*np.cosh(x*t)+x*np.sinh(x*t)

def phi(x):
    return 1-x

def phi_xx(x):
    return 0

def mu_l(t):
    return t-1

def mu_r(t):
    return 2*(1-t) + np.cosh(t) - 3*t*np.sinh(t)

def analyt_sol(x, t):
    return x*(t-1) + np.cosh(x*t)

def leftSecond(u, frame, hs):
    return (mu_l(frame) + (u[2]-4*u[1])/(2*hs)) /(3*(1-1/(2*hs)))

def rightSecond(u, frame, hs):
    return (mu_r(frame) + (u[-3] - 4 * u[-2])/hs)/(1 - 3/hs)

def TDMA(a, b, c, d):
    n = len(d)
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

def F(x,u_prev,t,dt,dx):
    result = np.ones_like(x)
    result[0] = mu_l(t)
    result[-1] = mu_r(t)
    result[1:-1] = f(x[1:-1], t - dt/2) + u_prev[1:-1]/dt + (1-sig)*k*(u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2])/dx**2
    return result

def numerical_solution(u_prev, x, t, dx, dt):
    F_cur = F(x,u_prev,t,dt,dx)
    a =  (-sig*k/dx**2)*np.ones_like(F_cur)
    a[0] = 0
    b = (1/dt + 2*sig*k/dx**2)*np.ones_like(F_cur)
    c = (-sig*k/dx**2)*np.ones_like(F_cur)
    c[-1] = 0
    
    F_cur[0] -= (beta_l*F_cur[1])/(2*dx*c[1])
    F_cur[-1] -= (beta_r*F_cur[-2])/(2*dx*a[-2])
    
    c[0] = -2*beta_l/dx - (beta_l*b[1])/(2*dx*c[1])
    b[0] = (alpha_l + 1.5*beta_l/dx) - (beta_l*a[1])/(2*dx*c[1])
    b[-1] = (alpha_r + 1.5*beta_r/dx) - (0.5*beta_r/dx)*(c[-2]/a[-2])
    a[-1] = (-2*beta_r/dx) - (0.5*beta_r/dx)*(b[-2]/a[-2])
    return TDMA(a[1:],b,c[:-1],F_cur)

fig, ax = plt.subplots(figsize=(12, 8))
analyt_line, = ax.plot([], [], label = 'Аналитическое решение', color='green')
num_line, =  ax.plot([], [], label = f'Кранка–Николсон h={h}', color='orange')

u = phi(x_h)

def init_animation():
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return num_line, analyt_line

def update(frame):
    global u
    u_analyt = analyt_sol(x_h, frame)
    analyt_line.set_data(x_h, u_analyt)
    u_new = numerical_solution(u,x_h,frame+tau,h,tau)
    num_line.set_data(x_h, u)
    ax.set_title(f'time = {frame:.2f}')
    if (frame > t_end - 0.5* tau):
        print(np.max(np.abs(u - u_analyt)))

    u = np.copy(u_new)

    return num_line, analyt_line
    
anim = FuncAnimation(fig=fig, func=update, frames=animation_t, init_func=init_animation, interval=h*3000, repeat=False, blit=True)

ax.grid(True)
ax.legend()

plt.show()