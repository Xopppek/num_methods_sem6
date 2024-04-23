import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x0 = -3
e = 7
a = np.sqrt(0.5)
h1 = 0.01
h2 = 0.001
c = 0.7
def tau(h : float):
    return c*h/a

y_min = 0.9
y_max = 2.5
x_min = 0
x_max = 1
t_end = 1

x_h1 = np.arange(x_min, x_max, h1)
x_h2 = np.arange(x_min, x_max, h2)
t_h1 = np.arange(0, t_end, tau(h1))
t_h2 = np.arange(0, t_end, tau(h2))
anim_step = 0.01
animation_t = np.arange(0, t_end, tau(anim_step))

def f(x, t):
    return ((1+t)**2 -2*x**2)/8 /(1+x+x*t)**1.5

def phi(x):
    return np.sqrt(1+x)

def phi_xx(x):
    return -1/4 /(1+x)**1.5

def psi(x):
    return x/2/np.sqrt(1+x)

def mu_l(t):
    return (7+t)/2

def mu_r(t):
    return 1/np.sqrt(2+t)

def analyt_sol(x, t):
    return np.sqrt(1+x+x*t)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
x_sol_axis, x_err_axis = axes[0], axes[1]
x_sol_axis.set_xlim(x_min, x_max)
x_sol_axis.set_ylim(y_min, y_max)
x_err_axis.set_xlabel('t')
x_err_axis.set_xlim(0, t_end)
x_err_axis.set_ylim(0, 0.005)


analyt_line, = x_sol_axis.plot([], [], label = 'Аналитическое решение', color='green')

num_line_h1, =  x_sol_axis.plot([], [], label = f'Крест h={h1}', color='orange')
num_line_h2, =  x_sol_axis.plot([], [], label = f'Крест h={h2}', color='dodgerblue', ls='--')
old_line_h1, = x_sol_axis.plot([], [], alpha=0)
old_line_h2,= x_sol_axis.plot([], [], alpha=0)
err_h1, = x_err_axis.plot([], [], label = f'h={h1}', color='orange')
err_h2, = x_err_axis.plot([], [], label = f'h={h2}', color='dodgerblue')


def get_line(x, h, prev, t, old):
    num = (int)(anim_step/h)
    err_temp = 0
    curr = np.zeros_like(x)
    for i in range(num):
        curr[1:-1] = (tau(h)*a/h)**2 *(prev[2:] - 2*prev[1:-1] + prev[0:-2]) + tau(h)**2 *f(x[1:-1], t + i*tau(h)) - old[1:-1] + 2*prev[1:-1]
        #curr[0] = (mu_l(t+i*tau(h)) + curr[2]/(2*h) - 2*curr[1]/h) / (3*(1 - 1/(2*h))) # левое 2 порядок
        curr[0] = (mu_l(t+i*(tau(h))) - curr[1]/h)/(3-1/h) # левое 1 порядок
        #curr[-1] = (mu_r(t+i*tau(h)) - 4/h * curr[-2] + curr[-3]/h)/(1 - 3/h) # правое 2 порядок
        curr[-1] = (mu_r(t+i*tau(h)) - 2*curr[-2]/h)/(1-2/h) # правое 1 порядок
        old[:] = prev
        prev[:] = curr
    err_temp = np.max(np.abs(analyt_sol(x, t + num * tau(h)) - curr))
    return curr, err_temp, old


def update(t):
    analyt_line.set_data(x_h2, analyt_sol(x_h2, t + tau(anim_step)))
    if (t == 0):
        old_line_h1.set_data(x_h1, phi(x_h1))
        old_line_h2.set_data(x_h2, phi(x_h2))
        #num_line_h1.set_data(x_h1, phi(x_h1) + tau(h1)*psi(x_h1) + tau(h1)**2 / 2 *(a**2 * phi_xx(x_h1) + f(x_h1, 0))) # начальное 2 порядок
        #num_line_h2.set_data(x_h2, phi(x_h2) + tau(h2)*psi(x_h2) + tau(h2)**2 / 2 *(a**2 * phi_xx(x_h2) + f(x_h2, 0))) # начальное 2 порядок
        num_line_h1.set_data(x_h1, phi(x_h1) + tau(h1)*psi(x_h1)) # начальное 1 порядок
        num_line_h2.set_data(x_h2, phi(x_h2) + tau(h2)*psi(x_h2)) # начальное 1 порядок
        err_h1.set_data([0], [0])
        err_h2.set_data([0], [0])
    
    prev_step = num_line_h1.get_ydata()
    old = old_line_h1.get_ydata()
    h1_temp, err_temp, old = get_line(x_h1, h1, prev_step, t, old)
    num_line_h1.set_ydata(h1_temp)
    old_line_h1.set_ydata(old)
    err_h1.set_data(np.concatenate((err_h1.get_xdata(), [t])), np.concatenate((err_h1.get_ydata(), [err_temp])))

    prev_step = num_line_h2.get_ydata()
    old = old_line_h2.get_ydata()
    h2_temp, err_temp, old = get_line(x_h2, h2, prev_step, t, old)
    num_line_h2.set_ydata(h2_temp)
    old_line_h2.set_ydata(old)
    err_h2.set_data(np.concatenate((err_h2.get_xdata(), [t])), np.concatenate((err_h2.get_ydata(), [err_temp])))

    #x_sol_axis.set_ylim(y_min, 1.2*max(np.max(num_line_h1.get_ydata()), np.max(num_line_h2.get_ydata())))    
    x_err_axis.set_ylim(0, 1.3 * max(np.max(err_h1.get_ydata()), np.max(err_h2.get_ydata())))
    x_sol_axis.set_title(f'time = {t:.2f}')
    x_err_axis.set_title(r"""$\epsilon_1/\epsilon_2$ =""" + f'{err_h1.get_ydata()[-1]/(err_h2.get_ydata()[-1] + 0.00000001):.2f}')

    
    
anim = FuncAnimation(fig, update, frames=animation_t, init_func=None, interval=2)

x_sol_axis.legend()
x_err_axis.legend()
#anim.save('wtf5.gif')
plt.show()