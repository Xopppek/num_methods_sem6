import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x0 = -3
e = 7
a = 1
h1 = 0.01
h2 = 0.001
l = 16
c = 0.7
def tau(h : float):
    return c*h/a
x_min = -l
x_max = l
y_min = -0.1
y_max = 1.45
t_end = 10

def speed(x):
    return -a*np.tanh(x/l)

x_h1 = np.arange(x_min, x_max, h1)
x_h2 = np.arange(x_min, x_max, h2)
t_h1 = np.arange(0, t_end, tau(h1))
t_h2 = np.arange(0, t_end, tau(h2))
animation_t = np.arange(0, t_end, tau(0.1))

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
    return phi4(x)
def mu_l(t):
    #return np.sin(4*t)
    return 0
def mu_r(t):
    #return -np.cos(t*5)
    return 0

def analyt_sol(x, t : float):
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
x_sol_axis.set_xlim(x_min, x_max)
x_sol_axis.set_ylim(y_min, y_max)
x_err_axis.set_xlabel('t')
x_err_axis.set_xlim(0, t_end)
x_err_axis.set_ylim(0, 1)


analyt_line, = x_sol_axis.plot([], [], label = 'аналитическое решение', color='green')

num_line_h1, =  x_sol_axis.plot([], [], label = f'уголок h={h1}', color='orange')
num_line_h2, =  x_sol_axis.plot([], [], label = f'уголок h={h2}', color='dodgerblue')
err_h1, = x_err_axis.plot([], [], label = f'h={h1}', color='orange')
err_h2, = x_err_axis.plot([], [], label = f'h={h2}', color='dodgerblue')

def update(t):
    analyt_line.set_data(x_h2, analyt_sol(x_h2, t))
    if (t == 0):
        num_line_h1.set_data(x_h1, phi(x_h1))
        num_line_h2.set_data(x_h2, phi(x_h2))
        err_h1.set_data([0], [0])
        err_h2.set_data([0], [0])
    else:
        l = x_h1 < 0
        r = x_h1 >= 0
        prev_step_h1 = num_line_h1.get_ydata()
        err_temp = 0
        for i in range(10):
            h1_temp = np.zeros_like(x_h1)
            z = prev_step_h1[l][1:] - speed(x_h1[l][1:]) * tau(h1)/h1 * (prev_step_h1[l][1:] - prev_step_h1[l][:-1])
            h1_temp[l] = np.concatenate(([0], z))
            z = prev_step_h1[r][:-1] - speed(x_h1[r][:-1]) * tau(h1)/h1 * (prev_step_h1[r][1:] - prev_step_h1[r][:-1])
            h1_temp[r] = np.concatenate((z, [0]))
            h1_temp[0] = mu_l(t+i*tau(h1))
            h1_temp[-1] = mu_r(t+i*tau(h1))
            prev_step_h1 = h1_temp
            err_temp = np.max(np.abs(analyt_sol(x_h1, t) - h1_temp))
        num_line_h1.set_data(x_h1, h1_temp)
        err_h1.set_data(np.concatenate((err_h1.get_xdata(), [t])), np.concatenate((err_h1.get_ydata(), [err_temp])))
        #print(err_h1.get_ydata())
        #print(np.abs(analyt_sol(x_h1, t) - h1_temp)[1500:-1000])
        l = x_h2 < 0
        r = x_h2 >= 0
        prev_step_h2 = num_line_h2.get_ydata() 
        for i in range(100):
            h2_temp = np.zeros_like(x_h2)
            z = prev_step_h2[l][1:] - speed(x_h2[l][1:]) * tau(h2)/h2 * (prev_step_h2[l][1:] - prev_step_h2[l][:-1])
            h2_temp[l] = np.concatenate(([0], z))
            z = prev_step_h2[r][:-1] - speed(x_h2[r][:-1]) * tau(h2)/h2 * (prev_step_h2[r][1:] - prev_step_h2[r][:-1])
            h2_temp[r] = np.concatenate((z, [0]))
            h2_temp[0] = mu_l(t+i*tau(h2))
            h2_temp[-1] = mu_r(t+i*tau(h2))
            prev_step_h2 = h2_temp
            err_temp = np.max(np.abs(analyt_sol(x_h2, t) - h2_temp))
        num_line_h2.set_data(x_h2, h2_temp)
        err_h2.set_data(np.concatenate((err_h2.get_xdata(), [t])), np.concatenate((err_h2.get_ydata(), [err_temp])))

    x_sol_axis.set_title(f'time = {t:.2f}')
    x_err_axis.set_title(r"""$\epsilon_1/\epsilon_2$ =""" + f'{err_h1.get_ydata()[-1]/(err_h2.get_ydata()[-1] + 0.0001):.2f}')

anim = FuncAnimation(fig, update, frames=animation_t, init_func=None, interval=1)

x_sol_axis.legend()
x_err_axis.legend()
#anim.save('analitical.gif')
plt.show()
