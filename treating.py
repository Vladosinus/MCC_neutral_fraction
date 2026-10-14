import re
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from initiate_constants import *
import auxillary_treater

filepath = 'output/mean_vals.txt'

sections = auxillary_treater.read_section_data(filepath)
l = np.linspace(0, L, len(sections))
drift_speed = np.zeros(len(sections)-1)

# Считаем скорость дрейфа для всех сечений
for idx in range(len(sections)-1):
    drift_speed[idx] = (L - sections[idx]['x'])/sections[idx]['mean_time']
l1 = l[:-1]

times = np.zeros(len(sections))
for i in range(len(sections)):
    times[i] = sections[i]['mean_time']
vx = (L - l[:-2])/times[:-2]

# Так как в конце частицы свободно покидают без столкновений пока аппроксимируем линецно.
extrapolator = interp1d(l[:-2], vx, kind = 'linear', fill_value='extrapolate', bounds_error=False)
v_drift = extrapolator(l)

## На этом этапе есть l и v_drift некоторой длины, неважно, какой
# Шаг вдоль нейтрализатора
dl = l[1] - l[0]
# Коэффициент диффузии для частиц, образовавшихся в результате диссоциации
D_diss = 1/3*v_drift*Dh*k

## Параметры для просто газа

# Коэффициент диффузии газа
D_gas = 1/3*((3*kB*T/M2)**0.5)*Dh*k
# Поток в нейтрализатор
inflow = Q/(kB*T*a*b)

# Пакуем в одну перемеенную, чтобы отправить в функцию расчета
l2set = l
D2set = D_gas
S2set = 0
vars = l2set, dl, D2set, S2set, inflow, v_drift

solution = auxillary_treater.calculate_diffusion_flow(vars, type = 'gas')
n_gas = solution.y[:, -1]
n_gas_still = n_gas

# Сечения процессов
generation_sigmas = auxillary_treater.load_sigmas()

## Считаем изменение компонентного состава пучка
sigmas = [generation_sigmas[0][1], generation_sigmas[1][1]]

## Считаем генерацию частиц пучком, на выходе нужно получить объемное рождение в каждом
# сечении по длине
# beam_properties = v_beam, n_beam
# Q = auxillary_treater.generation_by_beam(n_gas, generation_sigmas, beam_properties)
# Q_full = np.sum(Q, axis = 0)
# Строим график
# auxillary_treater.plot_generation(l, Q)

# n_atoms = auxillary_treater.distributions(l, D_diss, Q_full, inflow, v_drift)

def plot_atoms():
    fontdict_labels = {'fontsize':26,
                        'fontfamily': 'Times New Roman'}
    fontdict_ticks = {'fontsize':24,
                    'fontfamily': 'Times New Roman'}
    fontdict_title = {'fontsize':28,
                    'fontfamily': 'Times New Roman'}
    plt.rcParams['mathtext.fontset'] = 'stix'

    figure2 = plt.figure(figsize=(13, 9))
    ax2 = figure2.add_subplot()

    for spine in ax2.spines.values():
        spine.set_linewidth(2)     # толщина
        spine.set_color('black')   # цвет

    l1 = np.linspace(0, l[-1], 50)
    sum_interpolator = interp1d(l, n_atoms.sum(axis = 0), fill_value = 0, bounds_error=False, kind = 'cubic')
    n_sum = sum_interpolator(l1)

    ax2.plot(l1, n_sum, color = '#000080', lw = 3, label = 'Концентрация атомов водорода')
    ax2.plot(l, n_gas, color = "#800049", lw = 3, label = 'Концентрация молекул водорода')
    ax2.plot(l, n_gas_still, color = "#0B8000", lw = 3, label = 'Концентрация молекул водорода\nбез пучка')

    ax2.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 24})

    s = f"{np.max([np.max(n_gas), np.max(n_sum), np.max(n_gas_still)]):e}"

    mantissa_str, exp_str = s.split('e')
    _y_lim_max = (math.ceil(float(mantissa_str)*10)/10)*10**(float(exp_str))
    

    ax2.grid(True)
    ax2.set_xlim([-0.01*L, L + 0.01*L])
    xticks = ax2.get_xticks()
    ax2.set_xticklabels([f'{t:.1f}' for t in xticks], fontdict = fontdict_ticks)

    ax2.set_xlabel(fr'Координата вдоль пучка, $м$', fontdict_labels)
    ax2.set_ylabel(fr'Концентрация, $м^{{-3}}$', fontdict_labels)

    ax2.set_ylim([-0.01*_y_lim_max, _y_lim_max + 0.01*_y_lim_max])

    ticks = np.arange(0, _y_lim_max/10**float(exp_str) + _y_lim_max/10**float(exp_str)/10, _y_lim_max/10**float(exp_str)/10)
    ax2.set_yticks(ticks*10**float(exp_str))

    tick_labels = np.array([f'{t:.1f}' for t in ticks])
    ax2.set_yticklabels(tick_labels, fontdict = fontdict_ticks)

    ax2.text(-0.01, 1.02,
            fr'$10^{{{float(exp_str):.0f}}}$',
            fontsize = fontdict_ticks['fontsize'] + 2,
            transform = ax2.transAxes)

    plt.tight_layout()
    plt.show()

def plot_beam_comp(y):

    i0 = y[0]
    i1 = y[1]

    fontdict_labels = {'fontsize':26,
                        'fontfamily': 'Times New Roman'}
    fontdict_ticks = {'fontsize':24,
                    'fontfamily': 'Times New Roman'}
    fontdict_title = {'fontsize':28,
                    'fontfamily': 'Times New Roman'}
    plt.rcParams['mathtext.fontset'] = 'stix'

    figure2 = plt.figure(figsize=(13, 9))
    ax2 = figure2.add_subplot()

    for spine in ax2.spines.values():
        spine.set_linewidth(2)     # толщина
        spine.set_color('black')   # цвет

    l1 = np.linspace(0, l[-1], 50)
    i0_interpolator = interp1d(l, i0, fill_value = 0, bounds_error=False, kind = 'cubic')
    i1_interpolator = interp1d(l, i1, fill_value = 0, bounds_error=False, kind = 'cubic')
    I1 = i0_interpolator(l1)
    I0 = i1_interpolator(l1)

    ax2.plot(l1, I1, color = '#000080', lw = 3, label = 'Ток протонов')
    ax2.plot(l1, I0, color = "#800049", lw = 3, label = 'Ток атомов')
    

    ax2.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 24})

    s = f"{np.max([I1, I0]):e}"

    mantissa_str, exp_str = s.split('e')
    _y_lim_max = (math.ceil(float(mantissa_str)*10)/10)*10**(float(exp_str))
    

    ax2.grid(True)
    ax2.set_xlim([-0.01*L, L + 0.01*L])
    xticks = ax2.get_xticks()
    ax2.set_xticklabels([f'{t:.1f}' for t in xticks], fontdict = fontdict_ticks)

    ax2.set_xlabel(fr'Координата вдоль пучка, $м$', fontdict_labels)
    ax2.set_ylabel(fr'Ток, А', fontdict_labels)

    ax2.set_ylim([-0.01*_y_lim_max, _y_lim_max + 0.01*_y_lim_max])

    ticks = np.arange(0, _y_lim_max/10**float(exp_str) + _y_lim_max/10**float(exp_str)/10, _y_lim_max/10**float(exp_str)/10)
    ax2.set_yticks(ticks*10**float(exp_str))

    tick_labels = np.array([f'{t:.1f}' for t in ticks])
    ax2.set_yticklabels(tick_labels, fontdict = fontdict_ticks)


    plt.tight_layout()
    plt.show()


# auxillary_treater.plot_atoms_distributions(l, n_atoms)

## Закикленный расчет
iter_number = 0
n_gas_old = n_gas
while iter_number < 5:

    I = auxillary_treater.beam_transform(n_gas, sigmas, L, l)
    n_beam = I/(qe*v_beam*a1*b1)
    print(np.max(I[0]))
    

    
    vars = l, dl, D_gas, inflow, n_beam, v_beam, generation_sigmas, n_gas
    solution = auxillary_treater.calculate_diffusion_flow_with_beam(vars)
    n_gas = solution.y[:, -1]

    beam_properties = v_beam, n_beam
    Q = auxillary_treater.generation_by_beam(n_gas, generation_sigmas, beam_properties)
    Q_full = np.sum(Q, axis = 0)

    n_atoms = auxillary_treater.distributions(l, D_diss, Q_full, inflow, v_drift)

    # 
    
    # auxillary_treater.plot_generation(l, Q)
    # plt.plot(solution.t, solution.y[0, :]/np.max(solution.y[0, :]), 'red')
    # plt.plot(solution.t, solution.y[-1, :]/np.max(solution.y[-1, :]), 'green')
    # plt.show()
    # exit()
    if iter_number == 4:
        # fig = plt.figure(figsize=(13, 9))
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2)

        # ax1.plot(l, I[0])
        # ax1.plot(l, I[1])

        # ax1.grid(True)

        # ax2.plot(l, n_gas)
        # ax2.plot(l, n_gas_old)

        # ax2.grid(True)
        
        
        # plt.tight_layout()
        # plt.show()

        plot_atoms()

        auxillary_treater.plot_atoms_distributions(l, n_atoms)

        plot_beam_comp(I)


    # exit()

    iter_number += 1


















