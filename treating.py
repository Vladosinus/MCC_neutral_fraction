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

# Сечения процессов
generation_sigmas = auxillary_treater.load_sigmas()

## Считаем изменение компонентного состава пучка
sigmas = [generation_sigmas[0][1], generation_sigmas[1][1]]
I = auxillary_treater.beam_transform(n_gas, sigmas, L, l)
n_beam = I/(qe*v_beam*a1*b1)

## Считаем генерацию частиц пучком, на выходе нужно получить объемное рождение в каждом
# сечении по длине
beam_properties = v_beam, n_beam
Q = auxillary_treater.generation_by_beam(n_gas, generation_sigmas, beam_properties)
Q_full = np.sum(Q, axis = 0)
# Строим график
# auxillary_treater.plot_generation(l, Q)

n_atoms = auxillary_treater.distributions(l, D_diss, Q_full, inflow, v_drift)

auxillary_treater.plot_atoms_distributions(l, n_atoms)


















