import re
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import random

from initiate_constants import *

# Тут строятся графики, чтобы в каждый не вписывать
fontdict_labels = {'fontsize':26,
                    'fontfamily': 'Times New Roman'}
fontdict_ticks = {'fontsize':24,
                'fontfamily': 'Times New Roman'}
fontdict_title = {'fontsize':28,
                'fontfamily': 'Times New Roman'}
plt.rcParams['mathtext.fontset'] = 'stix'

# Загрузка результатов Монте-Карло
def read_section_data(filepath):
    """Читает mean_vals.txt с данными по сечениям.

    Возвращает:
        sections - список словарей вида
                   {'index': k, 'x': x, 'times': np.array,
                    'collisions': np.array,
                    'mean_time': float, 'mean_collisions': float}

    Формат файла (повторяется для каждого сечения):
        # section <k>, l = <координата> м
        time alive, seconds   collisions
        <время частицы, с>    <столкновений частицы, шт>
        ...
        mean time alive, seconds      mean collisions
        <среднее время, с>            <среднее столкновений, шт>
    """
    sections = []
    current = None
    # После заголовка "mean time alive..." идёт строка со средними значениями,
    # её нужно сохранить отдельно, а не как ещё одну частицу.
    expect_mean = False

    section_re = re.compile(r'#\s*section\s+(\d+)\s*,\s*l\s*=\s*([-\d.eE+]+)')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            m = section_re.match(stripped)
            if m:
                # Начался новый блок сечения
                if current is not None:
                    sections.append(current)
                current = {
                    'index': int(m.group(1)),
                    'x': float(m.group(2)),
                    'times': [],
                    'collisions': [],
                    'mean_time': None,
                    'mean_collisions': None,
                }
                expect_mean = False
                continue

            # Прочие строки-комментарии пропускаем
            if stripped.startswith('#'):
                continue

            if current is None:
                continue

            # Заголовок "mean time alive, seconds   mean collisions":
            # следующая числовая строка — средние значения по сечению
            if stripped.lower().startswith('mean'):
                expect_mean = True
                continue

            parts = stripped.split()
            try:
                t = float(parts[0])
                c = float(parts[1])
            except (ValueError, IndexError):
                # Строка-заголовок "time alive, seconds   collisions" и т.п.
                continue

            if expect_mean:
                current['mean_time'] = t
                current['mean_collisions'] = c
                expect_mean = False
                continue

            current['times'].append(t)
            current['collisions'].append(c)

    if current is not None:
        sections.append(current)

    # Преобразуем списки в массивы
    for s in sections:
        s['times'] = np.array(s['times'])
        s['collisions'] = np.array(s['collisions'])

    return sections
# Общая функция для расчета концентрации
def calculate_diffusion_flow(vars, type = 'gas'):
    l, dl, D, S, inflow, v_drift = vars

    # Просто натекание из левой границы, поток
    if type == 'gas':
        t_pass = L/((3*kB*T/M2)**0.5)
        t_max = 100*t_pass

        def diffusion(t, n):
            dndt = np.zeros_like(n)
            for i in range(1, len(dndt) - 1):
                d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
                dndt[i] = D*d2n_dx2

            dndt[0] = 2*D*(n[1] - n[0])/dl**2 + 2*inflow/dl
            dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2

            return dndt
        
        n_IC = np.zeros(len(l))
        solution = solve_ivp(fun=diffusion,
                            t_span=(0, t_max),
                            y0=n_IC,
                            method='RK45')
        
    elif type == 'dissociation':

        # Оценочное время прохождения частиц через нейтрализатор
        t_pass = L/v_drift
        t_max = 100*t_pass

        def diffusion(t, n):

            dndt = np.zeros_like(l)

            for i in range(1, len(dndt) - 1):
                d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
                dndt[i] = D*d2n_dx2 + S

            # dndt[0] = D[0]*(n[2] - 2*n[1] + n[0])/dl**2 + S[0]
            dndt[0] = D*(n[1] - n[0])/dl**2 + S
            dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2
            return dndt

        n_IC = np.zeros(len(l))
        solution = solve_ivp(fun=diffusion,
                    t_span=(0, t_max),
                    y0=n_IC,
                    method='RK45')
        
    return solution
# Общая функция для расчета генерации АТОМОВ пучком
def generation_by_beam(n_gas, generation_sigmas, beam_properties):
    v, n_beam = beam_properties

    ## Генерация атомами
    neutral_total_sigma = generation_sigmas[0][2] + 2*generation_sigmas[0][3] + generation_sigmas[0][6] + generation_sigmas[0][8]
    Q0 = n_gas*n_beam[0]*neutral_total_sigma*v
    
    ## Генерация ионами
    proton_total_sigma = 2*generation_sigmas[1][3] + generation_sigmas[1][5]
    Q1 = n_gas*n_beam[1]*proton_total_sigma*v
    
    return [Q0, Q1]
# Загрузка всех сечений из файла
def load_sigmas():
    numbers = []
    started = False # Маркер начала данных
    ## Загружаются сечения дли ионов
    with open('proton_sigmas.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not started:
                if line == '':
                    started = True
                continue
            if started:
                try:
                    row = [float(x) for x in line.split()]
                    numbers.append(row)
                except ValueError:
                    continue
    proton_sigmas = np.array(numbers).ravel()

    numbers = []
    started = False
    ## Загружаются сечения дли нейтралов
    with open('neutral_sigmas.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not started:
                    if line == '':
                        started = True
                    continue
                if started:
                    try:
                        row = [float(x) for x in line.split()]
                        numbers.append(row)
                    except ValueError:
                        continue
    neutral_sigmas = np.array(numbers).ravel()
    generation_sigmas = np.array([neutral_sigmas, proton_sigmas], dtype = object)
    return generation_sigmas
# Обычный расчет компонентного состава
def beam_transform(n_gas, sigmas, L, l):

    ## Тут будет логика расчета, и вызов решателя
    # Правая часть будет просто объявлена, чтобы не делать функцию внутри функции
    
    t_span = (0, L)
    # Нормировано на 1, лучше так делать, а потом уже умножать на всякое, если надо
    I0 = [0, 1]
    args = n_gas, sigmas, l

    solution = solve_ivp(rbt,
                         t_span = t_span,
                         y0 = I0,
                         method = 'RK45',
                         dense_output = True,
                         args = args,
                         max_step = L/20
                         )

    I_interpolated = solution.sol(l)
    
    return I_interpolated
# Правая часть для beam_transform
def rbt(t, I, n_gas_origin, sigmas, l):

    gas_interpolator = interp1d(l, n_gas_origin, fill_value=0, bounds_error=False, kind = 'linear')
    n_gas = gas_interpolator(t)

    sigma01 = sigmas[0]
    sigma10 = sigmas[1]    

    dI0 = n_gas*I[1]*sigma10 - n_gas*I[0]*sigma01
    dI1 = n_gas*I[0]*sigma01 - n_gas*I[1]*sigma10

    return [dI0, dI1]
# Функция для построения графика генерации
def plot_generation(l, Q):

    # Пересчет на подробную сетку, чтобы график был красивый
    l1 = np.linspace(l.min(), l.max(), 50)
    interpolator_0 = interp1d(l, Q[0], kind = 'cubic', fill_value='extrapolate', bounds_error=False)
    interpolator_1 = interp1d(l, Q[1], kind = 'cubic', fill_value='extrapolate', bounds_error=False)
    Q_0 = interpolator_0(l1)
    Q_1 = interpolator_1(l1)
    Q = np.array([Q_0, Q_1])
    l = l1

    Q_full = np.sum(Q, axis = 0)
    _Q_max = np.max(Q_full)

    
    figure1 = plt.figure(figsize=(13, 9))
    ax = figure1.add_subplot()

    for spine in ax.spines.values():
        spine.set_linewidth(2)     # толщина
        spine.set_color('black')   # цвет
    
    ax.plot(l, Q[0], lw = 3, label = 'Генерация атомами пучка', color = "#0E0EB4")
    ax.plot(l, Q[1], lw = 3, label = 'Генерация ионами пучка', color = "#2E792D")
    ax.plot(l, Q_full, lw = 4, label = 'Суммарная генерация', color = "#AB4082")

    ax.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 24}) #, 'style': 'italic'})

    ax.set_xlabel(fr'Координата вдоль пучка, $м$', fontdict_labels)
    ax.set_ylabel(fr'Скорость рождения частиц, $м^{{-3}}/с$', fontdict_labels)

    ax.grid(True)
    ax.set_xlim([-0.01*L, L + 0.01*L])
    xticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.1f}' for t in xticks], fontdict = fontdict_ticks)

    # Создаем максимальное значение по y
    s = f"{_Q_max:e}"
    mantissa_str, exp_str = s.split('e')
    _y_lim_max = (math.ceil(float(mantissa_str)))*10**(float(exp_str))

    ax.set_ylim([-0.01*_y_lim_max, _y_lim_max + 0.01*_y_lim_max])

    ticks = np.arange(0, _y_lim_max/10**float(exp_str) + _y_lim_max/10**float(exp_str)/10, _y_lim_max/10**float(exp_str)/10)
    ax.set_yticks(ticks*10**float(exp_str))

    tick_labels = np.array([f'{t:.1f}' for t in ticks])
    ax.set_yticklabels(tick_labels, fontdict = fontdict_ticks)

    ax.text(-0.01, 1.02,
            fr'$10^{{{float(exp_str):.0f}}}$',
            fontsize = fontdict_ticks['fontsize'] + 2,
            transform = ax.transAxes)

    plt.tight_layout()
    plt.show()
# Функция для расчета распределений
def distributions(l, D_diss, Q_full, inflow, v_drift):
    concentration = []
    dl = l[1] - l[0]
    # Цикл по сечениям
    for idx in range(len(l)-1):

        # Эти значения мы будем отправлять в считалку распределений, чтобы она думала, что мы считаем такой короткий участок
        l2set = l[idx:]
        D2set = D_diss[idx]
        S2set = Q_full[idx]
        vars = l2set, dl, D2set, S2set, inflow, v_drift[idx]
        # Вызывем решатель
        solution = calculate_diffusion_flow(vars, type = 'dissociation')
        # Массив концентраций от конкретного сечения, длиной len(l) - idx, надо в начало дописать нулей
        temp_n = solution.y[:, -1]
        _2ad = np.zeros(idx)
        # Чтобы совпадали по длине с числом сечений, дописываем в начало недостающие
        n_conc = np.concatenate((_2ad, temp_n))
        concentration.append(n_conc)
    concentration.append(np.zeros_like(l))

    return np.array(concentration)
# Функция для построение распределений атомов от пучка
def plot_atoms_distributions(l, n_atoms):

    # Пересчет на подробную сетку, чтобы график был красивый
    n1 = []
    l_merged = []

    # Распределения от отдельных сечений
    figure1 = plt.figure(figsize=(13, 9))
    ax1 = figure1.add_subplot()

    for spine in ax1.spines.values():
        spine.set_linewidth(2)     # толщина
        spine.set_color('black')   # цвет

    for i in range(len(n_atoms)-2):

        for j in range(len(n_atoms[i])-1):
            if n_atoms[i][j] > n_atoms[i][j+1]:
                break

        l1 = np.linspace(l[j], l[-1], 50)
        interpolator = interp1d(l[j:], n_atoms[i][j:], kind = 'cubic', fill_value=0, bounds_error=False)
        n1.append(interpolator(l1))

        l_merged.append(l1)
        print(l_merged)


        color_section = random.choice(plt.cm.tab20.colors)
        ax1.plot(l_merged[i], n1[i], color = color_section, lw = 3)
        plt.show()
    exit()

        
    ax1.plot(l[j:], n_atoms[i][j:], color = color_section, lw = 3)
    ax1.plot(l[j], n_atoms[i][j], '*', color = color_section, markersize = 15)
    
    ax1.grid(True)
    ax1.set_xlim([-0.01*L, L + 0.01*L])
    xticks = ax1.get_xticks()
    ax1.set_xticklabels([f'{t:.1f}' for t in xticks], fontdict = fontdict_ticks)

    ax1.set_xlabel(fr'Координата вдоль пучка, $м$', fontdict_labels)
    ax1.set_ylabel(fr'Концентрация вторичных атомов, $м^{{-3}}$', fontdict_labels)
    
    # Создаем максимальное значение по y
    s = f"{np.max(np.max(n_atoms)):e}"
    mantissa_str, exp_str = s.split('e')
    _y_lim_max = (math.ceil(float(mantissa_str)*10)/10)*10**(float(exp_str))

    ax1.set_ylim([-0.01*_y_lim_max, _y_lim_max + 0.01*_y_lim_max])

    ticks = np.arange(0, _y_lim_max/10**float(exp_str) + _y_lim_max/10**float(exp_str)/10, _y_lim_max/10**float(exp_str)/10)
    ax1.set_yticks(ticks*10**float(exp_str))

    tick_labels = np.array([f'{t:.1f}' for t in ticks])
    ax1.set_yticklabels(tick_labels, fontdict = fontdict_ticks)

    ax1.text(-0.01, 1.02,
            fr'$10^{{{float(exp_str):.0f}}}$',
            fontsize = fontdict_ticks['fontsize'] + 2,
            transform = ax1.transAxes)
    
    plt.tight_layout()

    plt.show()
    exit()
    # Суммарное
    figure2 = plt.figure(figsize=(13, 9))
    ax2 = figure2.add_subplot()

    for spine in ax2.spines.values():
        spine.set_linewidth(2)     # толщина
        spine.set_color('black')   # цвет


    ax2.plot(l, n_atoms.sum(axis = 0))

    plt.show()