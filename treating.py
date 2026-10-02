import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from initiate_constants import *

filepath = 'output/mean_vals.txt'

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

sections = read_section_data(filepath)
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
dl = l[1] - l[0]
# print(dl)
D = 1/3*v_drift*a*k

# print(len(D))

S = np.zeros_like(l)
S[0] = 1e15

# Общая функция для расчета концентрации
def calculate_diffusion_flow(l, D, S, type = 'gas'):
    if type == 'gas':
        Q = 3.3#16*1e-3*133.3
        inflow = Q/1.38e-23/300/0.16/0.47
        D = 183.5

        def diffusion(t, n):
            dndt = np.zeros_like(n)
            
            for i in range(1, len(dndt) - 1):
                d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
                dndt[i] = D*d2n_dx2

            dndt[0] = 2*D*(n[1] - n[0])/dl**2 + 2*inflow/dl
            dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2

            return dndt

        t0 = 0
        t_max = 5e-2
        n_IC = np.zeros(len(l))
        solution = solve_ivp(fun=diffusion,
                            t_span=(t0, t_max),
                            y0=n_IC,
                            method='RK45')
        

    elif type == 'dissociation':

        def diffusion(t, n):

            dndt = np.zeros_like(l)

            for i in range(1, len(dndt) - 1):
                d2n_dx2 = (n[i-1] - 2*n[i] + n[i+1])/dl**2
                dndt[i] = D*d2n_dx2 + S[i]

            # dndt[0] = D[0]*(n[2] - 2*n[1] + n[0])/dl**2 + S[0]
            dndt[0] = D*(n[1] - n[0])/dl**2 + S[0]
            dndt[-1] = D*(n[-2] - 2*n[-1])/dl**2 + S[-1]
            return dndt

        t0 = 0
        t_max = 5e-2
        n_IC = np.zeros(len(l))

        solution = solve_ivp(fun=diffusion,
                    t_span=(t0, t_max),
                    y0=n_IC,
                    method='RK45')
        
    return solution

# Функция для расчета S от пучка
def generation_by_beam():

    # Нвы выходе будем массив длиной, как количество сечений
    return 1


# Функция для расчета распределений
def distributions():
    concentration = []
    # Цикл по сечениям
    for idx in range(len(l)):
        # Эти значения мы будем отправлять в считалку распределений, чтобы она думала, что мы считаем такой короткий участок
        l2set = l[idx:]
        D2set = D[idx:]
        S2set = np.zeros(len(l2set))
        S2set[0] = S[idx]
        # Вызывем решатель
        # solution = calculate_diffusion_flow(l2set, D2set, S2set, type = 'gas')
        # # Массив концентраций от конкретного сечения, длиной len(l) - idx, надо в начало дописать нулей
        # temp_n = solution.y[:, -1]


        temp_n = np.ones(len(l2set))

        _2ad = np.zeros(idx)

        n_conc = np.concatenate((_2ad, temp_n))

        concentration.append(n_conc)


        

        if idx == 2:
            exit()
    return concentration



distributions()














exit()
# Сначала считаем чистое распределение
solution = calculate_diffusion_flow(D, type = 'gas')
# Считаем генерацию частиц

# Ситаем распределения от каждого сечения

time = solution.t
n = solution.y[:, -1]
plt.plot(l, solution.y[:, -1])
plt.show()
print((solution.y[:, -1]).mean())
















