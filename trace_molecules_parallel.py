import multiprocessing as mp
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import auxillary_recombination

from initiate_constants import *

# Организатор
def trace_molecules(L, N, x0, vx0, y0, vy0, z0, vz0):
    print('== Трассировка молекул ==')
    n_workers = os.cpu_count()
    t_max = 100e-3 #L/np.mean(vx0)*5
    tasks = []
    N = len(x0)
    # Пул задач
    for i in range(len(x0)):
        tasks.append((N, x0[i], vx0[i], y0[i], vy0[i], z0[i], vz0[i], t_max))

    chunksize = max(1, len(tasks)//(n_workers*4))
        
    # Сам расчет
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_trace_molecules_pool, tasks, chunksize=chunksize))

    # Подготовка переменных под распаковку результата
    # Время
    time = np.empty(N, dtype = object)
    # Траектории
    particles_trajectoies = np.empty(N, dtype = object)
    # Число столкновений
    collisions_count = np.zeros(N)
    # Распаковка результата
    for i in range(N):
        time[i] = np.array(results[i][0])
        particles_trajectoies[i] = np.array(results[i][1])
        collisions_count[i] = results[i][2]
    
    return time, particles_trajectoies, collisions_count

# Генератор заданий
def _trace_molecules_pool(args):
    # Распаковка аргументов
    N, x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current, t_max = args
    # Начальные условия
    vars0 = [x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current]
    # Ивенты
    events = [leave_neutr, hit_left, hit_right, hit_bottom, hit_top, hit_entrance]
    # Время, по которому расчет
    t0 = 0
    # Счетчик столкновений
    collisions = 0
    # Заготовки под накопление траектории
    t_history = [0]    
    prop_history = [np.array(vars0).copy()]
        
    # Так как используем ивенты, будем много перезапускать
    while t0 < t_max:
        # Считаем движение, так как есть ивенты счет остановится при соударении со стенкой
        
        solution = solve_ivp(
                            fun = moving,
                            t_span = (t0, t_max),
                            y0 = vars0,
                            method = 'RK45',
                            events = events,
                            max_step = t_max/10,
                            rtol=1e-8, atol=1e-10)

        # Накапливаетс ятраектория
        for i_history in range(1, len(solution.t)):
            t_history.append(solution.t[i_history].copy())
            prop_history.append(solution.y[:, i_history].copy())            

        # Если просто вылет из нейтрализатора
        if len(solution.t_events[0]) > 0:
            return t_history, prop_history, collisions

        # Первое столкновение со стенкой
        # Время, когда случилось соударение
        earliest_t = None
        # Индекс стенки, соответствцует индексу ивента в списке ивентов
        earliest_idx = None
        # Проходим по всме ивентам и ищем, какой из них сработал
        for idx in range(1, len(solution.t_events)):
            ev = solution.t_events[idx]
            if len(ev) > 0 and (earliest_t is None or ev[0] < earliest_t):
                earliest_t = ev[0]
                earliest_idx = idx

        if earliest_idx is None:
            # Ничего не произошло — таймаут? за все выделенное время остался в нейтрализаторе и не ударился, а остался в воздухе висеть
            return t_history, prop_history, collisions

        # Если рекомбинации не было, что продолжаем расчет
        # Переписываем координаты, где произошло столкновение
        vars0 = solution.y_events[earliest_idx][0]
        t0 = earliest_t
        collisions += 1
                
        # Переписываем начальные скорости по диффузному отражению и обрабатываем возможные события
        new_v = auxillary_recombination.diffuse_reflection(vars0, earliest_idx, 'molecules')
        vars0[1] = new_v[0]
        vars0[3] = new_v[1]
        vars0[5] = new_v[2]
        
    return t_history, prop_history, collisions

# ======================================================================
# Ивенты
def leave_neutr(t, part_prop):
    return part_prop[0] - L
leave_neutr.terminal = True
leave_neutr.direction = 1

def hit_left(t, part_prop):
    return part_prop[2] + a/2
hit_left.terminal = True
hit_left.direction = -1

def hit_right(t, part_prop):
    return part_prop[2] - a/2
hit_right.terminal = True
hit_right.direction = 1

def hit_bottom(t, part_prop):
    return part_prop[4] + b/2
hit_bottom.terminal = True
hit_bottom.direction = -1

def hit_top(t, part_prop):
    return part_prop[4] - b/2
hit_top.terminal = True
hit_top.direction = 1

def hit_entrance(t, part_prop):
    return part_prop[0]
hit_entrance.terminal = True
hit_entrance.direction = -1

# ======================================================================
# Правая часть
def moving(t, part_prop):
    return [part_prop[1], 0, part_prop[3], 0, part_prop[5], 0]