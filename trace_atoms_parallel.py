import multiprocessing as mp
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import auxillary_recombination

from initiate_constants import *

# ==========================================================================
## Трассировка частиц, правая часть и ивенты
# Правая часть
def moving(t, part_prop):
    return [part_prop[1], 0, part_prop[3], 0, part_prop[5], 0]

# Ивенты
# ------------------------------------
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
# ------------------------------------
# Генератор заданий
def _trace_atoms_pool(args):

    i, j, n, N, x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current, t_max, recombination_probability = args
    
    # Начальные условия
    vars0 = [x0, vx0_current, y_cell_current, vy0_current, z_cell_current, vz0_current]
    
    # Ивенты
    events = [leave_neutr, hit_left, hit_right, hit_bottom, hit_top, hit_entrance]
    
    # Время, по которому расчет
    t0 = 0
    # Счетчик столкновений
    collisions = 0
    # Координаты и прочее для столкновений
    collision_marks = []
    # Заготовки под накопление траектории
    t_history = [0]    
    prop_history = [np.array(vars0).copy()]
    recombined = False
    
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
            return recombined, i, j, n, t_history, prop_history, collisions, collision_marks

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
            return recombined, i, j, n, t_history, prop_history, collisions, collision_marks

        # Рекомбинация, если есть, заканчиваем, если нет, то считаем дальше
        if np.random.rand() <= recombination_probability:
            # Заканичваем расчет, ставим флаг
            recombined = True
            return recombined, i, j, n, t_history, prop_history, collisions, collision_marks
            
        else:
            # Если рекомбинации не было, что продолжаем расчет
            # Переписываем координаты, где произошло столкновение
            vars0 = solution.y_events[earliest_idx][0]
            t0 = earliest_t
            collisions += 1
            collision_marks.append(vars0.copy())
            
            # Переписываем начальные скорости по диффузному отражению и обрабатываем возможные события
            new_v = auxillary_recombination.diffuse_reflection(vars0, earliest_idx, 'atoms')
            vars0[1] = new_v[0]
            vars0[3] = new_v[1]
            vars0[5] = new_v[2]
        
    return recombined, i, j, n, t_history, prop_history, collisions, collision_marks
    
# Организатор
def trace_atoms(L, lk, a_grid, b_grid, N, vx0, y_cell, vy0, z_cell, vz0, recombination_probability):
    print('== Трассировка атомов ==')
    print(f'Начата трассировка сечения {lk} м ...')
    n_workers = os.cpu_count()
    t_max = L/vx0*100
    # print(t_max)
    # exit()
    tasks = []
    # Пул задач
    for i in range(len(a_grid)):
            for j in range(len(b_grid)):
                for n in range(N):
                    tasks.append((i, j, n, N, lk, vx0[i, j, n], y_cell[i, j, n], vy0[i, j, n], z_cell[i, j, n], vz0[i, j, n], t_max, recombination_probability))
    chunksize = max(1, len(tasks)//(n_workers*4))
    # Сам расчет
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_trace_atoms_pool, tasks, chunksize=chunksize))

    # Подготовка переменных под распаковку результата
    is_recombined = np.zeros((len(a_grid), len(b_grid), N))
    time = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    particles_trajectoies = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    collisions_count = np.zeros((len(a_grid), len(b_grid), N))
    collision_points = np.empty((len(a_grid), len(b_grid), N), dtype = object)
    

    # Распаковка результата
    for recombined, i, j, n, t_history, prop_history, collisions, collision_marks in results:
        time[i, j, n] = np.array(t_history)
        particles_trajectoies[i, j, n] = np.array(prop_history)
        collisions_count[i, j, n] = collisions
        collision_points[i, j, n] = np.array(collision_marks)
        is_recombined[i, j, n] = int(recombined)
    
    return time, particles_trajectoies, collisions_count, collision_points, is_recombined

# Правая часть
def moving(t, part_prop):
    return [part_prop[1], 0, part_prop[3], 0, part_prop[5], 0]